"""Push/fold strategies by stack depth: precomputed, cached, or solved on demand.

Integer stacks ship precomputed in ``strategies/pushfold.json``. Any other stack
(e.g. 7.5 BB) is solved live with vectorized CFR+ and cached.

Solving a new stack doesn't rebuild the game tree. The tree's shape is the same
at every stack depth; only the payoffs change, and they change linearly:
fold payoffs (-0.5, +1) don't depend on the stack, and an all-in pays
``stack * (2 * equity - 1)``. So two trees, built once, give the exact payoffs
for any stack by linear interpolation.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict

from pokerbot.eval.exploitability import to_mbb
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.poker.hand_classes import CLASS_NAMES, range_fraction
from pokerbot.solvers.tree import GameTree
from pokerbot.solvers.vector_cfr import VectorCFRSolver

MIN_STACK, MAX_STACK = 1.5, 30.0
_BASE_STACKS = (10.0, 20.0)


class PushFoldSolver:
    def __init__(self, precomputed: dict, iterations: int = 300, cache_size: int = 64):
        self.iterations = iterations
        self.precomputed = {float(k): v for k, v in precomputed["stacks"].items()}
        self.precomputed_iterations = precomputed["iterations"]
        self._cache: OrderedDict[float, dict] = OrderedDict()
        self._cache_size = cache_size
        self._trees: tuple[GameTree, GameTree] | None = None
        self._lock = threading.Lock()  # one solve at a time on a small server

    @staticmethod
    def normalize(stack: float) -> float:
        if not MIN_STACK <= stack <= MAX_STACK:
            raise ValueError(f"stack must be between {MIN_STACK:g} and {MAX_STACK:g} BB")
        return round(stack, 1)

    def strategy(self, stack: float) -> dict[str, list[float]]:
        return self.solve(stack)["strategy"]

    def solve(self, stack: float) -> dict:
        """{stack, source, iterations, exploitability_mbb, seconds, strategy}."""
        stack = self.normalize(stack)
        if stack in self.precomputed:
            entry = self.precomputed[stack]
            return {
                "stack": stack,
                "source": "precomputed",
                "iterations": self.precomputed_iterations,
                "exploitability_mbb": entry["exploitability_mbb"],
                "seconds": 0.0,
                "strategy": entry["strategy"],
            }
        with self._lock:
            if stack in self._cache:
                self._cache.move_to_end(stack)
                return {**self._cache[stack], "source": "cached"}
            result = self._solve_now(stack)
            self._cache[stack] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return result

    def tree_for(self, stack: float) -> GameTree:
        """A game tree with exact payoffs for `stack`, interpolated from two base trees."""
        if self._trees is None:
            self._trees = tuple(GameTree(PushFoldHoldem(s)) for s in _BASE_STACKS)
        (s0, s1), (t0, t1) = _BASE_STACKS, self._trees
        tree = copy.copy(t0)  # shares the structure arrays; only payoffs are replaced
        tree.payoff = t0.payoff + (stack - s0) * (t1.payoff - t0.payoff) / (s1 - s0)
        tree.game = None
        return tree

    def _solve_now(self, stack: float) -> dict:
        start = time.perf_counter()
        tree = self.tree_for(stack)
        solver = VectorCFRSolver(tree.game, "cfr+", tree=tree)
        solver.train(self.iterations)
        strategy = {k: [round(p, 4) for p in v] for k, v in solver.average_strategy().items()}
        return {
            "stack": stack,
            "source": "solved",
            "iterations": self.iterations,
            "exploitability_mbb": to_mbb(PushFoldHoldem, solver.exploitability()),
            "seconds": round(time.perf_counter() - start, 3),
            "strategy": strategy,
        }


def charts(strategy: dict[str, list[float]]) -> dict:
    """Push and call probabilities in hand-class order, plus range sizes."""
    push = [strategy[f"{name}|"][1] for name in CLASS_NAMES]
    call = [strategy[f"{name}|p"][1] for name in CLASS_NAMES]
    return {
        "push": push,
        "call": call,
        "push_pct": range_fraction(push),
        "call_pct": range_fraction(call),
    }
