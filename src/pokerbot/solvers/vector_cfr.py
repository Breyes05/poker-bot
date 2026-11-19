"""CFR and CFR+ on a compiled ``GameTree`` (vectorized with numpy).

``VectorCFRSolver(game, variant="cfr")`` computes exactly what the reference
``CFRSolver`` computes (the tests check they agree to ~1e-12), but each
iteration is a few numpy operations per tree level instead of a Python
recursion through the ``Game`` API.

One iteration:

1. strategy = regret matching on every info set's regrets (one matrix op)
2. reach probabilities, top down
3. node values for player 0, bottom up
4. for every edge out of a decision node, regret += counterfactual reach *
   (value(child) - value(node)), summed into its info set with ``bincount``

``variant="cfr+"`` (Tammelin, 2014) changes three things, and converges much
faster in practice:

* regrets are floored at zero after every update, so an action that was bad
  early on doesn't need many iterations to "climb back" once it becomes good;
* the players update in turn (player 0, then player 1 against player 0's new
  strategy) instead of simultaneously;
* the average strategy weights iteration t by t, so early, poor strategies
  fade out of the average.
"""

from __future__ import annotations

import numpy as np

from pokerbot.games.base import Game
from pokerbot.solvers.tree import BY_CHANCE, BY_P0, BY_P1, GameTree

VARIANTS = ("cfr", "cfr+")


class VectorCFRSolver:
    def __init__(self, game: Game, variant: str = "cfr", tree: GameTree | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.game = game
        self.variant = variant
        self.tree = tree if tree is not None else GameTree(game)
        shape = (self.tree.num_infosets, self.tree.max_actions)
        self.regret = np.zeros(shape)
        self.strategy_sum = np.zeros(shape)
        self.iterations = 0

        # Edges out of decision nodes, split by the acting player.
        t = self.tree
        self._edges = {}
        for player, chooser in ((0, BY_P0), (1, BY_P1)):
            child = np.flatnonzero(t.chooser == chooser)
            child = child[child > 0]
            self._edges[player] = (child, t.parent[child], t.edge_slot[child])

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            t = self.iterations + 1
            if self.variant == "cfr":
                self._update(players=(0, 1), avg_weight=1.0)
            else:
                self._update(players=(0,), avg_weight=t)
                self._update(players=(1,), avg_weight=t)
            self.iterations += 1

    def current_strategy(self) -> np.ndarray:
        return self.tree.regret_matching(self.regret)

    def average_strategy_matrix(self) -> np.ndarray:
        return self.tree.normalize(self.strategy_sum)

    def average_strategy(self) -> dict[str, list[float]]:
        """Map info set key -> action probabilities. This is what converges to Nash."""
        return self.tree.strategy_dict(self.average_strategy_matrix())

    def actions(self, key: str) -> list:
        return self.tree.infoset_actions[self.tree.key_index[key]]

    def exploitability(self) -> float:
        """Exploitability of the average strategy, in chips per hand."""
        return self.tree.exploitability(self.average_strategy_matrix())

    def _update(self, players: tuple[int, ...], avg_weight: float) -> None:
        tree = self.tree
        strategy = self.current_strategy()
        edge = tree.edge_probs(strategy)
        reach = tree.reach(edge)
        values = tree.values(edge)
        slots = self.regret.size

        for player in players:
            child, node, slot = self._edges[player]
            sign = 1.0 if player == 0 else -1.0
            counterfactual = reach[1 - player, node] * reach[BY_CHANCE, node]
            delta = counterfactual * sign * (values[child] - values[node])
            self.regret += np.bincount(slot, weights=delta, minlength=slots).reshape(
                self.regret.shape
            )
            own = reach[player, node] * edge[child]  # own reach * P(action)
            self.strategy_sum += avg_weight * np.bincount(
                slot, weights=own, minlength=slots
            ).reshape(self.regret.shape)

        if self.variant == "cfr+":
            np.maximum(self.regret, 0, out=self.regret)
