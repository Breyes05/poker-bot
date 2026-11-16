"""External-sampling Monte Carlo CFR (Lanctot et al., 2009).

Vanilla CFR walks the entire game tree every iteration. External sampling walks
only part of it: for one "traverser" at a time, it

* explores **every** action at the traverser's own decisions,
* **samples one** action at the opponent's decisions (from the opponent's
  current strategy), and
* **samples one** outcome at chance nodes (the deal, the board card).

Because the opponent's and chance's choices are sampled in proportion to their
probabilities, the sampled regret is an unbiased estimate of the counterfactual
regret, and the reach weights that vanilla CFR multiplies in cancel out. So the
update is just ``regret += value(action) - value(node)``.

Each iteration runs one traversal per player. Iterations are much cheaper than
vanilla CFR's (they touch a small slice of the tree), noisier, and in big games
far faster per unit of wall-clock time.

The average strategy is accumulated at the *opponent's* nodes during a
traversal: the opponent's current strategy is added each time play passes
through one of its info sets, which happens in proportion to how often play
reaches it.
"""

from __future__ import annotations

import random

from pokerbot.games.base import Game
from pokerbot.solvers.cfr import InfoSetNode
from pokerbot.solvers.regret_matching import average_strategy_helper, regret_matching


class MCCFRSolver:
    def __init__(self, game: Game, seed: int | None = None):
        self.game = game
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes: dict[str, InfoSetNode] = {}
        self.iterations = 0

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            for traverser in (0, 1):
                self._traverse(self.game.initial_state(), traverser)
            self.iterations += 1

    def average_strategy(self) -> dict[str, list[float]]:
        """Map info set key -> action probabilities. This is what converges to Nash."""
        return {key: average_strategy_helper(node.strategy_sum) for key, node in self.nodes.items()}

    def actions(self, key: str) -> list:
        """The legal actions at an info set, in the same order as its probabilities."""
        return self.nodes[key].actions

    def _node(self, key: str, actions: list) -> InfoSetNode:
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = InfoSetNode(actions)
        return node

    def _sample(self, items: list, weights: list[float]):
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _traverse(self, state, traverser: int) -> float:
        """Return a sampled estimate of the traverser's payoff from `state`."""
        game = self.game

        if game.is_terminal(state):
            return game.payoff(state, traverser)

        if game.is_chance(state):
            outcomes, probs = zip(*game.chance_outcomes(state))
            return self._traverse(game.next_state(state, self._sample(outcomes, probs)), traverser)

        actions = game.legal_actions(state)
        node = self._node(game.info_set_key(state), actions)
        strategy = regret_matching(node.regret_sum)

        if game.current_player(state) != traverser:
            # Opponent: update their average strategy, then follow one sampled action.
            for i, prob in enumerate(strategy):
                node.strategy_sum[i] += prob
            action = self._sample(actions, strategy)
            return self._traverse(game.next_state(state, action), traverser)

        # Traverser: try every action and update regrets.
        action_values = [self._traverse(game.next_state(state, a), traverser) for a in actions]
        node_value = sum(p * v for p, v in zip(strategy, action_values))
        for i, value in enumerate(action_values):
            node.regret_sum[i] += value - node_value
        return node_value
