"""Vanilla counterfactual regret minimization (Zinkevich et al., 2007).

Each iteration walks the entire game tree once. At every decision node the
acting player's strategy comes from regret matching on that info set's
accumulated regrets - the same rule as the M0 rock-paper-scissors notebook,
just applied separately at every info set.

The two quantities that make this "counterfactual":

* ``reach[p]`` is the probability that player ``p``'s own choices lead to the
  current node. ``reach[CHANCE]`` is the probability of the chance outcomes
  (the deal) so far.
* An info set's regrets are weighted by the *counterfactual reach*: the
  probability of getting here if the acting player had tried to reach it,
  i.e. the product of everyone else's reach (opponent and chance).

The average strategy is weighted by the acting player's *own* reach, because a
strategy only matters in proportion to how often that player actually plays it.
"""

from __future__ import annotations

from pokerbot.games.base import Game
from pokerbot.solvers.regret_matching import average_strategy_helper, regret_matching

CHANCE = 2  # index of the chance "player" in the reach-probability list


class InfoSetNode:
    """Running totals for one information set."""

    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions: list):
        self.actions = list(actions)
        self.regret_sum = [0.0] * len(actions)
        self.strategy_sum = [0.0] * len(actions)


class CFRSolver:
    def __init__(self, game: Game):
        self.game = game
        self.nodes: dict[str, InfoSetNode] = {}
        self.iterations = 0
        # Current-iteration strategies, cached so that every visit to the same
        # info set within one iteration uses the same strategy even though its
        # regrets are being updated during the traversal.
        self._iteration_strategy: dict[str, list[float]] = {}

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            self._iteration_strategy = {}
            self._cfr(self.game.initial_state(), [1.0, 1.0, 1.0])
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

    def _current_strategy(self, key: str, node: InfoSetNode) -> list[float]:
        strategy = self._iteration_strategy.get(key)
        if strategy is None:
            strategy = self._iteration_strategy[key] = regret_matching(node.regret_sum)
        return strategy

    def _cfr(self, state, reach: list[float]) -> float:
        """Return the expected payoff for player 0 from `state` under the current strategies.

        Payoffs are tracked from player 0's point of view throughout; because the
        game is zero-sum, player 1's payoff is just the negative.
        """
        game = self.game

        if game.is_terminal(state):
            return game.payoff(state, 0)

        if game.is_chance(state):
            value = 0.0
            for outcome, prob in game.chance_outcomes(state):
                child_reach = [reach[0], reach[1], reach[CHANCE] * prob]
                value += prob * self._cfr(game.next_state(state, outcome), child_reach)
            return value

        player = game.current_player(state)
        opponent = 1 - player
        key = game.info_set_key(state)
        actions = game.legal_actions(state)
        node = self._node(key, actions)
        strategy = self._current_strategy(key, node)

        # Recurse into each action, scaling the acting player's reach by the
        # probability they take that action.
        action_values = []
        for action, prob in zip(actions, strategy):
            child_reach = list(reach)
            child_reach[player] *= prob
            action_values.append(self._cfr(game.next_state(state, action), child_reach))

        node_value = sum(p * v for p, v in zip(strategy, action_values))

        # Regret = how much better each action would have done than the current
        # mix, from the acting player's perspective, weighted by the probability
        # that the opponent and chance bring play to this info set.
        sign = 1.0 if player == 0 else -1.0
        counterfactual_reach = reach[opponent] * reach[CHANCE]
        for i, value in enumerate(action_values):
            node.regret_sum[i] += counterfactual_reach * sign * (value - node_value)
            node.strategy_sum[i] += reach[player] * strategy[i]

        return node_value
