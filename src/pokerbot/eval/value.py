"""Expected value of a fixed strategy profile."""

from __future__ import annotations

from pokerbot.games.base import Game


def expected_value(game: Game, strategy: dict[str, list[float]], player: int = 0) -> float:
    """Expected payoff for `player` when both players follow `strategy`.

    `strategy` maps info set key -> action probabilities, in the order returned
    by `game.legal_actions`. Info sets missing from `strategy` are played uniformly.
    """
    return _value(game, game.initial_state(), strategy, player)


def strategy_probs(strategy: dict[str, list[float]], key: str, num_actions: int) -> list[float]:
    """Action probabilities at an info set, defaulting to uniform if it isn't in `strategy`."""
    return strategy.get(key, [1 / num_actions] * num_actions)


def _value(game: Game, state, strategy: dict[str, list[float]], player: int) -> float:
    if game.is_terminal(state):
        return game.payoff(state, player)
    if game.is_chance(state):
        return sum(
            prob * _value(game, game.next_state(state, outcome), strategy, player)
            for outcome, prob in game.chance_outcomes(state)
        )
    actions = game.legal_actions(state)
    probs = strategy_probs(strategy, game.info_set_key(state), len(actions))
    return sum(
        p * _value(game, game.next_state(state, a), strategy, player)
        for a, p in zip(actions, probs)
        if p > 0
    )
