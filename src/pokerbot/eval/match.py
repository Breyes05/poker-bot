"""Head-to-head play between two strategies."""

from pokerbot.games.base import Game


def play_match(game: Game, strategy_a: dict, strategy_b: dict, hands: int, seed: int | None = None):
    """Return mean winnings for A (and ideally a confidence interval)."""
    raise NotImplementedError
