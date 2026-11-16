"""Monte Carlo CFR (e.g. external sampling or outcome sampling)."""

from pokerbot.games.base import Game


class MCCFRSolver:
    def __init__(self, game: Game, seed: int | None = None):
        self.game = game
        self.seed = seed

    def train(self, iterations: int) -> None:
        raise NotImplementedError

    def average_strategy(self) -> dict:
        raise NotImplementedError
