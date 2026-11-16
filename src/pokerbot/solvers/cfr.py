"""Vanilla counterfactual regret minimization."""

from pokerbot.games.base import Game


class CFRSolver:
    def __init__(self, game: Game):
        self.game = game

    def train(self, iterations: int) -> None:
        raise NotImplementedError

    def average_strategy(self) -> dict:
        """Map info set key -> action probabilities. This is what converges to Nash."""
        raise NotImplementedError
