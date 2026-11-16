"""Interface every game should implement so solvers stay game-agnostic.

Feel free to change this interface — it's a starting point, not a requirement.
"""

from abc import ABC, abstractmethod


class Game(ABC):
    """Two-player zero-sum extensive-form game."""

    @abstractmethod
    def initial_state(self):
        """Return the root state of the game tree."""

    @abstractmethod
    def is_terminal(self, state) -> bool: ...

    @abstractmethod
    def is_chance(self, state) -> bool: ...

    @abstractmethod
    def current_player(self, state) -> int: ...

    @abstractmethod
    def legal_actions(self, state) -> list: ...

    @abstractmethod
    def next_state(self, state, action): ...

    @abstractmethod
    def chance_outcomes(self, state) -> list[tuple[object, float]]:
        """(outcome, probability) pairs for a chance node."""

    @abstractmethod
    def payoff(self, state, player: int) -> float:
        """Utility for `player` at a terminal state."""

    @abstractmethod
    def info_set_key(self, state) -> str:
        """What the current player can observe — states with the same key are indistinguishable."""
