"""Interface every game should implement so solvers stay game-agnostic.

Feel free to change this interface — it's a starting point, not a requirement.
"""

from abc import ABC, abstractmethod


class Game(ABC):
    """Two-player zero-sum extensive-form game.

    States must be hashable (e.g. frozen dataclasses) so evaluators can cache them.
    """

    #: Chips per big blind, used to report results in milli-big-blinds (mbb).
    #: Ante-only games (Kuhn, Leduc) treat the 1-chip ante as the big blind.
    big_blind: float = 1.0

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
