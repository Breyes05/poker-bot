"""Heads-up push/fold No-Limit Hold'em.

Both players start with ``stack`` big blinds. Player 0 is the small blind (posts
0.5 BB) and acts first; player 1 is the big blind (posts 1 BB).

* Small blind: **fold** (lose 0.5) or **push** all-in.
* Big blind, facing a push: **fold** (lose 1) or **call**.
* After a call the board is dealt and the better hand wins ``stack`` BB.

This is the standard model for short-stacked play: below roughly 15 BB, real
heads-up strategies are close to push-or-fold.

Abstractions (both lossless for this game):

* Hands are dealt as one of the 169 hand classes (AKs, 72o, ...), with
  probabilities that account for card removal. Suits can't matter preflop
  except through "suited or not".
* The board isn't dealt in the tree. A call ends the game with the
  *expected* payoff ``stack * (2 * equity - 1)``, using the precomputed
  equity table. CFR only needs expected values, so this is exact.

That makes the tree tiny: 28,561 deals, each with three possible endings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.games.base import Game
from pokerbot.poker.equity import deal_probabilities, load_equity_table
from pokerbot.poker.hand_classes import CLASS_NAMES, NUM_CLASSES

FOLD, PUSH, CALL = "f", "p", "c"
SMALL_BLIND = 0.5
TERMINAL_HISTORIES = {"f", "pf", "pc"}


@dataclass(frozen=True)
class PushFoldState:
    hands: tuple[int, int] | None  # hand-class index for (small blind, big blind)
    history: str = ""


class PushFoldHoldem(Game):
    big_blind = 1.0  # payoffs are in big blinds

    def __init__(self, stack: float = 10.0, equity: np.ndarray | None = None):
        if stack <= self.big_blind:
            raise ValueError("stack must be larger than the big blind")
        self.stack = stack
        self.equity = load_equity_table() if equity is None else equity
        probs = deal_probabilities()
        self._deals = [
            ((a, b), float(probs[a, b]))
            for a in range(NUM_CLASSES)
            for b in range(NUM_CLASSES)
            if probs[a, b] > 0
        ]

    def initial_state(self) -> PushFoldState:
        return PushFoldState(hands=None)

    def is_terminal(self, state: PushFoldState) -> bool:
        return state.history in TERMINAL_HISTORIES

    def is_chance(self, state: PushFoldState) -> bool:
        return state.hands is None

    def current_player(self, state: PushFoldState) -> int:
        return len(state.history)  # "" -> small blind (0), "p" -> big blind (1)

    def legal_actions(self, state: PushFoldState) -> list[str]:
        return [FOLD, PUSH] if state.history == "" else [FOLD, CALL]

    def next_state(self, state: PushFoldState, action) -> PushFoldState:
        if self.is_chance(state):
            return PushFoldState(hands=action)
        return PushFoldState(hands=state.hands, history=state.history + action)

    def chance_outcomes(self, state: PushFoldState) -> list[tuple[object, float]]:
        return self._deals

    def payoff(self, state: PushFoldState, player: int) -> float:
        if state.history == "f":
            p0 = -SMALL_BLIND
        elif state.history == "pf":
            p0 = self.big_blind
        else:  # "pc": all-in, expected value over the board
            sb, bb = state.hands
            p0 = self.stack * (2 * self.equity[sb, bb] - 1)
        return p0 if player == 0 else -p0

    def info_set_key(self, state: PushFoldState) -> str:
        player = self.current_player(state)
        return f"{CLASS_NAMES[state.hands[player]]}|{state.history}"
