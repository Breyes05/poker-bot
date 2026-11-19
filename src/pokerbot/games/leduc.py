"""Leduc Hold'em: 6-card deck (2 suits x J/Q/K), one private card, one public card, two betting rounds.

Rules (Southey et al., 2005 - "Bayes' Bluff")
---------------------------------------------
* Deck: J, J, Q, Q, K, K. Each player antes 1 chip and gets one private card.
* Round 1: a betting round with a fixed bet size of 2.
* A public "board" card is dealt.
* Round 2: a betting round with a fixed bet size of 4.
* Each round allows at most 2 raises (a bet, then one re-raise). Player 0 acts
  first in both rounds.
* Showdown: a private card that pairs the board wins; otherwise the higher card
  wins; equal ranks split the pot.

Actions
-------
* ``"f"`` fold  - only legal when facing a bet.
* ``"c"`` call  - check when there is nothing to call, otherwise call.
* ``"r"`` raise - bet when there is nothing to call, otherwise raise; only legal
  while the round has fewer than 2 raises.

State representation
--------------------
``(cards, board, rounds)``: ``cards`` is ``()`` before the deal and
``(card_p0, card_p1)`` after it, with cards 0-5 and rank = card // 2 (J, Q, K).
``board`` is the public card or ``None``. ``rounds`` holds one action string per
betting round, e.g. ``("rc", "cr")``.

Info sets use card *ranks* only. Suits never matter in Leduc, so strategies are
identical across suits; dropping them shrinks the game from 936 info sets (the
suit-specific count) to 288 without changing its equilibria.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot.games.base import Game

RANK_NAMES = "JQK"
FOLD, CALL, RAISE = "f", "c", "r"
ANTE = 1
BET_SIZES = (2, 4)  # round 1, round 2
MAX_RAISES = 2
NUM_CARDS = 6


def rank(card: int) -> int:
    return card // 2


def round_complete(history: str) -> bool:
    """A betting round ends with a call after at least two actions: cc, rc, crc, rrc, crrc."""
    return len(history) >= 2 and history[-1] == CALL


@dataclass(frozen=True)
class LeducState:
    cards: tuple[int, ...]
    board: int | None
    rounds: tuple[str, ...]


class LeducHoldem(Game):
    """Two-player limit Leduc Hold'em. Payoffs are zero-sum."""

    def initial_state(self) -> LeducState:
        return LeducState(cards=(), board=None, rounds=("",))

    def is_terminal(self, state: LeducState) -> bool:
        last = state.rounds[-1]
        return last.endswith(FOLD) or (len(state.rounds) == 2 and round_complete(last))

    def is_chance(self, state: LeducState) -> bool:
        if not state.cards:
            return True  # private cards not dealt yet
        return state.board is None and round_complete(state.rounds[0])

    def current_player(self, state: LeducState) -> int:
        return len(state.rounds[-1]) % 2

    def legal_actions(self, state: LeducState) -> list[str]:
        history = state.rounds[-1]
        actions = []
        if self._facing_bet(history):
            actions.append(FOLD)
        actions.append(CALL)
        if history.count(RAISE) < MAX_RAISES:
            actions.append(RAISE)
        return actions

    def next_state(self, state: LeducState, action) -> LeducState:
        if not state.cards:
            return LeducState(cards=action, board=None, rounds=("",))
        if self.is_chance(state):
            return LeducState(cards=state.cards, board=action, rounds=(state.rounds[0], ""))
        rounds = state.rounds[:-1] + (state.rounds[-1] + action,)
        return LeducState(cards=state.cards, board=state.board, rounds=rounds)

    def chance_outcomes(self, state: LeducState) -> list[tuple[object, float]]:
        if not state.cards:
            deals = [(a, b) for a in range(NUM_CARDS) for b in range(NUM_CARDS) if a != b]
            return [(deal, 1 / len(deals)) for deal in deals]
        remaining = [c for c in range(NUM_CARDS) if c not in state.cards]
        return [(card, 1 / len(remaining)) for card in remaining]

    def payoff(self, state: LeducState, player: int) -> float:
        contributions = self._contributions(state)
        last = state.rounds[-1]
        if last.endswith(FOLD):
            folder = (len(last) - 1) % 2
            winner = 1 - folder
        else:
            strengths = [self._hand_strength(state.cards[p], state.board) for p in (0, 1)]
            if strengths[0] == strengths[1]:
                return 0.0
            winner = 0 if strengths[0] > strengths[1] else 1
        # The winner wins whatever the loser put in the pot.
        loser = 1 - winner
        return contributions[loser] if player == winner else -contributions[loser]

    def info_set_key(self, state: LeducState) -> str:
        # Own card rank (+ board rank once dealt) | betting history, rounds split by "/".
        # Examples: "K|cr" (round 1), "QK|rc/r" (holding Q, board K, round 2).
        own = RANK_NAMES[rank(state.cards[self.current_player(state)])]
        board = RANK_NAMES[rank(state.board)] if state.board is not None else ""
        return f"{own}{board}|{'/'.join(state.rounds)}"

    def contributions(self, state: LeducState) -> list[int]:
        """Chips each player has put in the pot so far (antes included)."""
        return self._contributions(state)

    @staticmethod
    def _facing_bet(history: str) -> bool:
        # The player to act faces a bet iff the previous action was a raise.
        return history.endswith(RAISE)

    @staticmethod
    def _hand_strength(card: int, board: int) -> tuple[int, int]:
        """Comparable strength: a pair with the board beats any unpaired hand."""
        return (int(rank(card) == rank(board)), rank(card))

    @staticmethod
    def _contributions(state: LeducState) -> list[int]:
        """Chips each player has put in the pot, replaying the betting."""
        contributions = [ANTE, ANTE]
        for round_index, history in enumerate(state.rounds):
            bet = BET_SIZES[round_index]
            for i, action in enumerate(history):
                player = i % 2
                if action == CALL:
                    contributions[player] = max(contributions)
                elif action == RAISE:
                    contributions[player] = max(contributions) + bet
        return contributions
