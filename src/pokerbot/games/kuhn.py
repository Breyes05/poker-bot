"""Kuhn poker: 3 cards (J, Q, K), 1 card each, one betting round, ante 1.

Rules
-----
Both players ante 1 chip and are dealt one card from {J, Q, K}; the third card
is unused. Player 0 acts first. There are only two actions:

* ``"p"`` (pass) - check when there is no bet to face, fold when facing a bet.
* ``"b"`` (bet)  - bet 1 when there is no bet to face, call when facing a bet.

Every possible betting sequence ("history"):

    pp   check, check          -> showdown, pot 2  (winner +1)
    bp   bet, fold             -> player 0 wins +1
    bb   bet, call             -> showdown, pot 4  (winner +2)
    pbp  check, bet, fold      -> player 1 wins +1
    pbb  check, bet, call      -> showdown, pot 4  (winner +2)

State representation
--------------------
A state is ``(cards, history)``. ``cards`` is ``()`` at the root, before the deal
(a chance node), and ``(card_p0, card_p1)`` afterwards, with cards encoded as
0=J, 1=Q, 2=K. ``history`` is the string of actions taken so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from pokerbot.games.base import Game

CARD_NAMES = "JQK"
PASS, BET = "p", "b"
TERMINAL_HISTORIES = frozenset({"pp", "bp", "bb", "pbp", "pbb"})


@dataclass(frozen=True)
class KuhnState:
    cards: tuple[int, ...]
    history: str


class KuhnPoker(Game):
    """Two-player Kuhn poker. Payoffs are zero-sum: payoff(s, 0) == -payoff(s, 1)."""

    def initial_state(self) -> KuhnState:
        return KuhnState(cards=(), history="")

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in TERMINAL_HISTORIES

    def is_chance(self, state: KuhnState) -> bool:
        return not state.cards

    def current_player(self, state: KuhnState) -> int:
        # Players alternate, starting with player 0.
        return len(state.history) % 2

    def legal_actions(self, state: KuhnState) -> list[str]:
        return [PASS, BET]

    def next_state(self, state: KuhnState, action) -> KuhnState:
        if self.is_chance(state):
            return KuhnState(cards=action, history="")
        return KuhnState(cards=state.cards, history=state.history + action)

    def chance_outcomes(self, state: KuhnState) -> list[tuple[tuple[int, int], float]]:
        # Deal both cards at once: 6 ordered (p0, p1) pairs, all equally likely.
        deals = list(permutations(range(3), 2))
        return [(deal, 1 / len(deals)) for deal in deals]

    def payoff(self, state: KuhnState, player: int) -> float:
        h = state.history
        if h in ("bp", "pbp"):
            # Someone folded. The folder is whoever acted last; the other player
            # wins the folder's 1-chip ante.
            folder = (len(h) - 1) % 2
            p0_payoff = -1.0 if folder == 0 else 1.0
        else:
            # Showdown: each player put in 2 if there was a bet+call, else 1.
            stake = 2.0 if BET in h else 1.0
            p0_payoff = stake if state.cards[0] > state.cards[1] else -stake
        return p0_payoff if player == 0 else -p0_payoff

    def info_set_key(self, state: KuhnState) -> str:
        # The acting player sees their own card and the public betting history,
        # never the opponent's card. Example: "Q|pb" = holding Q, after check-bet.
        card = state.cards[self.current_player(state)]
        return f"{CARD_NAMES[card]}|{state.history}"
