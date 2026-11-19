"""Hands of poker between a human and the solved strategy.

A hand object owns the full game state (including the bot's hidden cards) and
exposes a ``view()`` with only what the human is allowed to see. Each time the
human acts, ``act()`` applies the action and then plays the bot's turns and any
chance events until it's the human's turn again or the hand is over.

The bot plays the Nash strategy by *sampling*: if its strategy at an info set is
"raise 70%, call 30%", it raises 70% of the time. Every bot action in the log
carries those probabilities, and the view carries the Nash probabilities for the
human's own spot, so the front end can show "what equilibrium play would do".
"""

from __future__ import annotations

import random
import uuid
from collections import OrderedDict

from pokerbot.games.leduc import FOLD, RAISE, LeducHoldem, LeducState
from pokerbot.poker.evaluator import CATEGORIES, card_str, category, evaluate
from pokerbot.poker.hand_classes import CLASS_NAMES, class_of

Strategy = dict[str, list[float]]


class InvalidAction(ValueError):
    """The human tried something that isn't legal right now."""


def _sample(rng: random.Random, actions: list[str], probs: list[float]) -> str:
    if sum(probs) <= 0:
        probs = [1.0] * len(actions)
    return rng.choices(actions, weights=probs, k=1)[0]


class Hand:
    """Shared bookkeeping: id, seats, the action log."""

    game_name = ""

    def __init__(self, strategy: Strategy, human_seat: int, rng: random.Random):
        if human_seat not in (0, 1):
            raise InvalidAction("seat must be 0 or 1")
        self.id = uuid.uuid4().hex
        self.strategy = strategy
        self.human_seat = human_seat
        self.bot_seat = 1 - human_seat
        self.rng = rng
        self.log: list[dict] = []

    def who(self, seat: int) -> str:
        return "you" if seat == self.human_seat else "bot"

    def nash(self, key: str, actions: list[str]) -> dict[str, float]:
        probs = self.strategy.get(key, [1 / len(actions)] * len(actions))
        return dict(zip(actions, probs))


# --- Leduc Hold'em --------------------------------------------------------------------


def leduc_card(card: int) -> str:
    """Leduc's six cards as rank + suit, e.g. 'Js', 'Jh', 'Qs', ..."""
    return "JQK"[card // 2] + "sh"[card % 2]


def leduc_label(history: str, action: str) -> str:
    facing = history.endswith(RAISE)
    return {
        "f": "Fold",
        "c": "Call" if facing else "Check",
        "r": "Raise" if facing else "Bet",
    }[action]


class LeducHand(Hand):
    game_name = "leduc"

    def __init__(self, strategy: Strategy, human_seat: int, rng: random.Random):
        super().__init__(strategy, human_seat, rng)
        self.game = LeducHoldem()
        self.state: LeducState = self.game.initial_state()
        self._advance()

    @property
    def human_to_act(self) -> bool:
        g, s = self.game, self.state
        return (
            not g.is_terminal(s) and not g.is_chance(s) and g.current_player(s) == self.human_seat
        )

    def act(self, action: str) -> None:
        if not self.human_to_act:
            raise InvalidAction("it's not your turn")
        if action not in self.game.legal_actions(self.state):
            raise InvalidAction(f"'{action}' isn't a legal action here")
        self._record(self.human_seat, action)
        self.state = self.game.next_state(self.state, action)
        self._advance()

    def _record(self, seat: int, action: str) -> None:
        g, s = self.game, self.state
        actions = g.legal_actions(s)
        self.log.append(
            {
                "actor": self.who(seat),
                "action": action,
                "label": leduc_label(s.rounds[-1], action),
                "round": len(s.rounds),
                "nash": self.nash(g.info_set_key(s), actions),
            }
        )

    def _advance(self) -> None:
        """Play chance events and bot turns until the human must act or the hand ends."""
        g = self.game
        while not g.is_terminal(self.state):
            s = self.state
            if g.is_chance(s):
                outcomes, probs = zip(*g.chance_outcomes(s))
                outcome = _sample(self.rng, list(outcomes), list(probs))
                if s.cards:  # the deal has happened, so this is the board card
                    self.log.append({"actor": "dealer", "board": leduc_card(outcome), "round": 2})
                self.state = g.next_state(s, outcome)
            elif g.current_player(s) == self.bot_seat:
                actions = g.legal_actions(s)
                probs = self.nash(g.info_set_key(s), actions)
                action = _sample(self.rng, actions, list(probs.values()))
                self._record(self.bot_seat, action)
                self.state = g.next_state(s, action)
            else:
                return

    def view(self) -> dict:
        g, s = self.game, self.state
        terminal = g.is_terminal(s)
        contributions = g.contributions(s)
        showdown = terminal and not s.rounds[-1].endswith(FOLD)
        view = {
            "hand_id": self.id,
            "game": self.game_name,
            "human_seat": self.human_seat,
            "your_cards": [leduc_card(s.cards[self.human_seat])],
            "bot_cards": [leduc_card(s.cards[self.bot_seat])] if showdown else None,
            "board": [leduc_card(s.board)] if s.board is not None else [],
            "pot": sum(contributions),
            "your_chips_in": contributions[self.human_seat],
            "bot_chips_in": contributions[self.bot_seat],
            "log": self.log,
            "legal_actions": [],
            "nash": None,
            "info_set": None,
            "result": None,
        }
        if self.human_to_act:
            actions = g.legal_actions(s)
            key = g.info_set_key(s)
            view["legal_actions"] = [
                {"action": a, "label": leduc_label(s.rounds[-1], a)} for a in actions
            ]
            view["nash"] = self.nash(key, actions)
            view["info_set"] = key
        if terminal:
            payoff = g.payoff(s, self.human_seat)
            view["result"] = {
                "payoff": payoff,
                "outcome": "showdown" if showdown else "fold",
                "winner": "you" if payoff > 0 else "bot" if payoff < 0 else "split",
            }
        return view


# --- Push/fold Hold'em ------------------------------------------------------------------

PUSHFOLD_LABELS = {"": {"f": "Fold", "p": "Push all-in"}, "p": {"f": "Fold", "c": "Call"}}


class PushFoldHand(Hand):
    """Real 52-card hands: the strategy looks up each player's hand class, and a
    called all-in deals an actual board and compares the actual hands."""

    game_name = "pushfold"

    def __init__(self, strategy: Strategy, stack: float, human_seat: int, rng: random.Random):
        super().__init__(strategy, human_seat, rng)
        self.stack = stack
        deck = rng.sample(range(52), 9)
        self.cards = (deck[0:2], deck[2:4])  # (small blind, big blind)
        self.board_cards = deck[4:9]
        self.classes = tuple(class_of(*c) for c in self.cards)
        self.history = ""
        self._advance()

    @property
    def terminal(self) -> bool:
        return self.history in ("f", "pf", "pc")

    @property
    def to_act(self) -> int:
        return len(self.history)  # 0 = small blind, 1 = big blind

    @property
    def human_to_act(self) -> bool:
        return not self.terminal and self.to_act == self.human_seat

    def legal_actions(self) -> list[str]:
        return list(PUSHFOLD_LABELS[self.history])

    def key(self, seat: int) -> str:
        return f"{CLASS_NAMES[self.classes[seat]]}|{self.history}"

    def act(self, action: str) -> None:
        if not self.human_to_act:
            raise InvalidAction("it's not your turn")
        if action not in self.legal_actions():
            raise InvalidAction(f"'{action}' isn't a legal action here")
        self._play(self.human_seat, action)
        self._advance()

    def _play(self, seat: int, action: str) -> None:
        self.log.append(
            {
                "actor": self.who(seat),
                "action": action,
                "label": PUSHFOLD_LABELS[self.history][action],
                "round": 1,
                "nash": self.nash(self.key(seat), self.legal_actions()),
            }
        )
        self.history += action

    def _advance(self) -> None:
        while not self.terminal and self.to_act == self.bot_seat:
            actions = self.legal_actions()
            probs = self.nash(self.key(self.bot_seat), actions)
            self._play(self.bot_seat, _sample(self.rng, actions, list(probs.values())))

    def _showdown_scores(self) -> list[int]:
        return [int(evaluate(list(c) + self.board_cards)) for c in self.cards]

    def sb_payoff(self) -> float:
        if self.history == "f":
            return -0.5
        if self.history == "pf":
            return 1.0
        sb, bb = self._showdown_scores()
        return self.stack if sb > bb else -self.stack if sb < bb else 0.0

    def chips_in(self) -> list[float]:
        if self.history in ("", "f"):
            return [0.5, 1.0]
        if self.history in ("p", "pf"):
            return [self.stack, 1.0]
        return [self.stack, self.stack]

    def view(self) -> dict:
        showdown = self.history == "pc"
        chips = self.chips_in()
        view = {
            "hand_id": self.id,
            "game": self.game_name,
            "stack": self.stack,
            "human_seat": self.human_seat,
            "your_cards": [card_str(c) for c in self.cards[self.human_seat]],
            "your_hand_class": CLASS_NAMES[self.classes[self.human_seat]],
            "bot_cards": [card_str(c) for c in self.cards[self.bot_seat]] if showdown else None,
            "board": [card_str(c) for c in self.board_cards] if showdown else [],
            "pot": sum(chips),
            "your_chips_in": chips[self.human_seat],
            "bot_chips_in": chips[self.bot_seat],
            "log": self.log,
            "legal_actions": [],
            "nash": None,
            "info_set": None,
            "result": None,
        }
        if self.human_to_act:
            actions = self.legal_actions()
            view["legal_actions"] = [
                {"action": a, "label": PUSHFOLD_LABELS[self.history][a]} for a in actions
            ]
            view["nash"] = self.nash(self.key(self.human_seat), actions)
            view["info_set"] = self.key(self.human_seat)
        if self.terminal:
            sb = self.sb_payoff()
            payoff = sb if self.human_seat == 0 else -sb
            result = {
                "payoff": payoff,
                "outcome": "showdown" if showdown else "fold",
                "winner": "you" if payoff > 0 else "bot" if payoff < 0 else "split",
            }
            if showdown:
                scores = self._showdown_scores()
                result["your_hand_name"] = CATEGORIES[category(scores[self.human_seat])]
                result["bot_hand_name"] = CATEGORIES[category(scores[self.bot_seat])]
            view["result"] = result
        return view


# --- Storage ----------------------------------------------------------------------------


class HandStore:
    """In-memory hands, oldest evicted first.

    Hands live on the server (not the client) so the bot's cards stay hidden.
    A restart (e.g. Render's free tier sleeping) drops them; the front end then
    just deals a new hand.
    """

    def __init__(self, max_hands: int = 5000):
        self.max_hands = max_hands
        self._hands: OrderedDict[str, Hand] = OrderedDict()

    def add(self, hand: Hand) -> Hand:
        self._hands[hand.id] = hand
        while len(self._hands) > self.max_hands:
            self._hands.popitem(last=False)
        return hand

    def get(self, hand_id: str) -> Hand:
        return self._hands[hand_id]  # KeyError -> 404 in the API

    def __len__(self) -> int:
        return len(self._hands)
