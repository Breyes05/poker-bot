"""Tests for push/fold Hold'em: rules, payoffs, and the solved strategy's shape."""

import pytest

from pokerbot.eval.exploitability import exploitability, to_mbb
from pokerbot.games.pushfold import PushFoldHoldem, PushFoldState
from pokerbot.poker.equity import load_equity_table
from pokerbot.poker.hand_classes import CLASS_INDEX, range_fraction
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.viz.charts import chart, text_chart

AA, KK, J7o = CLASS_INDEX["AA"], CLASS_INDEX["KK"], CLASS_INDEX["J7o"]


@pytest.fixture(scope="module")
def game():
    return PushFoldHoldem(stack=10)


@pytest.fixture(scope="module")
def solved():
    """Strategies after a short CFR run at two stack depths."""
    out = {}
    for stack in (5, 20):
        game = PushFoldHoldem(stack)
        solver = CFRSolver(game)
        solver.train(40)
        out[stack] = (game, solver.average_strategy())
    return out


# --- Rules -----------------------------------------------------------------------


def test_deal_covers_every_class_pair(game):
    outcomes = game.chance_outcomes(game.initial_state())
    assert len(outcomes) == 169 * 169
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)


def test_betting_structure(game):
    s = PushFoldState(hands=(AA, KK))
    assert game.current_player(s) == 0 and game.legal_actions(s) == ["f", "p"]
    s = game.next_state(s, "p")
    assert game.current_player(s) == 1 and game.legal_actions(s) == ["f", "c"]
    for history in ("f", "pf", "pc"):
        assert game.is_terminal(PushFoldState(hands=(AA, KK), history=history))


def test_payoffs(game):
    equity = load_equity_table()[AA, KK]
    cases = {"f": -0.5, "pf": 1.0, "pc": 10 * (2 * equity - 1)}
    for history, expected_p0 in cases.items():
        s = PushFoldState(hands=(AA, KK), history=history)
        assert game.payoff(s, 0) == pytest.approx(expected_p0)
        assert game.payoff(s, 1) == pytest.approx(-expected_p0)


def test_info_set_shows_only_own_hand(game):
    assert game.info_set_key(PushFoldState(hands=(AA, KK))) == "AA|"
    assert game.info_set_key(PushFoldState(hands=(AA, KK), history="p")) == "KK|p"


def test_stack_must_exceed_big_blind():
    with pytest.raises(ValueError):
        PushFoldHoldem(stack=1)


# --- Solving -----------------------------------------------------------------------


def test_cfr_visits_338_info_sets_and_converges(solved):
    for game, strategy in solved.values():
        assert len(strategy) == 2 * 169  # each class, for each player
        assert to_mbb(game, exploitability(game, strategy)) < 35


def test_premium_hands_always_push_and_call(solved):
    # Not exactly 1.0 after a short run: the average still includes the uniform
    # 50/50 strategy from the first iterations.
    for _, strategy in solved.values():
        for hand in ("AA", "KK", "AKs"):
            assert strategy[f"{hand}|"][1] > 0.95
            assert strategy[f"{hand}|p"][1] > 0.95


def test_ranges_tighten_as_stacks_get_deeper(solved):
    def fraction(stack, history):
        return range_fraction(chart(solved[stack][1], history, 1).ravel())

    assert fraction(5, "") > fraction(20, "")  # pushes
    assert fraction(5, "p") > fraction(20, "p")  # calls
    # At every depth the small blind pushes wider than the big blind calls.
    assert fraction(5, "") > fraction(5, "p")
    assert fraction(20, "") > fraction(20, "p")


def test_text_chart_marks_played_hands(solved):
    text = text_chart(chart(solved[20][1], "", 1))
    assert "AA" in text and "72o" not in text
