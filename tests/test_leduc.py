"""Tests for Leduc Hold'em rules, and vanilla CFR running on it unchanged."""

import pytest

from pokerbot.eval.exploitability import exploitability, to_mbb
from pokerbot.games.leduc import LeducHoldem, LeducState
from pokerbot.solvers.cfr import CFRSolver

# Cards 0-5: rank = card // 2, so (J, J, Q, Q, K, K).
J1, J2, Q1, Q2, K1, K2 = range(6)

game = LeducHoldem()


def state(cards, board, *rounds):
    return LeducState(cards=cards, board=board, rounds=tuple(rounds))


# --- Chance nodes --------------------------------------------------------------


def test_deal_is_uniform_over_ordered_pairs():
    root = game.initial_state()
    assert game.is_chance(root)
    outcomes = game.chance_outcomes(root)
    assert len(outcomes) == 30  # 6 * 5
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)


@pytest.mark.parametrize("round1", ["cc", "rc", "crc", "rrc", "crrc"])
def test_board_is_dealt_after_round_one_completes(round1):
    s = state((J1, K1), None, round1)
    assert game.is_chance(s)
    assert not game.is_terminal(s)
    outcomes = game.chance_outcomes(s)
    assert sorted(card for card, _ in outcomes) == [J2, Q1, Q2, K2]  # excludes dealt cards
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    assert game.next_state(s, Q1) == state((J1, K1), Q1, round1, "")


@pytest.mark.parametrize("round1", ["", "c", "r", "cr", "rr", "crr"])
def test_round_one_in_progress_is_a_decision_node(round1):
    s = state((J1, K1), None, round1)
    assert not game.is_chance(s)
    assert not game.is_terminal(s)


# --- Betting -------------------------------------------------------------------


@pytest.mark.parametrize(
    "history, expected",
    [
        ("", ["c", "r"]),  # nothing to call: no fold
        ("c", ["c", "r"]),
        ("r", ["f", "c", "r"]),
        ("cr", ["f", "c", "r"]),
        ("rr", ["f", "c"]),  # raise cap (2) reached
        ("crr", ["f", "c"]),
    ],
)
def test_legal_actions(history, expected):
    assert game.legal_actions(state((J1, K1), None, history)) == expected
    assert game.legal_actions(state((J1, K1), Q1, "cc", history)) == expected


def test_player_zero_acts_first_in_both_rounds():
    assert game.current_player(state((J1, K1), None, "")) == 0
    assert game.current_player(state((J1, K1), None, "c")) == 1
    assert game.current_player(state((J1, K1), Q1, "crc", "")) == 0
    assert game.current_player(state((J1, K1), Q1, "crc", "r")) == 1


# --- Payoffs (player 0's view; worked out by hand) -----------------------------


@pytest.mark.parametrize(
    "s, expected_p0",
    [
        # Folds: the winner takes what the folder put in.
        (state((J1, K1), None, "rf"), 1),  # p1 folds to a 2-chip bet, loses ante
        (state((J1, K1), None, "crrf"), 3),  # p1 bet (in for 3), p0 raised, p1 folds
        (state((J1, K1), Q1, "cc", "rf"), 1),  # round-2 fold, only antes in
        (state((K1, J1), Q1, "rc", "rrf"), -7),  # p0: 1+2 (r1) +4 (r2 bet), then folds
        # Showdowns: the winner takes the loser's contribution.
        (state((K1, J1), Q1, "cc", "cc"), 1),  # K high beats J high
        (state((J1, K1), J2, "cc", "cc"), 1),  # pair of jacks beats K high
        (state((Q1, K1), J1, "rc", "rc"), -7),  # 1 + 2 + 4 each, K beats Q
        (state((K1, Q1), J1, "rrc", "rrc"), 13),  # max pot: 1 + 4 + 8 each
        (state((Q1, Q2), K1, "rc", "cc"), 0),  # same rank: split
    ],
)
def test_payoffs(s, expected_p0):
    assert game.is_terminal(s)
    assert game.payoff(s, 0) == expected_p0
    assert game.payoff(s, 1) == -expected_p0


# --- Info sets -----------------------------------------------------------------


def test_info_set_hides_opponent_card():
    assert game.info_set_key(state((Q1, J1), None, "")) == game.info_set_key(
        state((Q1, K2), None, "")
    )


def test_info_set_ignores_suits():
    assert game.info_set_key(state((Q1, J1), K1, "cc", "")) == game.info_set_key(
        state((Q2, J1), K2, "cc", "")
    )


def test_info_set_key_format():
    assert game.info_set_key(state((K1, J1), None, "cr")) == "K|cr"
    assert game.info_set_key(state((Q1, J1), K2, "rc", "r")) == "JK|rc/r"  # p1 to act


# --- Whole-game checks ---------------------------------------------------------


def test_cfr_runs_unchanged_and_visits_all_288_info_sets():
    solver = CFRSolver(game)
    solver.train(1)
    assert len(solver.nodes) == 288  # 18 round-1 + 270 round-2 rank-based info sets


def test_uniform_random_exploitability_matches_openspiel():
    # OpenSpiel's NashConv for uniform random Leduc poker is 4.747222...; our
    # exploitability is NashConv / 2.
    assert exploitability(game, {}) == pytest.approx(4.747222222222222 / 2)


def test_cfr_exploitability_falls_on_leduc():
    solver = CFRSolver(game)
    solver.train(20)
    early = to_mbb(game, exploitability(game, solver.average_strategy()))
    solver.train(180)
    late = to_mbb(game, exploitability(game, solver.average_strategy()))
    assert late < early
    assert late < 150
