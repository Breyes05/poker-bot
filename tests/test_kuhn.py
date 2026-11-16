"""Tests for Kuhn poker rules and CFR convergence."""

import pytest

from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker, KuhnState
from pokerbot.solvers.cfr import CFRSolver

J, Q, K = 0, 1, 2
GAME_VALUE = -1 / 18  # Kuhn (1950): player 0's expected payoff at equilibrium

game = KuhnPoker()


def state(cards, history):
    return KuhnState(cards=cards, history=history)


# --- Rules -----------------------------------------------------------------


def test_root_is_chance_node_with_uniform_deals():
    root = game.initial_state()
    assert game.is_chance(root)
    outcomes = game.chance_outcomes(root)
    assert len(outcomes) == 6  # 3 * 2 ordered deals
    assert {deal for deal, _ in outcomes} == {(a, b) for a in range(3) for b in range(3) if a != b}
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "history, expected_p0",
    [
        ("pp", 1),  # check-check, K beats J at showdown
        ("bp", 1),  # p1 folds to a bet
        ("bb", 2),  # bet-call showdown
        ("pbp", -1),  # p0 folds to a bet
        ("pbb", 2),  # check-bet-call showdown
    ],
)
def test_payoffs_when_p0_holds_higher_card(history, expected_p0):
    s = state((K, J), history)
    assert game.is_terminal(s)
    assert game.payoff(s, 0) == expected_p0
    assert game.payoff(s, 1) == -expected_p0


@pytest.mark.parametrize(
    "history, expected_p0",
    [("pp", -1), ("bp", 1), ("bb", -2), ("pbp", -1), ("pbb", -2)],
)
def test_payoffs_when_p0_holds_lower_card(history, expected_p0):
    # Folds pay the same regardless of cards; showdowns flip.
    s = state((J, Q), history)
    assert game.payoff(s, 0) == expected_p0
    assert game.payoff(s, 1) == -expected_p0


@pytest.mark.parametrize("history", ["", "p", "b", "pb"])
def test_non_terminal_histories(history):
    s = state((Q, K), history)
    assert not game.is_terminal(s)
    assert not game.is_chance(s)
    assert game.legal_actions(s) == ["p", "b"]


def test_players_alternate_starting_with_p0():
    assert [game.current_player(state((Q, K), h)) for h in ["", "p", "b", "pb"]] == [0, 1, 1, 0]


def test_info_set_hides_opponent_card():
    # Player 0 holding Q cannot tell whether player 1 has J or K.
    assert game.info_set_key(state((Q, J), "")) == game.info_set_key(state((Q, K), ""))
    assert game.info_set_key(state((Q, J), "pb")) == game.info_set_key(state((Q, K), "pb"))
    # Player 1 holding J cannot tell whether player 0 has Q or K.
    assert game.info_set_key(state((Q, J), "b")) == game.info_set_key(state((K, J), "b"))


def test_info_set_distinguishes_own_card_and_history():
    assert game.info_set_key(state((Q, J), "")) != game.info_set_key(state((K, J), ""))
    assert game.info_set_key(state((Q, J), "")) != game.info_set_key(state((Q, J), "pb"))


def test_info_set_key_format():
    assert game.info_set_key(state((K, J), "")) == "K|"
    assert game.info_set_key(state((K, J), "b")) == "J|b"


# --- CFR convergence ---------------------------------------------------------


@pytest.fixture(scope="module")
def avg_strategy():
    solver = CFRSolver(game)
    solver.train(10_000)
    return solver.average_strategy()


def bet(strategy, key):
    """Probability of 'b' (bet or call) at an info set."""
    return strategy[key][1]


def test_cfr_finds_all_twelve_info_sets(avg_strategy):
    assert len(avg_strategy) == 12  # 3 cards x 4 decision histories


def test_cfr_converges_to_game_value(avg_strategy):
    assert expected_value(game, avg_strategy, player=0) == pytest.approx(GAME_VALUE, abs=1e-3)
    assert expected_value(game, avg_strategy, player=1) == pytest.approx(-GAME_VALUE, abs=1e-3)


def test_cfr_matches_kuhn_equilibrium_family(avg_strategy):
    """Kuhn's equilibria form a one-parameter family indexed by alpha in [0, 1/3],
    where alpha is how often player 0 bluffs with a Jack."""
    tol = 0.03
    alpha = bet(avg_strategy, "J|")
    assert -tol <= alpha <= 1 / 3 + tol

    # Player 0
    assert bet(avg_strategy, "Q|") == pytest.approx(0, abs=tol)  # never open-bets a Queen
    assert bet(avg_strategy, "K|") == pytest.approx(3 * alpha, abs=tol)  # bets K at 3x bluff rate
    assert bet(avg_strategy, "J|pb") == pytest.approx(0, abs=tol)  # always folds J to a bet
    assert bet(avg_strategy, "Q|pb") == pytest.approx(alpha + 1 / 3, abs=tol)
    assert bet(avg_strategy, "K|pb") == pytest.approx(1, abs=tol)  # always calls with K

    # Player 1 (same for every alpha)
    assert bet(avg_strategy, "J|b") == pytest.approx(0, abs=tol)  # folds J to a bet
    assert bet(avg_strategy, "Q|b") == pytest.approx(1 / 3, abs=tol)  # calls Q 1/3 of the time
    assert bet(avg_strategy, "K|b") == pytest.approx(1, abs=tol)  # always calls K
    assert bet(avg_strategy, "J|p") == pytest.approx(1 / 3, abs=tol)  # bluffs J 1/3 after a check
    assert bet(avg_strategy, "Q|p") == pytest.approx(0, abs=tol)  # checks Q behind
    assert bet(avg_strategy, "K|p") == pytest.approx(1, abs=tol)  # always bets K after a check


def test_uniform_strategy_is_not_the_game_value():
    # Sanity check that the value test can fail: random play is far from -1/18.
    assert expected_value(game, {}, player=0) != pytest.approx(GAME_VALUE, abs=1e-2)
