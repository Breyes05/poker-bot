"""Tests for external-sampling MCCFR."""

import pytest

from pokerbot.eval.exploitability import exploitability, to_mbb
from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.games.leduc import LeducHoldem
from pokerbot.solvers.mccfr import MCCFRSolver


def test_mccfr_converges_on_kuhn():
    game = KuhnPoker()
    solver = MCCFRSolver(game, seed=0)
    solver.train(50_000)
    strategy = solver.average_strategy()
    assert len(strategy) == 12
    assert expected_value(game, strategy) == pytest.approx(-1 / 18, abs=5e-3)
    assert to_mbb(game, exploitability(game, strategy)) < 5


def test_mccfr_exploitability_falls_on_leduc():
    game = LeducHoldem()
    solver = MCCFRSolver(game, seed=0)
    solver.train(2_000)
    early = to_mbb(game, exploitability(game, solver.average_strategy()))
    solver.train(18_000)
    late = to_mbb(game, exploitability(game, solver.average_strategy()))
    assert late < early
    assert late < 200


def test_mccfr_is_reproducible_with_a_seed():
    game = KuhnPoker()
    a, b = MCCFRSolver(game, seed=42), MCCFRSolver(game, seed=42)
    a.train(500)
    b.train(500)
    assert a.average_strategy() == b.average_strategy()


def test_mccfr_strategies_are_valid_distributions():
    game = LeducHoldem()
    solver = MCCFRSolver(game, seed=1)
    solver.train(1_000)
    for key, probs in solver.average_strategy().items():
        assert len(probs) == len(solver.actions(key))
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0 for p in probs)
