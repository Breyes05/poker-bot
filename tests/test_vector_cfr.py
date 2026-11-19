"""The compiled game tree and vectorized CFR must agree with the reference code."""

import numpy as np
import pytest

from pokerbot.eval.exploitability import best_response_value, exploitability, to_mbb
from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.games.leduc import LeducHoldem
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.solvers.tree import GameTree
from pokerbot.solvers.vector_cfr import VectorCFRSolver

GAMES = {"kuhn": KuhnPoker(), "leduc": LeducHoldem()}


@pytest.fixture(scope="module")
def trees():
    return {name: GameTree(game) for name, game in GAMES.items()}


def random_strategy(tree: GameTree, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    matrix = rng.random((tree.num_infosets, tree.max_actions)) * tree.action_mask
    return tree.strategy_dict(matrix / matrix.sum(axis=1, keepdims=True))


# --- Tree structure ----------------------------------------------------------------


def test_tree_shape(trees):
    kuhn, leduc = trees["kuhn"], trees["leduc"]
    assert kuhn.num_infosets == 12 and leduc.num_infosets == 288
    # Kuhn: root + 6 deals + 6 * (2 + 2 + 2 + 2 terminals/decisions under each deal).
    assert kuhn.num_nodes == 1 + 6 * 9
    # Breadth-first: parents come before children, and depth levels are contiguous.
    assert np.all(kuhn.parent[1:] < np.arange(1, kuhn.num_nodes))
    assert kuhn.level_bounds[0] == 0 and kuhn.level_bounds[-1] == kuhn.num_nodes


def test_regret_matching_matches_m0_rule(trees):
    tree = trees["leduc"]
    regret = np.random.default_rng(0).normal(size=(tree.num_infosets, tree.max_actions))
    regret[0] = -1  # all negative -> uniform
    strategy = tree.regret_matching(regret)
    assert np.allclose(strategy.sum(axis=1), 1)
    assert np.all(strategy[~tree.action_mask] == 0)
    assert np.allclose(strategy[0, : tree.num_actions[0]], 1 / tree.num_actions[0])


# --- Evaluation matches the reference implementation --------------------------------


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
@pytest.mark.parametrize("seed", [0, 1])
def test_expected_value_and_best_response_match_reference(trees, name, seed):
    game, tree = GAMES[name], trees[name]
    strategy = random_strategy(tree, seed)
    matrix = tree.strategy_matrix(strategy)
    assert tree.expected_value(matrix) == pytest.approx(expected_value(game, strategy), abs=1e-12)
    for player in (0, 1):
        assert tree.best_response_value(matrix, player) == pytest.approx(
            best_response_value(game, strategy, player), abs=1e-12
        )
    assert tree.exploitability(matrix) == pytest.approx(exploitability(game, strategy), abs=1e-12)


def test_missing_info_sets_default_to_uniform(trees):
    tree = trees["kuhn"]
    assert np.allclose(tree.strategy_matrix({}), tree.uniform_strategy())
    assert tree.exploitability(tree.strategy_matrix({})) == pytest.approx(11 / 24)


# --- Vectorized CFR is the same algorithm --------------------------------------------


@pytest.mark.parametrize("name, iterations", [("kuhn", 300), ("leduc", 25)])
def test_vector_cfr_matches_reference_cfr(trees, name, iterations):
    reference = CFRSolver(GAMES[name])
    reference.train(iterations)
    vector = VectorCFRSolver(GAMES[name], "cfr", tree=trees[name])
    vector.train(iterations)
    ref, vec = reference.average_strategy(), vector.average_strategy()
    assert ref.keys() == vec.keys()
    for key in ref:
        assert vec[key] == pytest.approx(ref[key], abs=1e-10)
        assert vector.actions(key) == reference.actions(key)


def test_vector_cfr_matches_reference_on_pushfold():
    game = PushFoldHoldem(stack=10)
    reference = CFRSolver(game)
    reference.train(3)
    vector = VectorCFRSolver(game, "cfr")
    vector.train(3)
    ref, vec = reference.average_strategy(), vector.average_strategy()
    assert max(abs(a - b) for k in ref for a, b in zip(ref[k], vec[k])) < 1e-10


# --- CFR+ ------------------------------------------------------------------------------


def test_cfr_plus_converges_on_kuhn(trees):
    solver = VectorCFRSolver(GAMES["kuhn"], "cfr+", tree=trees["kuhn"])
    solver.train(2000)
    matrix = solver.average_strategy_matrix()
    assert trees["kuhn"].expected_value(matrix) == pytest.approx(-1 / 18, abs=1e-3)
    assert to_mbb(GAMES["kuhn"], solver.exploitability()) < 0.1


def test_cfr_plus_beats_cfr_on_leduc(trees):
    results = {}
    for variant in ("cfr", "cfr+"):
        solver = VectorCFRSolver(GAMES["leduc"], variant, tree=trees["leduc"])
        solver.train(500)
        results[variant] = to_mbb(GAMES["leduc"], solver.exploitability())
    assert results["cfr+"] < 2  # vanilla CFR is still ~65 mbb here
    assert results["cfr+"] < results["cfr"] / 10


def test_cfr_plus_keeps_regrets_non_negative(trees):
    solver = VectorCFRSolver(GAMES["leduc"], "cfr+", tree=trees["leduc"])
    solver.train(10)
    assert solver.regret.min() >= 0


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        VectorCFRSolver(GAMES["kuhn"], "dcfr")
