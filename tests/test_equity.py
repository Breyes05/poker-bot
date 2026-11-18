"""Tests for hand classes, deal probabilities, and the cached preflop equity table."""

import numpy as np
import pytest

from pokerbot.poker.equity import deal_probabilities, load_equity_table
from pokerbot.poker.evaluator import cards
from pokerbot.poker.hand_classes import (
    CLASS_INDEX,
    CLASS_NAMES,
    COMBO_COUNTS,
    COMBOS,
    class_of,
    range_fraction,
)

# --- Hand classes ----------------------------------------------------------------


def test_169_classes_cover_all_1326_combos():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)) == 169
    assert len(COMBOS) == 1326
    assert COMBO_COUNTS.sum() == 1326
    assert sorted(set(COMBO_COUNTS.tolist())) == [4, 6, 12]


@pytest.mark.parametrize(
    "name, combos, row_col",
    [("AA", 6, (0, 0)), ("AKs", 4, (0, 1)), ("AKo", 12, (1, 0)), ("32o", 12, (12, 11))],
)
def test_chart_layout(name, combos, row_col):
    index = CLASS_INDEX[name]
    assert divmod(index, 13) == row_col
    assert COMBO_COUNTS[index] == combos


def test_class_of():
    assert CLASS_NAMES[class_of(*cards("As Ks"))] == "AKs"
    assert CLASS_NAMES[class_of(*cards("Ks Ad"))] == "AKo"
    assert CLASS_NAMES[class_of(*cards("7h 7c"))] == "77"
    assert CLASS_NAMES[class_of(*cards("2d 7c"))] == "72o"


def test_range_fraction():
    assert range_fraction(np.ones(169)) == pytest.approx(1.0)
    pairs_only = np.array([1.0 if len(n) == 2 else 0.0 for n in CLASS_NAMES])
    assert range_fraction(pairs_only) == pytest.approx(78 / 1326)


# --- Deal probabilities ------------------------------------------------------------


def test_deal_probabilities_account_for_card_removal():
    probs = deal_probabilities()
    deals = 1326 * 1225  # ordered pairs of non-overlapping hands
    assert probs.sum() == pytest.approx(1.0)
    assert np.allclose(probs, probs.T)
    # AA vs AA: 6 ways to pick the first pair of aces, then exactly 1 left.
    assert probs[CLASS_INDEX["AA"], CLASS_INDEX["AA"]] * deals == pytest.approx(6)
    # AKs vs AKs: 4 suits for the first, 3 left for the second.
    assert probs[CLASS_INDEX["AKs"], CLASS_INDEX["AKs"]] * deals == pytest.approx(12)
    # AA vs 72o has no card overlap: 6 * 12.
    assert probs[CLASS_INDEX["AA"], CLASS_INDEX["72o"]] * deals == pytest.approx(72)


# --- Equity table ------------------------------------------------------------------


def test_equity_table_is_consistent():
    equity = load_equity_table()
    assert equity.shape == (169, 169)
    assert np.all((equity >= 0) & (equity <= 1))
    assert np.allclose(equity + equity.T, 1.0, atol=1e-6)  # one side's win is the other's loss
    assert np.allclose(np.diag(equity), 0.5, atol=1e-6)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Exact values from scripts/build_equity_table.py (full board enumeration).
        ("AA", "KK", 0.8195),
        ("AKo", "QQ", 0.4324),
        ("22", "AKo", 0.5265),
        ("72o", "AA", 0.1180),
        ("JTs", "A2o", 0.4858),
    ],
)
def test_equity_matches_exact_enumeration(a, b, expected):
    equity = load_equity_table()
    assert equity[CLASS_INDEX[a], CLASS_INDEX[b]] == pytest.approx(expected, abs=0.006)


def test_better_hands_have_more_equity_against_a_random_hand():
    equity = load_equity_table()
    vs_random = (equity * deal_probabilities()).sum(axis=1) / deal_probabilities().sum(axis=1)
    assert vs_random[CLASS_INDEX["AA"]] == pytest.approx(0.852, abs=0.005)
    assert vs_random[CLASS_INDEX["AA"]] == vs_random.max()
    assert vs_random[CLASS_INDEX["32o"]] == vs_random.min()
    assert vs_random[CLASS_INDEX["AKs"]] > vs_random[CLASS_INDEX["AKo"]]
