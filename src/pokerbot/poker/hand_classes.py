"""The 169 preflop hand classes (AA, AKs, AKo, ..., 32o).

Before the flop, suits only matter through "same suit or not", so the 1,326
two-card starting hands collapse into 169 classes: 13 pairs (6 combos each),
78 suited hands (4 combos each), and 78 offsuit hands (12 combos each).

Classes are laid out like a standard 13x13 range chart, ranks descending (A=0):

* row == col: pairs (diagonal)
* row <  col: suited hands (above the diagonal), e.g. AKs at (0, 1)
* row >  col: offsuit hands (below the diagonal), e.g. AKo at (1, 0)

A class's index is ``row * 13 + col``.
"""

from __future__ import annotations

import itertools

import numpy as np

GRID_RANKS = "AKQJT98765432"  # row/column order of the chart
NUM_CLASSES = 169


def _grid_pos(rank: int) -> int:
    """Evaluator rank (0 = deuce ... 12 = ace) -> chart row/column (0 = ace)."""
    return 12 - rank


def class_name(index: int) -> str:
    row, col = divmod(index, 13)
    if row == col:
        return GRID_RANKS[row] * 2
    hi, lo = min(row, col), max(row, col)
    return GRID_RANKS[hi] + GRID_RANKS[lo] + ("s" if row < col else "o")


def class_of(c1: int, c2: int) -> int:
    """Class index of the two-card hand (c1, c2)."""
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = _grid_pos(max(r1, r2)), _grid_pos(min(r1, r2))
    if r1 == r2:
        return hi * 13 + hi
    if c1 % 4 == c2 % 4:
        return hi * 13 + lo  # suited: above the diagonal
    return lo * 13 + hi  # offsuit: below the diagonal


CLASS_NAMES: list[str] = [class_name(i) for i in range(NUM_CLASSES)]
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

#: All 1,326 two-card combos, shape (1326, 2), and the class of each.
COMBOS: np.ndarray = np.array(list(itertools.combinations(range(52), 2)))
COMBO_CLASS: np.ndarray = np.array([class_of(a, b) for a, b in COMBOS])
#: How many combos each class has (6, 4, or 12).
COMBO_COUNTS: np.ndarray = np.bincount(COMBO_CLASS, minlength=NUM_CLASSES)


def range_fraction(probs) -> float:
    """Fraction of all 1,326 hands played, given a per-class probability (169,)."""
    return float(np.dot(probs, COMBO_COUNTS) / len(COMBOS))
