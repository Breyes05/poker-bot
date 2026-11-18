"""Tests for the vectorized hand evaluator, including a brute-force cross-check."""

import itertools
from collections import Counter

import numpy as np
import pytest

from pokerbot.poker.evaluator import CATEGORIES, cards, category, evaluate


def score(text: str) -> int:
    return int(evaluate(cards(text)))


def name(text: str) -> str:
    return CATEGORIES[category(score(text))]


# --- Reference evaluator: simple, slow, and written independently --------------


def reference_5(hand) -> tuple:
    ranks = sorted((c // 4 for c in hand), reverse=True)
    flush = len({c % 4 for c in hand}) == 1
    counts = Counter(ranks)
    # Ranks ordered by (how many, rank), e.g. full house -> [trips rank, pair rank].
    ordered = [r for r, _ in sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    shape = sorted(counts.values(), reverse=True)
    top = None
    if len(counts) == 5:
        if ranks[0] - ranks[4] == 4:
            top = ranks[0]
        elif ranks == [12, 3, 2, 1, 0]:
            top = 3  # wheel: A-2-3-4-5 is a five-high straight
    if top is not None and flush:
        return (8, top)
    if shape == [4, 1]:
        return (7, *ordered)
    if shape == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *ranks)
    if top is not None:
        return (4, top)
    if shape == [3, 1, 1]:
        return (3, *ordered)
    if shape == [2, 2, 1]:
        return (2, *ordered)
    if shape == [2, 1, 1, 1]:
        return (1, *ordered)
    return (0, *ranks)


def reference_7(hand) -> tuple:
    return max(reference_5(five) for five in itertools.combinations(hand, 5))


# --- Categories ------------------------------------------------------------------


@pytest.mark.parametrize(
    "hand, expected",
    [
        ("As Ks Qs Js Ts 2d 3c", "straight flush"),
        ("Ah 2h 3h 4h 5h Kd Kc", "straight flush"),  # steel wheel
        ("9c 9d 9h 9s 2c 3d 4h", "four of a kind"),
        ("Kc Kd Kh 5s 5c 2d 3h", "full house"),
        ("Kc Kd Kh 5s 5c 5d 3h", "full house"),  # two sets of trips
        ("2h 7h 9h Jh Kh Ac Ad", "flush"),
        ("Ac 2d 3h 4s 5c 9d Kh", "straight"),  # wheel
        ("Tc Jd Qh Ks Ac 2d 2h", "straight"),
        ("7c 7d 7h As Kc 2d 4h", "three of a kind"),
        ("7c 7d 4h 4s Kc Kd 2h", "two pair"),  # three pairs
        ("7c 7d 4h 9s Kc Jd 2h", "pair"),
        ("2c 4d 6h 8s Tc Qd Ah", "high card"),
    ],
)
def test_categories(hand, expected):
    assert name(hand) == expected


# --- Ordering --------------------------------------------------------------------


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        ("2c 3c 4c 5c 6c Ad Kd", "Ac 2d 3h 4s 5h Kd Kc"),  # six-high SF > wheel straight
        ("2c 3d 4h 5s 6c Kd Qh", "Ac 2d 3h 4s 5c Kd Qh"),  # six-high straight > wheel
        ("Ac Ad Ah As 2c 3d 4h", "Kc Kd Kh Ks Ac Ad 4h"),  # higher quads win
        ("Kc Kd Kh Ks Ac 2d 3h", "Kc Kd Kh Ks Qc Jd Th"),  # quads: kicker plays
        ("Ac Ad Ah 2s 2c 3d 4h", "Kc Kd Kh As Ac 3d 4h"),  # trips rank decides boats
        ("7c 7d 4h 4s 5c 5d Ah", "7c 7d 4h 4s 5c 5d Kh"),  # two pair: best kicker
        ("7c 7d 4h 4s 6c 6d 2h", "7c 7d 4h 4s 5c 5d Ah"),  # the top two pairs play
        ("Ac Ad Kh Qs 9c 3d 2h", "Ac Ad Kh Qs 8c 3d 2h"),  # pair: third kicker
        ("Ah 9h 7h 4h 2h Kc Qd", "Kh Qh Jh Th 8h Ac Ad"),  # ace-high flush wins
    ],
)
def test_ordering(stronger, weaker):
    assert score(stronger) > score(weaker)


def test_ties_when_the_board_plays():
    assert score("Ac Kd Tc Jd Qh 2s 3s") == score("Ah Kc Tc Jd Qh 4s 5s")  # same straight
    assert score("Ac Ad Kh Kd Qs Jc 2c") == score("Ah As Kc Ks Qd Jh 3h")  # kicker Q


def test_batch_shape_is_preserved():
    hands = np.array([[cards("As Ks Qs Js Ts 2d 3c")] * 3] * 2)
    assert evaluate(hands).shape == (2, 3)


def test_matches_brute_force_reference_on_random_hands():
    rng = np.random.default_rng(0)
    hands = np.array([rng.permutation(52)[:7] for _ in range(3000)])
    scores = evaluate(hands)
    refs = [reference_7(h.tolist()) for h in hands]
    # Same category for every hand...
    assert [category(s) for s in scores] == [r[0] for r in refs]
    # ...and the same order for every pair of hands: map reference tuples to their
    # position in sorted order, then compare all 3000 x 3000 pairwise comparisons.
    position = {r: i for i, r in enumerate(sorted(set(refs)))}
    ref_rank = np.array([position[r] for r in refs])
    assert np.array_equal(
        np.sign(scores[:, None] - scores[None, :]),
        np.sign(ref_rank[:, None] - ref_rank[None, :]),
    )


def test_all_five_card_hands_have_the_textbook_frequencies():
    hands = np.array(list(itertools.combinations(range(52), 5)), dtype=np.int8)
    counts = np.bincount(category(evaluate(hands)), minlength=9)
    assert counts.tolist() == [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
