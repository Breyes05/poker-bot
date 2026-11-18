"""Vectorized 5-7 card poker hand evaluator.

Cards are ints 0-51: ``card = rank * 4 + suit`` with rank 0-12 meaning 2..A and
suit 0-3 meaning c, d, h, s.

``evaluate`` scores many hands at once with numpy. A score is an int where a
higher score means a stronger hand and equal scores tie::

    score = category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5

``category`` is 0 (high card) to 8 (straight flush), and r1..r5 are the ranks
that break ties within the category (rank + 1, so 0 means "unused"). For
example, a full house of kings over fives is ``6 << 20 | 12 << 16 | 4 << 12``.

The evaluator works directly on 7 cards (no looping over 5-card subsets): it
builds a rank-by-suit presence grid per hand and reads every category off it.
"""

from __future__ import annotations

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CATEGORIES = (
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

_RANK_IDS = np.arange(13)
# Straights as (top rank, bitmask over ranks), lowest first. The wheel A-2-3-4-5
# has top rank 3 (the five).
_STRAIGHTS = [(3, (1 << 12) | 0b1111)] + [(top, 0b11111 << (top - 4)) for top in range(4, 13)]


def card(text: str) -> int:
    """Parse one card like ``"As"`` or ``"Td"``."""
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def cards(text: str) -> list[int]:
    """Parse space-separated cards like ``"As Kd 7h"``."""
    return [card(t) for t in text.split()]


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def category(score) -> np.ndarray | int:
    """Category index (see ``CATEGORIES``) of a score or array of scores."""
    return np.asarray(score) >> 20


def _top_ranks(mask: np.ndarray, k: int) -> np.ndarray:
    """The k highest ranks where `mask` (N, 13) is true, descending; -1 pads."""
    ranks = np.where(mask, _RANK_IDS, -1)
    return -np.sort(-ranks, axis=1)[:, :k]


def _straight_top(present: np.ndarray) -> np.ndarray:
    """Top rank of the best straight in each row of `present` (N, 13), or -1."""
    bits = (present.astype(np.int64) << _RANK_IDS).sum(axis=1)
    top = np.full(len(bits), -1)
    for rank, mask in _STRAIGHTS:  # ascending, so higher straights overwrite lower
        top[(bits & mask) == mask] = rank
    return top


def _pack(cat: int, *ranks: np.ndarray) -> np.ndarray:
    score = np.full(len(ranks[0]), cat << 20, dtype=np.int64)
    for i, r in enumerate(ranks):
        score |= (r + 1) << (16 - 4 * i)
    return score


def evaluate(hands) -> np.ndarray:
    """Score hands of 5-7 cards. Input shape (..., n_cards); output shape (...)."""
    hands = np.asarray(hands, dtype=np.int64)
    batch_shape, n_cards = hands.shape[:-1], hands.shape[-1]
    if not 5 <= n_cards <= 7:
        raise ValueError(f"need 5-7 cards per hand, got {n_cards}")
    flat = hands.reshape(-1, n_cards)
    n = len(flat)

    # grid[h, rank, suit] is True if hand h holds that card.
    grid = np.zeros((n, 52), dtype=bool)
    grid[np.arange(n)[:, None], flat] = True
    grid = grid.reshape(n, 13, 4)

    counts = grid.sum(axis=2)  # (n, 13) cards per rank
    suit_counts = grid.sum(axis=1)  # (n, 4) cards per suit
    present = counts > 0

    # At most one suit can have 5+ cards out of 7.
    flush_suit = suit_counts.argmax(axis=1)
    has_flush = suit_counts.max(axis=1) >= 5
    flush_ranks = grid[np.arange(n), :, flush_suit] & has_flush[:, None]

    straight_flush = _straight_top(flush_ranks)
    straight = _straight_top(present)
    quads = _top_ranks(counts == 4, 1)[:, 0]
    trips = _top_ranks(counts == 3, 2)
    pairs = _top_ranks(counts == 2, 3)
    singles = _top_ranks(counts == 1, 5)
    flush5 = _top_ranks(flush_ranks, 5)

    def best_excluding(*excluded: np.ndarray) -> np.ndarray:
        """Highest rank present other than the excluded ranks (the kicker)."""
        mask = present.copy()
        for r in excluded:
            mask &= _RANK_IDS != r[:, None]
        return _top_ranks(mask, 1)[:, 0]

    # Highest category first: np.select picks the first condition that holds.
    conditions = [
        straight_flush >= 0,
        quads >= 0,
        (trips[:, 0] >= 0) & ((trips[:, 1] >= 0) | (pairs[:, 0] >= 0)),
        has_flush,
        straight >= 0,
        trips[:, 0] >= 0,
        pairs[:, 1] >= 0,
        pairs[:, 0] >= 0,
    ]
    choices = [
        _pack(8, straight_flush),
        _pack(7, quads, best_excluding(quads)),
        _pack(6, trips[:, 0], np.maximum(trips[:, 1], pairs[:, 0])),
        _pack(5, *flush5.T),
        _pack(4, straight),
        _pack(3, trips[:, 0], singles[:, 0], singles[:, 1]),
        _pack(2, pairs[:, 0], pairs[:, 1], best_excluding(pairs[:, 0], pairs[:, 1])),
        _pack(1, pairs[:, 0], singles[:, 0], singles[:, 1], singles[:, 2]),
    ]
    scores = np.select(conditions, choices, default=_pack(0, *singles.T))
    return scores.reshape(batch_shape)
