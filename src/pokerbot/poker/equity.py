"""Preflop all-in equities between the 169 hand classes, and deal probabilities.

``equity[a, b]`` is the probability that a hand of class ``a`` beats a hand of
class ``b`` when all five board cards are dealt, counting a tie as half a win.
It averages over every pair of specific combos that can be dealt together (no
shared cards), so card removal is handled: AA vs AKo is computed only over
deals where the two hands don't share an ace.

Computing it exactly means 1.7M boards for each of ~800k combo pairs, which is
too slow in Python. Instead ``compute_equity_table`` samples random boards and,
for each board, scores **all** 1,326 combos at once and compares every pair of
them. One board therefore updates every matchup, and 50k boards give equities
accurate to a few tenths of a percent. The result is cached in
``data/preflop_equity.npz`` (built by ``scripts/build_equity_table.py``).

``deal_probabilities`` is exact: the chance that player 0 gets class ``a`` and
player 1 gets class ``b``, again accounting for card removal.
"""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import numpy as np

from pokerbot.poker.evaluator import evaluate
from pokerbot.poker.hand_classes import COMBO_CLASS, COMBOS, NUM_CLASSES

EQUITY_PATH = Path(__file__).parent / "data" / "preflop_equity.npz"

_NUM_COMBOS = len(COMBOS)
# combo_cards[i, c] is True if combo i contains card c.
_combo_cards = np.zeros((_NUM_COMBOS, 52), dtype=bool)
_combo_cards[np.arange(_NUM_COMBOS)[:, None], COMBOS] = True
# compatible[i, j] is True if combos i and j share no card (can be dealt together).
_compatible = ~(_combo_cards.astype(np.int32) @ _combo_cards.T.astype(np.int32)).astype(bool)
# class_onehot[i, k] is 1 if combo i belongs to class k.
_class_onehot = np.zeros((_NUM_COMBOS, NUM_CLASSES))
_class_onehot[np.arange(_NUM_COMBOS), COMBO_CLASS] = 1.0


def _to_classes(combo_matrix: np.ndarray) -> np.ndarray:
    """Sum a (1326, 1326) combo-pair matrix into a (169, 169) class-pair matrix."""
    return _class_onehot.T @ combo_matrix @ _class_onehot


@cache
def deal_probabilities() -> np.ndarray:
    """P(player 0 holds class a, player 1 holds class b), shape (169, 169)."""
    pairs = _to_classes(_compatible.astype(np.float64))
    return pairs / pairs.sum()  # total = 1326 * 1225 ordered deals


def compute_equity_table(
    num_boards: int = 50_000, seed: int = 0, batch: int = 200, verbose: bool = False
) -> np.ndarray:
    """Monte Carlo class-vs-class equity table, shape (169, 169)."""
    rng = np.random.default_rng(seed)
    # sign_sum[i, j] accumulates +1 per board where combo i beats combo j, -1 per loss.
    sign_sum = np.zeros((_NUM_COMBOS, _NUM_COMBOS), dtype=np.int32)
    valid_counts = np.zeros((_NUM_COMBOS, _NUM_COMBOS), dtype=np.float64)
    start = time.perf_counter()

    for done in range(0, num_boards, batch):
        n = min(batch, num_boards - done)
        boards = rng.random((n, 52)).argsort(axis=1)[:, :5]  # 5 distinct cards each
        # A combo is "valid" on a board if it shares no card with it.
        valid = ~_combo_cards[:, boards].any(axis=2).T  # (n, 1326)
        seven = np.concatenate(
            [
                np.broadcast_to(COMBOS, (n, _NUM_COMBOS, 2)),
                np.repeat(boards[:, None], _NUM_COMBOS, 1),
            ],
            axis=2,
        )
        ranks = evaluate(seven)  # (n, 1326)
        # Give invalid combos rank -1 so they lose to everything; corrected below.
        ranks = np.where(valid, ranks, -1)
        for r in ranks:
            # Compress scores to 0..1325 so the 1326 x 1326 comparison fits in int16.
            order = np.unique(r, return_inverse=True)[1].astype(np.int16)
            diff = order[:, None] - order[None, :]
            np.sign(diff, out=diff)  # +1 win, 0 tie, -1 loss
            np.add(sign_sum, diff, out=sign_sum, casting="unsafe")
        v = valid.astype(np.float64)
        valid_counts += v.T @ v  # boards where both i and j were valid
        if verbose and (done // batch) % 25 == 0:
            print(f"  {done + n:>7}/{num_boards} boards  {time.perf_counter() - start:6.1f}s")

    # Count "half-wins": 2 per win, 1 per tie, 0 per loss, i.e. sign + 1 per board.
    score2 = sign_sum + num_boards
    # Remove the contributions of boards where i or j was invalid. With rank -1,
    # (valid i, invalid j) scored 2, (invalid, valid) scored 0, (invalid, invalid) 1.
    total_boards = float(num_boards)
    valid_any = np.diag(valid_counts)  # boards where each combo was valid
    i_only = valid_any[:, None] - valid_counts  # i valid, j invalid
    neither = total_boards - valid_any[:, None] - valid_any[None, :] + valid_counts
    wins2 = score2 - 2 * i_only - neither

    # Pool combo pairs into class pairs: total half-wins / (2 * total valid boards).
    mask = _compatible.astype(np.float64)
    return _to_classes(wins2 * mask) / (2 * _to_classes(valid_counts * mask))


def save_equity_table(equity: np.ndarray, path: Path = EQUITY_PATH, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, equity=equity.astype(np.float32), **metadata)


@cache
def load_equity_table(path: Path = EQUITY_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `uv run python scripts/build_equity_table.py`"
        )
    return np.load(path)["equity"].astype(np.float64)


def exact_combo_equity(hand_a, hand_b) -> float:
    """Exact equity of two specific hands, enumerating all C(48, 5) = 1,712,304 boards."""
    import itertools

    used = set(hand_a) | set(hand_b)
    deck = [c for c in range(52) if c not in used]
    boards = np.array(list(itertools.combinations(deck, 5)), dtype=np.int8)
    total = 0.0
    for chunk in np.array_split(boards, 8):
        a = evaluate(np.concatenate([np.broadcast_to(hand_a, (len(chunk), 2)), chunk], axis=1))
        b = evaluate(np.concatenate([np.broadcast_to(hand_b, (len(chunk), 2)), chunk], axis=1))
        total += (a > b).sum() + 0.5 * (a == b).sum()
    return total / len(boards)


def exact_class_equity(class_a: int, class_b: int) -> float:
    """Exact class-vs-class equity.

    Combo pairs that differ only by relabeling suits have the same equity, so
    each group is enumerated once and weighted by its size.
    """
    import itertools

    perms = list(itertools.permutations(range(4)))

    def canonical(a, b):
        forms = []
        for p in perms:
            ma = tuple(sorted(c // 4 * 4 + p[c % 4] for c in a))
            mb = tuple(sorted(c // 4 * 4 + p[c % 4] for c in b))
            forms.append((ma, mb))
        return min(forms)

    groups: dict[tuple, int] = {}
    for a in COMBOS[COMBO_CLASS == class_a]:
        for b in COMBOS[COMBO_CLASS == class_b]:
            if not set(a) & set(b):
                key = canonical(a.tolist(), b.tolist())
                groups[key] = groups.get(key, 0) + 1
    weighted = sum(n * exact_combo_equity(*key) for key, n in groups.items())
    return weighted / sum(groups.values())
