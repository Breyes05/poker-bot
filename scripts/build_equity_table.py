"""Build the cached preflop equity table (169 x 169 hand classes).

Example:
    uv run python scripts/build_equity_table.py --boards 200000

Writes src/pokerbot/poker/data/preflop_equity.npz, then spot-checks a few
matchups against exact enumeration.
"""

import argparse
import time

from pokerbot.poker.equity import (
    EQUITY_PATH,
    compute_equity_table,
    exact_class_equity,
    save_equity_table,
)
from pokerbot.poker.hand_classes import CLASS_INDEX

SPOT_CHECKS = [("AA", "KK"), ("AKo", "QQ"), ("22", "AKo"), ("72o", "AA"), ("JTs", "A2o")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=200_000, help="random boards to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-check", action="store_true", help="skip the exact spot checks")
    args = parser.parse_args()

    start = time.perf_counter()
    equity = compute_equity_table(args.boards, seed=args.seed, verbose=True)
    print(f"Computed in {time.perf_counter() - start:.1f}s")
    save_equity_table(equity, num_boards=args.boards, seed=args.seed)
    print(f"Saved {EQUITY_PATH}")

    if args.skip_check:
        return
    print(f"\n{'matchup':<12} {'table':>8} {'exact':>8} {'error':>8}")
    for a, b in SPOT_CHECKS:
        ia, ib = CLASS_INDEX[a], CLASS_INDEX[b]
        exact = exact_class_equity(ia, ib)
        table = equity[ia, ib]
        print(f"{a + ' v ' + b:<12} {table:>8.4f} {exact:>8.4f} {table - exact:>+8.4f}")


if __name__ == "__main__":
    main()
