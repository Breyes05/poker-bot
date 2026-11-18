"""Benchmark the vectorized hand evaluator.

Example:
    uv run python scripts/bench_evaluator.py --hands 1000000
"""

import argparse
import time

import numpy as np

from pokerbot.poker.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1_000_000)
    parser.add_argument("--cards", type=int, default=7, choices=[5, 6, 7])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    hands = rng.random((args.hands, 52)).argsort(axis=1)[:, : args.cards]
    evaluate(hands[:1000])  # warm up

    start = time.perf_counter()
    evaluate(hands)
    seconds = time.perf_counter() - start
    rate = args.hands / seconds
    print(f"{args.hands:,} {args.cards}-card hands in {seconds:.3f}s: {rate / 1e6:.2f}M hands/s")


if __name__ == "__main__":
    main()
