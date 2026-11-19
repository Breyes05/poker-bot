"""Before/after benchmark for M5: reference CFR vs vectorized CFR vs CFR+.

Example:
    uv run python scripts/benchmark.py

Prints Markdown tables (pasted into the README):

1. Time per iteration, and per exploitability evaluation, for each game.
2. Time to reach a target exploitability. The reference and vectorized CFR
   compute identical iterates, so the reference's time is its per-iteration
   cost times the iterations vectorized CFR needed (running it would take
   minutes); CFR+ is timed directly.
"""

import argparse
import time

from pokerbot.eval.exploitability import exploitability, to_mbb
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.games.leduc import LeducHoldem
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.solvers.tree import GameTree
from pokerbot.solvers.vector_cfr import VectorCFRSolver

GAMES = {
    "Kuhn": (KuhnPoker, 1.0),
    "Leduc": (LeducHoldem, 20.0),
    "Push/fold 10 BB": (lambda: PushFoldHoldem(10), 1.0),
}


def per_call(fn, min_seconds: float = 1.0, max_calls: int = 10_000) -> float:
    """Average seconds per call of fn(), repeating until min_seconds has passed."""
    calls, start = 0, time.perf_counter()
    while calls < max_calls:
        fn()
        calls += 1
        if time.perf_counter() - start >= min_seconds:
            break
    return (time.perf_counter() - start) / calls


def iterations_to_target(game, tree, variant: str, target_mbb: float, limit: int = 200_000):
    """Train until the average strategy is within target_mbb; return (iterations, seconds)."""
    solver = VectorCFRSolver(game, variant, tree=tree)
    step, seconds = 1, 0.0
    while solver.iterations < limit:
        start = time.perf_counter()
        solver.train(step)
        seconds += time.perf_counter() - start
        if to_mbb(game, solver.exploitability()) <= target_mbb:
            return solver.iterations, seconds
        step = max(1, solver.iterations // 10)  # check roughly every 10%
    return None, seconds


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def bench_game(name: str, game, target: float, min_seconds: float) -> tuple[str, str]:
    """Return (speed row, time-to-target row) for one game."""
    tree = GameTree(game)
    ref = CFRSolver(game)
    vec = VectorCFRSolver(game, "cfr", tree=tree)
    plus = VectorCFRSolver(game, "cfr+", tree=tree)
    ref.train(1)  # warm up (creates info set nodes)
    t_ref = per_call(lambda: ref.train(1), min_seconds)
    t_vec = per_call(lambda: vec.train(1), min_seconds)
    t_plus = per_call(lambda: plus.train(1), min_seconds)

    strategy = vec.average_strategy()
    matrix = tree.strategy_matrix(strategy)
    e_ref = per_call(lambda: exploitability(game, strategy), min_seconds)
    e_vec = per_call(lambda: tree.exploitability(matrix), min_seconds)
    speed_row = (
        f"| {name} | {tree.num_nodes:,} | {fmt_time(t_ref)} | {fmt_time(t_vec)} "
        f"| **{t_ref / t_vec:.0f}×** | {fmt_time(t_plus)} | {fmt_time(e_ref)} "
        f"| {fmt_time(e_vec)} | **{e_ref / e_vec:.0f}×** |"
    )

    n_cfr, s_cfr = iterations_to_target(game, tree, "cfr", target)
    n_plus, s_plus = iterations_to_target(game, tree, "cfr+", target)
    if n_cfr is None or n_plus is None:
        return speed_row, f"| {name} | {target:g} mbb | not reached | | | |"
    ref_estimate = n_cfr * t_ref
    target_row = (
        f"| {name} | {target:g} mbb | {n_cfr:,} it · {fmt_time(ref_estimate)} "
        f"| {n_cfr:,} it · {fmt_time(s_cfr)} | {n_plus:,} it · {fmt_time(s_plus)} "
        f"| **{ref_estimate / s_plus:,.0f}×** |"
    )
    return speed_row, target_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-seconds", type=float, default=1.0, help="timing window per item")
    args = parser.parse_args()

    speed_rows, target_rows = [], []
    for name, (make_game, target) in GAMES.items():
        speed_row, target_row = bench_game(name, make_game(), target, args.min_seconds)
        speed_rows.append(speed_row)
        target_rows.append(target_row)
        print(f"done: {name}", flush=True)

    print(
        "\n| Game | Tree nodes | Reference CFR / iter | Vectorized CFR / iter | Speedup "
        "| CFR+ / iter | Reference exploitability | Vectorized exploitability | Speedup |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(speed_rows))
    print(
        "\n| Game | Target | Reference CFR (est.) | Vectorized CFR | Vectorized CFR+ "
        "| Overall speedup |"
    )
    print("|---|---|---|---|---|---|")
    print("\n".join(target_rows))


if __name__ == "__main__":
    main()
