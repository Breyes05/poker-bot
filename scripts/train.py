"""Train a solver on a game, track exploitability, and save the resulting strategy.

Examples:
    uv run python scripts/train.py --game kuhn --iterations 10000
    uv run --extra viz python scripts/train.py --game kuhn --iterations 100000 --plot
"""

import argparse
import csv
import pickle
import time
from pathlib import Path

from pokerbot.eval.exploitability import exploitability, to_mbb
from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver

GAMES = {"kuhn": KuhnPoker}
SOLVERS = {"cfr": CFRSolver}


def checkpoints(iterations: int, points: int) -> list[int]:
    """Roughly log-spaced iteration counts from 1 to `iterations` (good for log-log plots)."""
    if points <= 1:
        return [iterations]
    ratio = iterations ** (1 / (points - 1))
    return sorted({max(1, round(ratio**i)) for i in range(points)} | {iterations})


def print_strategy(strategy: dict[str, list[float]], actions: dict[str, list]) -> None:
    width = max(len(key) for key in strategy)
    for key in sorted(strategy):
        probs = "  ".join(f"{a}={p:.3f}" for a, p in zip(actions[key], strategy[key]))
        print(f"  {key:<{width}}  {probs}")


def save_plot(rows: list[dict], path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("--plot needs matplotlib: run with `uv run --extra viz ...`") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([r["iteration"] for r in rows], [r["exploitability_mbb"] for r in rows], marker="o")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exploitability (mbb/hand)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["kuhn", "leduc"], default="kuhn")
    parser.add_argument("--solver", choices=["cfr", "mccfr"], default="cfr")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--eval-points", type=int, default=20, help="exploitability checkpoints")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--plot", action="store_true", help="save a convergence plot (needs viz)")
    args = parser.parse_args()

    if args.game not in GAMES:
        raise SystemExit(f"game '{args.game}' is not implemented yet (planned for M3)")
    if args.solver not in SOLVERS:
        raise SystemExit(f"solver '{args.solver}' is not implemented yet (planned for M3)")

    game = GAMES[args.game]()
    solver = SOLVERS[args.solver](game)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.game}_{args.solver}"

    rows = []
    train_time = 0.0
    print(f"{'iteration':>10}  {'exploitability (mbb/hand)':>26}  {'train time':>10}")
    for target in checkpoints(args.iterations, args.eval_points):
        start = time.perf_counter()
        solver.train(target - solver.iterations)
        train_time += time.perf_counter() - start

        mbb = to_mbb(game, exploitability(game, solver.average_strategy()))
        rows.append({"iteration": target, "exploitability_mbb": mbb, "train_seconds": train_time})
        print(f"{target:>10}  {mbb:>26.3f}  {train_time:>9.2f}s")

    strategy = solver.average_strategy()
    actions = {key: solver.actions(key) for key in strategy}
    print(f"\nGame value for player 0: {expected_value(game, strategy, player=0):+.5f}")
    print("Average strategy (info set = own card | betting history; p = check/fold, b = bet/call):")
    print_strategy(strategy, actions)

    curve_path = out_dir / f"{name}_convergence.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strategy_path = out_dir / f"{name}_strategy.pkl"
    with strategy_path.open("wb") as f:
        pickle.dump(
            {
                "game": args.game,
                "solver": args.solver,
                "iterations": args.iterations,
                "exploitability_mbb": rows[-1]["exploitability_mbb"],
                "strategy": strategy,
                "actions": actions,
            },
            f,
        )
    print(f"\nSaved {strategy_path} and {curve_path}")

    if args.plot:
        plot_path = out_dir / f"{name}_convergence.png"
        save_plot(rows, plot_path, f"{args.solver.upper()} on {args.game.title()} poker")
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
