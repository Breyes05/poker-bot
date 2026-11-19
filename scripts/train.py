"""Train a solver on a game, track exploitability, and save the resulting strategy.

Examples:
    uv run python scripts/train.py --game kuhn --iterations 10000
    uv run --extra viz python scripts/train.py --game kuhn --iterations 100000 --plot
    uv run python scripts/train.py --game leduc --solver mccfr --iterations 1000000 --seed 0
    uv run python scripts/train.py --game leduc --solver cfr+ --iterations 3000

Solvers: cfr (reference, pure Python), vcfr (same algorithm, vectorized), cfr+
(vectorized CFR+), mccfr (external sampling). Exploitability is always measured
with the vectorized best response on a compiled game tree.
"""

import argparse
import csv
import pickle
import time
from pathlib import Path

from pokerbot.eval.exploitability import to_mbb
from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.games.leduc import LeducHoldem
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.solvers.mccfr import MCCFRSolver
from pokerbot.solvers.tree import GameTree
from pokerbot.solvers.vector_cfr import VectorCFRSolver

GAMES = {"kuhn": KuhnPoker, "leduc": LeducHoldem}
SOLVERS = {
    "cfr": lambda game, tree, seed: CFRSolver(game),
    "vcfr": lambda game, tree, seed: VectorCFRSolver(game, "cfr", tree=tree),
    "cfr+": lambda game, tree, seed: VectorCFRSolver(game, "cfr+", tree=tree),
    "mccfr": lambda game, tree, seed: MCCFRSolver(game, seed=seed),
}
ACTION_LEGEND = {
    "kuhn": "info set = own card | history; p = check/fold, b = bet/call",
    "leduc": "info set = own card + board | round 1 / round 2; f = fold, c = check/call, "
    "r = bet/raise",
}


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
    parser.add_argument("--solver", choices=list(SOLVERS), default="cfr+")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--eval-points", type=int, default=20, help="exploitability checkpoints")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--plot", action="store_true", help="save a convergence plot (needs viz)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (mccfr only)")
    parser.add_argument(
        "--show-strategy",
        action="store_true",
        help="print every info set's strategy (default: only for small games)",
    )
    args = parser.parse_args()

    game = GAMES[args.game]()
    tree = GameTree(game)  # compiled once; used for fast exploitability (and by vcfr/cfr+)
    solver = SOLVERS[args.solver](game, tree, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.game}_{args.solver.replace('+', 'plus')}"

    rows = []
    train_time = 0.0
    print(f"{'iteration':>10}  {'exploitability (mbb/hand)':>26}  {'train time':>10}")
    for target in checkpoints(args.iterations, args.eval_points):
        start = time.perf_counter()
        solver.train(target - solver.iterations)
        train_time += time.perf_counter() - start

        mbb = to_mbb(game, tree.exploitability(tree.strategy_matrix(solver.average_strategy())))
        rows.append({"iteration": target, "exploitability_mbb": mbb, "train_seconds": train_time})
        print(f"{target:>10}  {mbb:>26.3f}  {train_time:>9.2f}s")

    strategy = solver.average_strategy()
    actions = {key: solver.actions(key) for key in strategy}
    print(f"\nGame value for player 0: {expected_value(game, strategy, player=0):+.5f}")
    if args.show_strategy or len(strategy) <= 30:
        print(f"Average strategy ({ACTION_LEGEND[args.game]}):")
        print_strategy(strategy, actions)
    else:
        print(f"{len(strategy)} info sets (pass --show-strategy to print them all)")

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
