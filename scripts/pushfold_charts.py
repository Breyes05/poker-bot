"""Solve heads-up push/fold Hold'em at several stack depths and render the charts.

Examples:
    uv run python scripts/pushfold_charts.py --stacks 10
    uv run --extra viz python scripts/pushfold_charts.py --stacks 5 10 15 20 --plot

For each stack depth, prints the exploitability, how often each player plays,
and text charts; writes results/pushfold_{stack}bb.json (the strategy, for the
demo app) and, with --plot, results/pushfold_{stack}bb.png.
"""

import argparse
import json
import time
from pathlib import Path

from pokerbot.eval.exploitability import to_mbb
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.poker.hand_classes import CLASS_NAMES, range_fraction
from pokerbot.solvers.vector_cfr import VectorCFRSolver
from pokerbot.viz.charts import chart, plot_charts, text_chart


def fmt_stack(stack: float) -> str:
    return f"{stack:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", type=float, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--variant", choices=["cfr+", "cfr"], default="cfr+")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", action="store_true", help="save PNG charts (needs viz)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stack in args.stacks:
        name = f"pushfold_{fmt_stack(stack)}bb"
        game = PushFoldHoldem(stack)
        start = time.perf_counter()
        solver = VectorCFRSolver(game, args.variant)
        solver.train(args.iterations)
        seconds = time.perf_counter() - start
        strategy = solver.average_strategy()
        mbb = to_mbb(game, solver.exploitability())
        push, call = chart(strategy, "", 1), chart(strategy, "p", 1)
        sb_value = solver.tree.expected_value(solver.average_strategy_matrix())

        print(
            f"=== {fmt_stack(stack)} BB  ({args.variant}, {args.iterations} iterations, "
            f"{seconds:.1f}s) ==="
        )
        print(f"exploitability {mbb:.3f} mbb/hand, small blind EV {sb_value:+.4f} BB")
        print(f"small blind pushes {range_fraction(push.ravel()):.1%} of hands")
        print(text_chart(push))
        print(f"big blind calls {range_fraction(call.ravel()):.1%} of hands")
        print(text_chart(call))
        print()

        with (args.out_dir / f"{name}.json").open("w") as f:
            json.dump(
                {
                    "stack_bb": stack,
                    "solver": args.variant,
                    "iterations": args.iterations,
                    "exploitability_mbb": mbb,
                    "push": dict(zip(CLASS_NAMES, push.ravel().round(4).tolist())),
                    "call": dict(zip(CLASS_NAMES, call.ravel().round(4).tolist())),
                },
                f,
                indent=1,
            )
        if args.plot:
            plot_charts(
                {
                    f"Small blind push ({range_fraction(push.ravel()):.1%})": push,
                    f"Big blind call ({range_fraction(call.ravel()):.1%})": call,
                },
                args.out_dir / f"{name}.png",
                title=f"Heads-up push/fold Nash, {fmt_stack(stack)} BB",
            )
            print(f"Saved {args.out_dir / name}.png")


if __name__ == "__main__":
    main()
