"""Solve the demo games and export the strategies the API serves.

Example:
    uv run python scripts/export_strategies.py

Writes api/strategies/leduc.json and api/strategies/pushfold.json. The API only
reads these files, so it never has to solve anything at startup (the on-demand
solver handles push/fold stacks that aren't precomputed).
"""

import argparse
import json
import time
from pathlib import Path

from pokerbot.eval.exploitability import to_mbb
from pokerbot.games.leduc import LeducHoldem
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.solvers.vector_cfr import VectorCFRSolver

OUT_DIR = Path(__file__).resolve().parent.parent / "api" / "strategies"


def solve(game, iterations: int) -> tuple[dict, float]:
    solver = VectorCFRSolver(game, "cfr+")
    solver.train(iterations)
    strategy = {k: [round(p, 4) for p in v] for k, v in solver.average_strategy().items()}
    return strategy, to_mbb(game, solver.exploitability())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leduc-iterations", type=int, default=5000)
    parser.add_argument("--pushfold-iterations", type=int, default=1000)
    parser.add_argument("--stacks", type=int, nargs=2, default=[2, 30], metavar=("MIN", "MAX"))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    strategy, mbb = solve(LeducHoldem(), args.leduc_iterations)
    with (OUT_DIR / "leduc.json").open("w") as f:
        json.dump(
            {"iterations": args.leduc_iterations, "exploitability_mbb": mbb, "strategy": strategy},
            f,
            separators=(",", ":"),
        )
    print(f"leduc: {mbb:.4f} mbb/hand ({time.perf_counter() - start:.1f}s)")

    stacks = {}
    for stack in range(args.stacks[0], args.stacks[1] + 1):
        start = time.perf_counter()
        strategy, mbb = solve(PushFoldHoldem(stack), args.pushfold_iterations)
        stacks[str(stack)] = {"exploitability_mbb": mbb, "strategy": strategy}
        print(f"push/fold {stack} BB: {mbb:.4f} mbb/hand ({time.perf_counter() - start:.1f}s)")
    with (OUT_DIR / "pushfold.json").open("w") as f:
        json.dump(
            {"iterations": args.pushfold_iterations, "stacks": stacks}, f, separators=(",", ":")
        )


if __name__ == "__main__":
    main()
