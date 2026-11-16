"""Train a solver on a game and save the resulting strategy.

Example:
    uv run python scripts/train.py --game kuhn --iterations 10000
"""

import argparse
import pickle
import time
from pathlib import Path

from pokerbot.eval.value import expected_value
from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver

GAMES = {"kuhn": KuhnPoker}
SOLVERS = {"cfr": CFRSolver}


def print_strategy(strategy: dict[str, list[float]], actions: dict[str, list]) -> None:
    width = max(len(key) for key in strategy)
    for key in sorted(strategy):
        probs = "  ".join(f"{a}={p:.3f}" for a, p in zip(actions[key], strategy[key]))
        print(f"  {key:<{width}}  {probs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["kuhn", "leduc"], default="kuhn")
    parser.add_argument("--solver", choices=["cfr", "mccfr"], default="cfr")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--out", default="results/strategy.pkl")
    args = parser.parse_args()

    if args.game not in GAMES:
        raise SystemExit(f"game '{args.game}' is not implemented yet (planned for M3)")
    if args.solver not in SOLVERS:
        raise SystemExit(f"solver '{args.solver}' is not implemented yet (planned for M3)")

    game = GAMES[args.game]()
    solver = SOLVERS[args.solver](game)

    start = time.perf_counter()
    solver.train(args.iterations)
    elapsed = time.perf_counter() - start

    strategy = solver.average_strategy()
    actions = {key: solver.actions(key) for key in strategy}

    print(f"Trained {args.solver} on {args.game}: {args.iterations} iterations in {elapsed:.2f}s")
    print(f"Game value for player 0: {expected_value(game, strategy, player=0):+.5f}")
    print("Average strategy (info set = own card | betting history; p = check/fold, b = bet/call):")
    print_strategy(strategy, actions)
    # TODO(M2): report exploitability here.

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(
            {
                "game": args.game,
                "solver": args.solver,
                "iterations": args.iterations,
                "strategy": strategy,
                "actions": actions,
            },
            f,
        )
    print(f"Saved strategy to {out}")


if __name__ == "__main__":
    main()
