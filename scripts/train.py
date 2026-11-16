"""Train a solver on a game and save the resulting strategy."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["kuhn", "leduc"], default="kuhn")
    parser.add_argument("--solver", choices=["cfr", "mccfr"], default="cfr")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--out", default="results/strategy.pkl")
    args = parser.parse_args()

    # TODO: build game + solver, train, report exploitability, save strategy
    raise NotImplementedError(f"train {args.solver} on {args.game}")


if __name__ == "__main__":
    main()
