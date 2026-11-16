"""Overlay exploitability curves written by train.py.

Example (compare CFR and MCCFR on Leduc by wall-clock time):
    uv run --extra viz python scripts/plot_convergence.py \
        results/leduc_cfr_convergence.csv results/leduc_mccfr_convergence.csv \
        --x seconds --title "Leduc Hold'em: CFR vs MCCFR" --out results/leduc_cfr_vs_mccfr.png
"""

import argparse
import csv
from pathlib import Path

X_COLUMNS = {"iterations": ("iteration", "Iterations"), "seconds": ("train_seconds", "Seconds")}


def load(path: Path) -> tuple[list[float], list[float], list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return (
        [float(r["iteration"]) for r in rows],
        [float(r["train_seconds"]) for r in rows],
        [float(r["exploitability_mbb"]) for r in rows],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", type=Path, help="convergence CSVs from train.py")
    parser.add_argument("--x", choices=list(X_COLUMNS), default="iterations")
    parser.add_argument("--title", default="Exploitability")
    parser.add_argument("--out", type=Path, default=Path("results/convergence.png"))
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("needs matplotlib: run with `uv run --extra viz ...`") from exc

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for path in args.csvs:
        iterations, seconds, mbb = load(path)
        xs = iterations if args.x == "iterations" else seconds
        # Label from the file name, e.g. leduc_mccfr_convergence.csv -> "MCCFR".
        label = path.stem.split("_")[1].upper() if path.stem.count("_") >= 2 else path.stem
        points = [(x, y) for x, y in zip(xs, mbb) if x > 0]
        ax.loglog([p[0] for p in points], [p[1] for p in points], marker="o", ms=4, label=label)

    ax.set_xlabel(X_COLUMNS[args.x][1])
    ax.set_ylabel("Exploitability (mbb/hand)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
