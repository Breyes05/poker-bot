"""13x13 range charts for push/fold strategies.

A chart maps each of the 169 hand classes to a probability (e.g. how often the
small blind pushes it), laid out like a standard range chart: pairs on the
diagonal, suited hands above it, offsuit hands below it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.poker.hand_classes import CLASS_NAMES, GRID_RANKS, NUM_CLASSES


def chart(strategy: dict[str, list[float]], history: str, action_index: int) -> np.ndarray:
    """(13, 13) grid of P(action) for each hand class at the given betting history.

    For push/fold: ``chart(s, "", 1)`` is the push chart (small blind), and
    ``chart(s, "p", 1)`` is the call chart (big blind).
    """
    probs = np.array([strategy[f"{name}|{history}"][action_index] for name in CLASS_NAMES])
    return probs.reshape(13, 13)


def text_chart(grid: np.ndarray, threshold: float = 0.5) -> str:
    """Compact text chart: hands played more than `threshold` of the time are shown."""
    lines = ["    " + "    ".join(GRID_RANKS)]
    for row in range(13):
        cells = []
        for col in range(13):
            name = CLASS_NAMES[row * 13 + col]
            cells.append(f"{name:<4}" if grid[row, col] > threshold else " .  ")
        lines.append(f"{GRID_RANKS[row]}  " + " ".join(cells))
    return "\n".join(lines)


def plot_charts(grids: dict[str, np.ndarray], path: Path, title: str = "") -> None:
    """Save side-by-side 13x13 heatmaps, one per entry of `grids` (title -> grid)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("plotting needs matplotlib: run with `uv run --extra viz ...`") from exc

    fig, axes = plt.subplots(1, len(grids), figsize=(6.2 * len(grids), 6.4), squeeze=False)
    for ax, (name, grid) in zip(axes[0], grids.items()):
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1)
        for idx in range(NUM_CLASSES):
            row, col = divmod(idx, 13)
            p = grid[row, col]
            color = "white" if p > 0.6 else "black"
            ax.text(
                col,
                row - 0.12,
                CLASS_NAMES[idx],
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold",
            )
            ax.text(
                col,
                row + 0.25,
                f"{p:.0%}" if 0.005 < p < 0.995 else "",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
