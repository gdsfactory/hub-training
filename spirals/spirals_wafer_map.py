# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Build a spatial wafer map from per-die JSON results for the spirals tutorial."""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def main(
    files: list[Path],
    /,
    *,
    output_key: str = "propagation_loss",
    min_output: float = 0.0,
    max_output: float = 5.0,
    output_name: str = "wafer_map",
) -> Path:
    """Build a spatial wafer map from per-die JSON results.

    Args:
        files: List of JSON files from die analyses (with die_x, die_y fields)
        output_key: The key in the JSON to aggregate (e.g., "propagation_loss")
        min_output: Minimum output value for good die
        max_output: Maximum output value for good die
        output_name: Name for output PNG file

    Returns:
        Path to the generated wafer map PNG
    """
    analyses = {}
    for file in files:
        data = json.loads(file.read_text())
        value = data.get(output_key, np.nan)
        x, y = int(data["die_x"]), int(data["die_y"])
        analyses[(x, y)] = value

    if not analyses:
        msg = "No die analyses found with valid coordinates"
        raise ValueError(msg)

    # Build the wafer grid
    coords = np.array(sorted(analyses.keys()))
    die_xs = np.unique(coords[:, 0])
    die_ys = np.unique(coords[:, 1])
    x_min, x_max = int(die_xs.min()), int(die_xs.max())
    y_min, y_max = int(die_ys.min()), int(die_ys.max())
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    # Cell edges at half-integers so cells are centred on integer die coordinates
    x_edges = np.arange(nx + 1) + x_min - 0.5
    y_edges = np.arange(ny + 1) + y_min - 0.5
    X, Y = np.meshgrid(x_edges, y_edges, indexing="ij")

    data_grid = np.full((nx, ny), np.nan)
    for (x, y), value in analyses.items():
        data_grid[x - x_min, y - y_min] = value

    exists = ~np.isnan(data_grid)
    toolow = exists & (data_grid < min_output)
    toohigh = exists & (data_grid > max_output)
    good = exists & ~toolow & ~toohigh
    ones = np.ones((nx, ny))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Left panel: pass/fail map ---
    # Use masked arrays so only the relevant cells are drawn per layer
    ax0.pcolormesh(X, Y, np.ma.masked_where(~good, ones),
                   cmap=mcolors.ListedColormap(["#00ff00"]), vmin=0, vmax=1, alpha=0.8)
    ax0.pcolormesh(X, Y, np.ma.masked_where(~toolow, ones),
                   cmap=mcolors.ListedColormap(["blue"]), vmin=0, vmax=1, alpha=0.8)
    ax0.pcolormesh(X, Y, np.ma.masked_where(~toohigh, ones),
                   cmap=mcolors.ListedColormap(["red"]), vmin=0, vmax=1, alpha=0.8)
    ax0.pcolor(X, Y, np.ma.masked_where(exists, ones),
               hatch="//", edgecolor="C7", facecolor="none", linewidth=0)

    ax0.plot([], [], "s", c="#00ff00", alpha=0.8, label="good")
    ax0.plot([], [], "s", c="blue", alpha=0.8, label=f"too low [<{min_output:.2f}]")
    ax0.plot([], [], "s", c="red", alpha=0.8, label=f"too high [>{max_output:.2f}]")
    ax0.legend(loc="upper right", fontsize=7)

    # --- Right panel: continuous values ---
    ax1.pcolormesh(X, Y, np.ma.masked_where(~exists, data_grid),
                   vmin=min_output, vmax=max_output)
    ax1.pcolor(X, Y, np.ma.masked_where(exists, ones),
               hatch="//", edgecolor="C7", facecolor="none", linewidth=0)

    # Value labels and axis formatting for both panels
    for ax in (ax0, ax1):
        for i in range(nx):
            for j in range(ny):
                v = data_grid[i, j]
                if not np.isnan(v):
                    ax.text(i + x_min, j + y_min, f"{v:.2f}",
                            ha="center", va="center", color="black", fontsize=8)
        ax.set_xticks(die_xs)
        ax.set_xticks(0.5 * (die_xs[1:] + die_xs[:-1]), minor=True)
        ax.set_yticks(die_ys)
        ax.set_yticks(0.5 * (die_ys[1:] + die_ys[:-1]), minor=True)
        ax.grid(visible=True, which="minor", ls=":")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("die x")
        ax.set_ylabel("die y")

    fig.suptitle(output_key)
    plt.tight_layout()

    output_path = files[0].parent / f"{output_name}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    return output_path
