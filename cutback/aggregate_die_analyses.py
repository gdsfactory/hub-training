# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "gfhub",
# ]
# ///
"""Aggregate die analyses into a wafer map."""

import json
from hashlib import md5
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from gfhub import tags


def main(
    files: list[Path],
    /,
    *,
    output_key: str = "component_loss",
    min_output: float = 0.0,
    max_output: float = 0.115,
    output_name: str = "wafer_map",
) -> Path:
    """Aggregate die analyses into a wafer map.

    Args:
        files: List of JSON files from die analyses (with die tags)
        output_key: The key in the JSON to aggregate (e.g., "component_loss")
        min_output: Minimum output value for good die
        max_output: Maximum output value for good die
        output_name: Name for output PNG file

    Returns:
        Path to the generated wafer map PNG
    """
    # Parse die coordinates and values from JSON files and their tags
    analyses = {}
    for file in files:
        # Read the JSON output from die analysis
        data = json.loads(file.read_text())
        value = data.get(output_key, np.nan)
        x, y = int(data["die_x"]), int(data["die_y"])
        analyses[(x, y)] = {"value": value, "failed": np.isnan(value)}

    if not analyses:
        raise ValueError("No die analyses found with valid coordinates")

    # Build wafer grid
    die_xys = np.array(sorted(analyses.keys()))
    die_xs = np.unique(die_xys[:, 0])
    die_ys = np.unique(die_xys[:, 1])
    die_x_min, die_x_max = min(die_xs), max(die_xs) + 1
    die_y_min, die_y_max = min(die_ys), max(die_ys) + 1
    nx = die_x_max - die_x_min
    ny = die_y_max - die_y_min
    X, Y = np.mgrid[die_x_min:die_x_max, die_y_min:die_y_max]

    data_grid = np.full((nx, ny), fill_value=np.nan)
    fails = np.full((nx, ny), fill_value=False)
    exists = np.full((nx, ny), fill_value=False)
    toolow = np.full((nx, ny), fill_value=False)
    toohigh = np.full((nx, ny), fill_value=False)

    def set_value(
        x: int,
        y: int,
        *,
        value: float = np.nan,
        failed: bool = False,
    ) -> None:
        exists[x - die_x_min, y - die_y_min] = True
        data_grid[x - die_x_min, y - die_y_min] = value
        if not np.isnan(value):
            if value < min_output:
                toolow[x - die_x_min, y - die_y_min] = True
            if value > max_output:
                toohigh[x - die_x_min, y - die_y_min] = True
        if failed:
            fails[x - die_x_min, y - die_y_min] = True

    for (x, y), analysis in analyses.items():
        set_value(x, y, value=analysis["value"], failed=analysis["failed"])

    # Create wafer map visualization
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[0.9, 0.1], wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Pass-fail wafermap
    ax0.pcolor(X, Y, np.ma.masked_less(~exists, 1), hatch="//", edgecolor="C7", facecolor="none", linewidth=0.0, label="no data")
    ax0.pcolor(X, Y, toolow, cmap=cmap_into_color("blue", 0.8))
    ax0.plot([], [], "s", c=get_color("blue", 0.8), label=f"too low [<{min_output:.2f}]")
    ax0.pcolor(X, Y, toohigh, cmap=cmap_into_color("red", 0.8))
    ax0.plot([], [], "s", c=get_color("red", 0.8), label=f"too high [>{max_output:.2f}]")
    ax0.pcolor(
        X,
        Y,
        exists & (~toolow) & (~toohigh) & (~fails),
        cmap=cmap_into_color("#00ff00", 0.8),
    )
    ax0.plot([], [], "s", c=get_color("#00ff00", 0.8), label="good")
    ax0.pcolor(X, Y, fails, cmap=cmap_into_color("red", 0.2))
    ax0.plot([], [], "s", c=(1, 0, 0, 0.2), label="failed pipeline")

    # Values wafermap
    ax1.pcolormesh(X, Y, data_grid, vmin=min_output, vmax=max_output)
    ax1.pcolor(X, Y, np.ma.masked_less(~exists, 1), hatch="//", edgecolor="C7", facecolor="none", linewidth=0.0)
    ax1.pcolor(X, Y, fails, cmap=cmap_into_color("red", 0.2))

    # Add value labels to both plots
    for a in (ax0, ax1):
        for i in range(nx):
            for j in range(ny):
                value = data_grid[i, j]
                if not np.isnan(value):
                    a.text(
                        i + die_x_min,
                        j + die_y_min,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        a.set_xticks(die_xs)
        a.set_xticks(0.5 * (die_xs[1:] + die_xs[:-1]), minor=True)
        a.set_yticks(die_ys)
        a.set_yticks(0.5 * (die_ys[1:] + die_ys[:-1]), minor=True)
        a.grid(visible=True, which="minor", ls=":")
        a.set_aspect("equal", adjustable="box")
        a.set_xlabel("die x")
        a.set_ylabel("die y")

    plt.suptitle(f"{output_key}")
    fig.legend(ncol=5, bbox_to_anchor=(0.9, -0.2))

    # Save plot
    output_path = files[0].parent / f"{output_name}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    return output_path


def cmap_into_color(
    color: Any,
    alpha: float | None = None,
) -> mcolors.LinearSegmentedColormap:
    """Create a colormap going from transparent into the color."""
    r, g, b, a = get_color(color, alpha)
    name = md5(np.array([r * 255, g * 255, b * 255, a * 255], dtype=np.uint8).tobytes()).hexdigest()[:8]
    return mcolors.LinearSegmentedColormap.from_list(name, [(0, 0, 0, 0), (r, g, b, a)])


def get_color(
    color: Any,
    alpha: float | None = None,
) -> tuple[float, float, float, float]:
    """Get RGBA values from a color."""
    r, g, b, a = mcolors.to_rgba(color)
    if alpha is not None:
        a = alpha
    return r, g, b, a
