#!/usr/bin/env python3
"""
Plot SMA‑grid metrics from one or many CSV files.

Features
--------
* 2‑D line plot (default)
* 3‑D scatter plot (``--type 3d``)
* Heat‑map / contour plot (``--type heatmap``)
* Batch mode: supply multiple CSVs and they will be over‑plotted (legend shows file stem).

This version extracts the heat‑map generation into a dedicated ``plot_heatmap`` helper,
and enriches that helper with detailed, step‑by‑step comments for clarity.

Usage examples
--------------
# Default 2‑D line plot
./plot.py --type 2d results.csv

# 3‑D scatter plot saved to PNG
./plot.py --type 3d -o plot.png results1.csv results2.csv

# Heat‑map shown on screen
./plot.py --type heatmap sma_grid.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3‑D projection

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot SMA‑grid metrics from files.")

    parser.add_argument(
        "--type",
        choices=["2d", "3d", "heatmap"],
        default="2d",
        help="plot type: 2d (default), 3d or heatmap",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="optional filename to save the plot instead of displaying it",
    )

    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="input file(s)",
    )

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Return a heat‑map & contour plot.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing *x_col*, *y_col* and *z_col*.
    x_col, y_col, z_col : str
        Column names to use for X (rows), Y (columns) and the Z value plotted.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The Figure and its primary Axes so the caller can further tweak or save.
    """

    # ------------------------------------------------------------------
    # 1. Reshape the long‑format *data* frame into a 2‑D grid (a pivot
    #    table) so that each unique X‑value becomes a **row** and each
    #    unique Y‑value becomes a **column**.  The cell holds the mean of
    #    Z for that (X, Y) pair.  Example transformation:
    #
    #        ┌───────── original (long) format ─────────┐
    #        │ SMA  LMA   score │
    #        │  5    20   0.83  │
    #        │  5    50   0.79  │    →  pivot_table →   LMA→   20    50   100
    #        │ 10    20   0.81  │                       SMA ↓ ┌───────────────┐
    #        │ 10    50   0.77  │                             │ 0.83  0.79  0.76│
    #        │ 10    100  0.74  │                             │ 0.81  0.77  0.74│
    #        └──────────────────┘                             └───────────────┘
    #
    #    Duplicate (SMA, LMA) pairs—if any—are *averaged* via ``aggfunc="mean"``.
    #    The resulting 2‑D matrix (Z) is what ``imshow`` and ``contour`` expect.
    # ------------------------------------------------------------------
    grid = data.pivot_table(
        index=x_col,          # rows   → X‑axis (SMA in the example)
        columns=y_col,        # columns→ Y‑axis (LMA)
        values=z_col,         # cell values = Z (score)
        aggfunc="mean",      # handle duplicates by averaging their scores
    )

    # Extract axis tick labels and the raw numeric matrix from the pivot.
    x_vals = grid.columns.values   # Unique Y values → X‑axis tick labels in image space
    y_vals = grid.index.values     # Unique X values → Y‑axis tick labels in image space
    Z = grid.values.astype(float)  # The ndarray used for plotting (rows=y, cols=x)

    # ------------------------------------------------------------------
    # 2. Create the Matplotlib Figure/Axes pair. A single Axes is enough
    #    here, but returning *fig* as well allows the caller to save it
    #    or embed it into a larger multi‑panel layout.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()

    # ------------------------------------------------------------------
    # 3. Render the heat‑map (pixel view) and overlay contour lines to
    #    highlight iso‑value boundaries (areas where Z is constant).
    # ------------------------------------------------------------------
    im = ax.imshow(
        Z,
        origin="lower",  # Put (row=0,col=0) at bottom‑left like a Cartesian plane
        aspect="auto",   # Stretch to fill allocated space; keep data pixels square‑ish
        cmap='plasma',
        #vmax=5.0,
    )
    cs = ax.contour(
        Z,
        colors="black",
        linewidths=0.5,
        alpha=0.7,
    )
    ax.clabel(cs, fmt="%.1f", fontsize=8)  # Inline labels on contour lines

    # ------------------------------------------------------------------
    # 4. Decorate axes: ticks, labels, title, and color‑bar.
    # ------------------------------------------------------------------
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels(y_vals)
    ax.set_xlabel(y_col)
    ax.set_ylabel(x_col)
    ax.set_title(f"Heat‑map of {z_col}")

    fig.colorbar(im, ax=ax, label=z_col)  # Show mapping from color→value

    # ------------------------------------------------------------------
    # 5. Return Figure and Axes for optional post‑processing.
    # ------------------------------------------------------------------
    return fig, ax

# ──────────────────────────────────────────────────────────────────────────────
# Plot dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def plot_data(
    input_files: list[Path],
    plot_type: str = "2d",
    output: Path | None = None,
) -> None:
    """Read *input_files* (one or many) and create the chosen plot."""

    # --------------------------------------------------------------
    # Read and concatenate all CSVs so we can treat them uniformly.
    # --------------------------------------------------------------
    dfs: list[pd.DataFrame] = []
    for fp in input_files:
        df = pd.read_csv(fp)
        if df.shape[1] < 3:
            raise ValueError(f"{fp} must contain at least three columns")
        df["__source__"] = fp.stem  # Track source file for legend grouping
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    x_col, y_col, z_col = data.columns[:3]

    # Dispatch to the specific plot type requested.
    if plot_type == "heatmap":
        fig, ax = plot_heatmap(data, x_col, y_col, z_col)

    elif plot_type == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for src, grp in data.groupby("__source__"):
            ax.scatter(grp[x_col], grp[y_col], grp[z_col], label=src)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"{z_col} vs {x_col} & {y_col}")
        if data["__source__"].nunique() > 1:
            ax.legend()

    else:  # "2d" line plot
        fig, ax = plt.subplots()
        for src, grp in data.groupby("__source__"):
            # Combine the X & Y columns to create a unique tick label per point
            combined = grp[[x_col, y_col]].apply(lambda r: f"({r[x_col]},{r[y_col]})", axis=1)
            ax.plot(combined, grp[z_col], marker="o", label=src)
        ax.set_xlabel(f"({x_col}, {y_col})")
        ax.set_ylabel(z_col)
        ax.set_title(z_col)
        plt.xticks(rotation=45, ha="right")
        if data["__source__"].nunique() > 1:
            ax.legend()
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Finally either save the figure to disk or show it interactively.
    # ------------------------------------------------------------------
    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Prevent figure leaks in batch workflows.
        print(f"Plot saved to: {output}")
    else:
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Main entry‑point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    plot_data(args.input_files, args.type, args.output)

if __name__ == "__main__":
    main()
