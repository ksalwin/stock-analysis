#!/usr/bin/env python3
"""
Plot SMA-grid back‑test metrics from one or many CSV files.

Features
--------
* 2‑D line plot (default)
* 3‑D scatter plot (``--type 3d``)
* Heat‑map / contour plot (``--type heatmap``)
* Batch mode: supply multiple CSVs and they will be over‑plotted (legend shows file stem).

Usage examples
--------------
# Default 2‑D line plot
./plot_sma_heatmap.py --type 2d results.csv

# 3‑D scatter plot saved to PNG
./plot_sma_heatmap.py --type 3d -o plot.png results1.csv results2.csv

# Heat‑map shown on screen
./plot_sma_heatmap.py --type heatmap sma_grid.csv
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
    parser = argparse.ArgumentParser(description="Plot SMA-grid metrics from files.")

    parser.add_argument("--type",
                        choices=["2d", "3d", "heatmap"], default="2d",
                        help="plot type: 2d (default), 3d or heatmap")

    parser.add_argument("--output",
                        type=Path,
                        help="optional filename to save the plot instead of displaying it")

    parser.add_argument("input_files",
                        type=Path,
                        nargs="+",
                        help="input file(s)")

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_data(input_files: list[Path], plot_type: str = "2d", output: Path | None = None) -> None:
    """Read *input_files* (one or many) and create the chosen plot."""
    dfs = []
    for fp in input_files:
        df = pd.read_csv(fp)
        if df.shape[1] < 3:
            raise ValueError(f"{fp} must contain at least three columns")
        df["__source__"] = fp.stem
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    x_col, y_col, z_col = data.columns[:3]

    if plot_type == "heatmap":
        # Pivot to grid and plot as an image + contour lines
        grid = data.pivot_table(index=x_col, columns=y_col, values=z_col, aggfunc="mean")
        x_vals = grid.columns.values
        y_vals = grid.index.values
        Z = grid.values.astype(float)

        fig, ax = plt.subplots()
        im = ax.imshow(Z, origin="lower", aspect="auto")
        cs = ax.contour(Z, colors="black", linewidths=0.5, alpha=0.7)
        ax.clabel(cs, fmt="%.1f", fontsize=8)

        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels(y_vals)
        ax.set_xlabel(y_col)
        ax.set_ylabel(x_col)
        ax.set_title(f"Heat‑map of {z_col}")
        fig.colorbar(im, ax=ax, label=z_col)

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

    else:  # "2d"
        fig, ax = plt.subplots()
        for src, grp in data.groupby("__source__"):
            combined = grp[[x_col, y_col]].apply(lambda r: f"({r[x_col]},{r[y_col]})", axis=1)
            ax.plot(combined, grp[z_col], marker="o", label=src)
        ax.set_xlabel(f"({x_col}, {y_col})")
        ax.set_ylabel(z_col)
        ax.set_title(z_col)
        plt.xticks(rotation=45, ha="right")
        if data["__source__"].nunique() > 1:
            ax.legend()
        plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output}")
    else:
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    plot_data(args.input_files, args.type, args.output)

if __name__ == "__main__":
    main()
