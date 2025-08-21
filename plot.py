#!/usr/bin/env python3
"""
Plot metrics from one or many CSV files.

Features
--------
* 2‑D line plot (default)
* 3‑D scatter plot (``--type 3d``)
* Heat‑map / contour plot (``--type heatmap``)
* Surface (``--type surface``)
* Batch mode: supply multiple CSVs and they will be over‑plotted (legend shows file stem).

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

    # Positional arguments
    parser.add_argument(
        "input_file", type=Path, help="input file",
    )

    # Required arguments
    parser.add_argument(
        "--type", choices=["2d", "3d", "heatmap", "surface"], default="2d",
        help="plot type: 2d (default), 3d, heatmap, surface",
    )
    parser.add_argument(
        "--x-col", required=True,
        help="column to use for X values (2-D and 3-D/heatmap/surface)"
    )
    parser.add_argument(
        "--y-col", required=True,
        help="column to use for Y values (2-D and 3-D/heatmap/surface)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--z-col",
        help="column to use for Z values (3-D/heatmap/surface). Ignored for 2-D unless legacy mode."
    )

    parser.add_argument(
        "--output", type=Path,
        help="optional filename to save the plot instead of displaying it",
    )

    # Miscellaneous arguments
    

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────


def read_csv(
    path: Path,
    x_col: str,
    y_col: str,
    z_col: str | None = None
) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """
    Read a CSV file and return the data for the x, y and z columns.

    Parameters
    ----------
    path : Path
        The path to the CSV file.
    x_col : str
        The name of the column to use for the x-axis.
    y_col : str
        The name of the column to use for the y-axis.
    z_col : str | None
        (Optional) The name of the column to use for the z-axis.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series | None]
        The data for the x, y and z columns. If z_col is None, the third element is None.
    """

    # Read the CSV file
    df = pd.read_csv(path)

    # Verify that the file has the required columns
    if x_col in df.columns:
        x_data = df[x_col]
    else:
        raise ValueError(f"Column {x_col} not found in {path}")

    if y_col in df.columns:
        y_data = df[y_col]
    else:
        raise ValueError(f"Column {y_col} not found in {path}")

    if z_col is not None:
        if z_col in df.columns:
            z_data = df[z_col]
        else:
            raise ValueError(f"Column {z_col} not found in {path} and is required for the chosen plot type.")
    else:
        z_data = None

    # Return the DataFrame
    return x_data, y_data, z_data


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d(
    x_col: str,
    x_data: pd.Series,
    y_col: str,
    y_data: pd.Series,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D line plot.

    Parameters
    ----------
    x_col : str
        The name of the column to use for the x-axis.
    x_data : pd.Series
        The data for the x-axis.
    y_col : str
        The name of the column to use for the y-axis.
    y_data : pd.Series
        The data for the y-axis.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The Figure and its Axes so the caller can further tweak or save.
    """

    # Create the figure and axes
    fig, ax = plt.subplots()
    # Plot the data
    ax.plot(x_data, y_data)
    # Set the labels
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    # Set the title
    ax.set_title(f"{x_col} vs {y_col}")
    # Rotate the x-axis labels
    plt.xticks(rotation=45, ha="right")
    # Tighten the layout
    plt.tight_layout()
    # Return the figure and axes
    return fig, ax

def plot_heatmap(
    x_y_col: str,
    x_y_data: pd.Series,
    z_col: str,
    z_data: pd.Series,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap and contour plot.
    
    Parameters
    ----------
    x_y_col : str
        The name of the column to use for the x-axis and y-axis.
    x_y_data : pd.Series
        The data for the x-axis and y-axis.
    z_col : str
        The name of the column to use for the z-axis.
    z_data : pd.Series
        The data for the z-axis.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The Figure and its Axes so the caller can further tweak or save.
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

def plot_surface(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
) -> tuple[plt.Figure, Axes3D]:
    """Return a 3‑D **surface** plot.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing *x_col*, *y_col* and *z_col*.
    x_col, y_col, z_col : str
        Column names to use for X (rows), Y (columns) and the Z value plotted.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, mpl_toolkits.mplot3d.Axes3D]
        The Figure and its 3‑D Axes so the caller can further tweak or save.

    Notes
    -----
    The implementation mirrors :pyfunc:`plot_heatmap` in spirit: we first pivot the
    long‑format *data* frame into a dense 2‑D grid suitable for surface plotting and
    then render it with :pyfunc:`Axes3D.plot_surface`, augmenting the plot with
    thorough, inline commentary for educational purposes.
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
        index=x_col,          # rows   → X‑axis values
        columns=y_col,        # columns→ Y‑axis values
        values=z_col,         # cell values = Z
        aggfunc="mean",
    )

    # Extract the *numeric* vectors representing the axis coordinates.
    # ``pivot_table`` leaves them as (possibly) hetero‑typed Index objects,
    # so we convert to *float64* ndarray for safe numeric processing.
    x_vals = grid.columns.values.astype(float)  # shape = (N,)
    y_vals = grid.index.values.astype(float)    # shape = (M,)

    # ``plot_surface`` expects *meshgrid* inputs: two 2‑D arrays specifying
    # the X & Y coordinates for every Z cell.  NumPy's ``meshgrid`` makes
    # this trivial — note the ``indexing='xy'`` (default) which aligns with
    # the Cartesian‑plane convention used by Matplotlib.
    X, Y = np.meshgrid(x_vals, y_vals)  # both shape=(M, N)
    Z = grid.values.astype(float)       # shape=(M, N)

    # ------------------------------------------------------------------
    # 2. Create a Figure and a single 3‑D Axes.  We use the familiar
    #    "111" (one‑row, one‑col, first subplot) syntax and request the
    #    "3d" projection.  Returning *fig* makes downstream saving easy.
    # ------------------------------------------------------------------
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------------
    # 3. Render the surface.  ``plot_surface`` internally converts the
    #    (X, Y, Z) numeric grids into a *Poly3DCollection* — effectively
    #    a mesh of quadrilaterals.  By default the Z‑value drives the
    #    face‑color via the active colormap; we pick "viridis" because it
    #    is perceptually uniform and color‑blind friendly.
    # ------------------------------------------------------------------
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="inferno",
        edgecolor="none",  # hide gridlines for a cleaner look
        antialiased=True,
    )

    # ------------------------------------------------------------------
    # 4. Decorate axes: labels, title and a color‑bar to communicate the
    #    Z‑to‑color mapping.  We also rotate the view so the surface is
    #    easily interpretable (elev=30°, azim=-135° is a common default).
    # ------------------------------------------------------------------
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"Surface plot of {z_col} vs {x_col} & {y_col}")
    ax.view_init(elev=30, azim=-135)

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=z_col)

    # ------------------------------------------------------------------
    # 5. Return Figure and Axes for optional post‑processing.
    # ------------------------------------------------------------------
    return fig, ax

# ──────────────────────────────────────────────────────────────────────────────
# Plot dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def plot_data(
    x_col: str,
    x_data: pd.Series,
    y_col: str,
    y_data: pd.Series,
    z_col: str,
    z_data: pd.Series | None,
    plot_type: str,
    output: Path | None = None,
) -> None:
    """
    Plot the data.
    
    Parameters
    ----------
    x_data : pd.Series
        The data for the x-axis.
    y_data : pd.Series
        The data for the y-axis.
    z_data : pd.Series | None
        The data for the z-axis.
    plot_type : str
        The type of plot to create.
    output : Path | None
        The path to save the plot to.
    """


    # Dispatch to the specific plot type requested.
    if plot_type == "heatmap":
        fig, ax = plot_heatmap(x_data, y_data, z_data)

    elif plot_type == "surface":
        fig, ax = plot_surface(x_data, y_data, z_data)

    elif plot_type == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_data, y_data, z_data)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"{z_col} vs {x_col} & {y_col}")
    else:  # "2d" line plot
        fig, ax = plot_2d(x_col, x_data, y_col, y_data)

    # ------------------------------------------------------------------
    # Finally either save the figure to disk or show it interactively.
    # ------------------------------------------------------------------
    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Prevent figure leaks in batch workflows.
        print(f"Plot saved to: {output}")
    
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Main entry‑point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Read data from input file
    x_data, y_data, z_data = read_csv(args.input_file, args.x_col, args.y_col, args.z_col)

    # Plot the data
    plot_data(
        args.x_col, x_data,
        args.y_col, y_data,
        args.z_col, z_data,
        args.type,
        args.output
    )

if __name__ == "__main__":
    main()
