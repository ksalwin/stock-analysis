#!/usr/bin/env python3
"""
Plot SMA-grid back-test results.

Usage examples
--------------
# Show a 2-D plot (default)
./plot.py results.txt

# Make an interactive 3-D scatter and save it
./plot.py results.txt --type 3d -o plot.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (needed for 3-D plot)

def plot_data(csv_file: Path, plot_type: str = "2d", output: Path | None = None) -> None:
    df = pd.read_csv(csv_file)
    x_col, y_col, z_col = df.columns[:3]  # use only the first three columns

    if plot_type == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df[x_col], df[y_col], df[z_col], marker="o")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"{z_col} vs {x_col} & {y_col}")
    else:                               # ---- 2-D variant ----
        # Combine column-1 & column-2 values into a single categorical x axis label
        combined = df[[x_col, y_col]].apply(lambda r: f"({r[x_col]},{r[y_col]})", axis=1)
        fig, ax = plt.subplots()
        ax.plot(combined, df[z_col], marker="o")
        ax.set_xlabel(f"({x_col}, {y_col})")
        ax.set_ylabel(z_col)
        ax.set_title(z_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output}")
    else:
        plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SMA-grid back-test metrics from CSV.")
    parser.add_argument("csv_file", type=Path, help="Input CSV file")
    parser.add_argument("--type", choices=["2d", "3d"], default="2d",
                        help="Choose '2d' (default) or '3d' plot")
    parser.add_argument("-o", "--output", type=Path,
                        help="Optional filename to save the plot instead of showing it")
    args = parser.parse_args()

    plot_data(args.csv_file, args.type, args.output)

if __name__ == "__main__":
    main()

