#!/usr/bin/env python3
"""
Generate a bar chart based on the 'Difference (positive – |negative|)' lines
from all *-signals-report.txt files in the specified directory (default: current),
sorted from the smallest difference on the left to the largest on the right.

Usage:
    python signals_histogram.py [directory]

The script scans for files ending with '-signals-report.txt', extracts the numeric
value from the 'Difference' line, sorts the results ascending by that value, and
displays a bar chart with file names on the x‑axis and differences on the y‑axis
(green for positive, red for negative).
"""
import glob
import os
import re
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

DIFF_REGEX = re.compile(r"Difference .*: *([+-]?\d+\.\d+)")


def extract_difference(path: str) -> float | None:
    """Return the float after 'Difference' in the given report file, or None."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = DIFF_REGEX.search(line)
            if match:
                return float(match.group(1))
    return None


def collect_differences(directory: str) -> Tuple[List[str], List[float]]:
    """Return parallel lists of labels and difference values from matching files."""
    pattern = os.path.join(directory, "*-signals-report.txt")
    files = sorted(glob.glob(pattern))
    labels, values = [], []
    for file in files:
        diff = extract_difference(file)
        if diff is not None:
            base = os.path.basename(file)
            label = base[:-len("-signals-report.txt")] if base.endswith("-signals-report.txt") else base
            labels.append(label)
            values.append(diff)
        else:
            print(f"[warn] No difference found in {file}; skipping", file=sys.stderr)
    return labels, values


def sort_by_value(labels: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    """Return labels and values sorted ascending by value."""
    pairs = sorted(zip(labels, values), key=lambda p: p[1])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_differences(labels: List[str], values: List[float]) -> None:
    if not values:
        print("No difference values to plot.")
        return

    plt.figure(figsize=(max(8, len(values) * 0.8), 6))
    bars = plt.bar(labels, values)

    # Color bars: green for >=0, red for <0
    for bar, val in zip(bars, values):
        bar.set_color("green" if val >= 0 else "red")
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    plt.ylabel("Difference (positive – |negative|)")
    plt.title("Differences from signals reports (sorted ascending)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    labels, values = collect_differences(directory)
    if not labels:
        print(f"No *-signals-report.txt files with valid data found in {directory}")
        sys.exit(1)

    labels, values = sort_by_value(labels, values)
    plot_differences(labels, values)


if __name__ == "__main__":
    main()
