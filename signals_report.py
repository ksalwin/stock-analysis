#!/usr/bin/env python3
"""
signals_report.py  –  analyse **one or many** Buy/Sell signal files, create text reports, and (optionally) draw equity curves and/or print statistics.

USAGE examples
--------------
# Single file → report only (default)
python signals_report.py out-wse_stocks/abc/abc-20-100-signals.txt

# Batch mode → many files, report only
python signals_report.py out-wse_stocks/*/*-signals.txt

# One file with chart
python signals_report.py out-wse_stocks/abc/abc-20-100-signals.txt --plot

# Batch mode with chart and console print
python signals_report.py out-wse_stocks/abc/abc-20-100-signals.txt out-wse_stocks/xyz/xyz-20-100-signals.txt --plot --print
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_signals(fname: str) -> List[dict]:
    """Read the CSV‑like text file and return a list of dicts."""
    with open(fname, newline="") as f:
        rdr = csv.DictReader(f)
        return [
            {
                "date": datetime.strptime(row["DATE"], "%Y-%m-%d"),
                "price": float(row["Price"]),
                "signal": row["Signal"].strip(),
            }
            for row in rdr
        ]


def analyse(data: List[dict]) -> Tuple[List[datetime], List[float], List[str]]:
    """Produce equity curve points and formatted report lines."""
    lines: List[str] = []
    dates: List[datetime] = []
    equity_pts: List[float] = []

    pos_tot = neg_tot = 0.0
    pos_cnt = neg_cnt = 0
    equity = 0.0

    lines.append("Buy‑Sell pairs and differences")
    lines.append("-" * 60)

    i = 0
    while i < len(data) - 1:
        if data[i]["signal"] == "Buy" and data[i + 1]["signal"] == "Sell":
            buy, sell = data[i], data[i + 1]
            pnl = sell["price"] - buy["price"]
            equity += pnl
            dates.append(sell["date"])
            equity_pts.append(equity)

            lines.append(
                f"{buy['date'].date()} @ {buy['price']:.4f} → "
                f"{sell['date'].date()} @ {sell['price']:.4f} = {pnl:.4f}"
            )

            if pnl >= 0:
                pos_tot += pnl
                pos_cnt += 1
            else:
                neg_tot += pnl
                neg_cnt += 1
            i += 2
        else:
            i += 1

    lines.append("-" * 60)
    diff = pos_tot - abs(neg_tot)
    lines.extend(
        [
            f"Number of positive / breakeven trades:     {pos_cnt}",
            f"Number of negative trades:                 {neg_cnt}",
            f"Total positive / breakeven PnL (≥0):       {pos_tot:.4f}",
            f"Total negative PnL (<0):                  {neg_tot:.4f}",
            f"Difference (positive – |negative|):        {diff:.4f}",
        ]
    )

    return dates, equity_pts, lines


def plot_equity(dates: List[datetime], equity_pts: List[float], title: str) -> None:
    """Render an equity curve for a single file."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity_pts, marker="o")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def process_file(in_file: str, *, do_print: bool, do_plot: bool) -> None:
    """Read, analyse and output results for one file."""
    data = read_signals(in_file)
    dates, equity_pts, report_lines = analyse(data)

    # write report file next to input file
    root, ext = os.path.splitext(in_file)
    out_file = f"{root}-report{ext}"
    with open(out_file, "w") as f:
        f.write("\n".join(report_lines))

    if do_print:
        print(f"\n=== {in_file} ===")
        print("\n".join(report_lines))

    if do_plot:
        plot_equity(dates, equity_pts, title=os.path.basename(in_file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal reports from one or many Buy/Sell files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="paths to *-signals.txt files (one or many)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="draw equity curve for each input file",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="print statistics to console",
    )

    args = parser.parse_args()

    for in_file in args.input_files:
        process_file(in_file, do_print=args.do_print, do_plot=args.plot)


if __name__ == "__main__":
    main()
