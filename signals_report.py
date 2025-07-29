#!/usr/bin/env python3
"""
signals_report.py – analyse **one or many** Buy/Sell signal files, create text summary reports, and (optionally) draw equity curves or print statistics.

Generated parameters
--------------------
* **Number of positive / breakeven trades** – count of Buy‑Sell pairs where PnL ≥ 0
* **Number of negative trades** – count of Buy‑Sell pairs where PnL < 0
* **Total positive / breakeven PnL** – sum of all non‑negative trade results
* **Total negative PnL** – sum of all negative trade results
* **Difference (positive − |negative|)** – net PnL across all trades
* **Win rate [%]** – (positive trades / total trades) × 100
* **Average win / loss** – mean positive PnL divided by mean |negative| PnL (ratio > 1 is desirable)
* **Profit factor** – Σ positive PnL ÷ |Σ negative PnL|

    > 2.0   Very good: highly profitable and robust  
    1.5–2.0 Strong  
    1.1–1.5 Acceptable but could be improved  
    ≈ 1.0   Breakeven: profit barely covers losses  
    < 1.0   Losing strategy: losses exceed profits

* **Expectancy per trade** – net PnL ÷ total trades (average profit/loss each trade)

USAGE examples
--------------
# Report only (summary, no pair list)
python signals_report.py file‑signals.txt

# Report with full Buy‑Sell pair list
python signals_report.py file‑signals.txt --pairs

# Batch mode with pairs, chart & console output
python signals_report.py dir/*/*-signals.txt --pairs --plot --print
"""

import argparse
import csv
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_signals(fname: str) -> List[dict]:
    """Read the CSV‑like text file and return list of rows."""
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


def analyse(data: List[dict], include_pairs: bool) -> Tuple[List[datetime], List[float], List[str]]:
    """Produce equity curve points and formatted report lines."""
    lines: List[str] = []
    dates: List[datetime] = []
    equity_pts: List[float] = []

    pos_tot = neg_tot = 0.0
    pos_cnt = neg_cnt = 0
    equity = 0.0
    pos_trades: List[float] = []
    neg_trades: List[float] = []

    if include_pairs:
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

            if include_pairs:
                lines.append(
                    f"{buy['date'].date()} @ {buy['price']:.4f} → "
                    f"{sell['date'].date()} @ {sell['price']:.4f} = {pnl:.4f}"
                )

            if pnl >= 0:
                pos_tot += pnl
                pos_cnt += 1
                pos_trades.append(pnl)
            else:
                neg_tot += pnl
                neg_cnt += 1
                neg_trades.append(pnl)
            i += 2
        else:
            i += 1

    total_trades = pos_cnt + neg_cnt
    diff = pos_tot - abs(neg_tot)

    # extra metrics
    win_rate = (pos_cnt / total_trades * 100) if total_trades else 0.0
    avg_win = sum(pos_trades) / pos_cnt if pos_cnt else 0.0
    avg_loss = abs(sum(neg_trades) / neg_cnt) if neg_cnt else 0.0
    avg_win_loss = (avg_win / avg_loss) if avg_loss else float("inf") if avg_win else 0.0
    profit_factor = (pos_tot / abs(neg_tot)) if neg_tot else float("inf") if pos_tot else 0.0
    expectancy = diff / total_trades if total_trades else 0.0

    if include_pairs:
        lines.append("-" * 60)

    # summary block
    lines.extend(
        [
            f"Number of positive / breakeven trades:     {pos_cnt}",
            f"Number of negative trades:                 {neg_cnt}",
            f"Total positive / breakeven PnL (≥0):       {pos_tot:.4f}",
            f"Total negative PnL (<0):                  {neg_tot:.4f}",
            f"Difference (positive – |negative|):        {diff:.4f}",
            f"Win rate [%]:                              {win_rate:.2f}",
            f"Average win / loss:                        {avg_win_loss:.4f}",
            f"Profit factor:                             {profit_factor:.4f}",
            f"Expectancy per trade:                      {expectancy:.4f}",
        ]
    )

    return dates, equity_pts, lines


def plot_equity(dates: List[datetime], equity_pts: List[float], title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity_pts, marker="o")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def process_file(in_file: str, *, include_pairs: bool, do_print: bool, do_plot: bool) -> None:
    data = read_signals(in_file)
    dates, equity_pts, report_lines = analyse(data, include_pairs)

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

    parser.add_argument("input_files", nargs="+", help="paths to *-signals.txt files")
    parser.add_argument("--plot", action="store_true", help="draw equity curve for each file")
    parser.add_argument("--print", dest="do_print", action="store_true", help="print statistics to console")
    parser.add_argument("--pairs", action="store_true", help="include Buy‑Sell pair list in the report")

    args = parser.parse_args()

    for path in args.input_files:
        process_file(path, include_pairs=args.pairs, do_print=args.do_print, do_plot=args.plot)


if __name__ == "__main__":
    main()
