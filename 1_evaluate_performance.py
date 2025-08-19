#!/usr/bin/env python3
"""
signals_report.py – analyse **one or many** Buy/Sell signal files and create text summary reports.

(No plotting – the script only generates output files.)

Generated parameters
--------------------
* **Number of positive / breakeven trades** – count of Buy‑Sell pairs where PnL ≥ 0
* **Number of negative trades** – count of Buy‑Sell pairs where PnL < 0
* **Total positive / breakeven PnL** – sum of all non‑negative trade results
* **Total negative PnL** – sum of all negative trade results
* **Difference (positive − |negative|)** – net PnL across all trades
* **Win rate [%]** – (positive trades / total trades) × 100
* **Average win / loss** – mean positive PnL divided by mean |negative| PnL (ratio > 1 desirable)
* **Profit factor** – Σ positive PnL ÷ |Σ negative PnL|  
  > 2.0 — Very good | 1.5–2.0 — Strong | 1.1–1.5 — OK | ≈1 — Breakeven | <1 — Losing
* **Expectancy per trade** – net PnL ÷ total trades (average profit/loss each trade)

USAGE examples
--------------
# Explicit file list (sequential)
python signals_report.py out/*-signals.txt

# Directory search with glob pattern (recursive, 8 parallel jobs)
python signals_report.py -d out -p "*-signals.txt" -r --jobs 8 --pairs --print
"""

import argparse
import csv
import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ────────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
            description="Generate signal reports (text‑only)."
    )

    parser.add_argument(
            "--out-dir", default="out/",
            help="Root output directory (default: current)")

    # Processing & output options
    parser.add_argument(
            "--jobs", type=int, default=1,
            help="number of parallel jobs/processes (default 1)")
    parser.add_argument(
            "--include-pairs", action="store_true",
            help="include Buy‑Sell pair list in report")
    parser.add_argument(
            "--print", dest="do_print", action="store_true",
            help="print statistics to console")

    parser.add_argument("files", nargs="*",
                        help="explicit paths to *-allsignals.txt files")

    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# Core analytics helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_signals(path: Path) -> pd.DataFrame:
    """
    Read a CSV‑like text file into a pandas DataFrame.
    The file is expected to have the following columns:
    - DATETIME: date and time of the signal
    - TICKER: ticker of the asset
    - PER: period of the signal
    - SMA_: Simple Moving Average
    - SIG_x_y: signal type (Buy or Sell) for SMA_x vs SMA_y

    PARAMETERS
    ----------
    path: Path
        The path to the file to read

    RETURNS
    -------
    pd.DataFrame
        A DataFrame with columns "DATETIME", "TICKER", "PER", "SMA_", and "SIG_x_y"
    """
    return pd.read_csv(
            path,
            parse_dates=["DATETIME"],   # parse the DATETIME column as a datetime object
            index_col="DATETIME",       # set the DATETIME column as the index
            dtype={"TICKER": "string",
                   "PER"   : "category"},
            # low_memory=False,         # uncomment if you see dtypes errors
            #na_values=["", "NA", "NaN"]# optional: explicit NA markers (empty fields already -> NaN)
    )


def analyse(df: pd.DataFrame, include_pairs: bool) -> List[str]:
    """
    Analyze the signals and return a list of report lines.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with columns "DATE", "Price", and "Signal"
    include_pairs : bool
        Whether to include the Buy‑Sell pair list in the report.

    Returns
    -------
    List[str]
        The list of report lines.

    Notes
    -----
    - Calculates the total positive and negative PnL.
    - Calculates the number of positive and negative trades.
    - Calculates the average win and loss.
    - Calculates the profit factor and expectancy.
    """
    lines: List[str] = []
    pos_tot = neg_tot = 0.0
    pos_cnt = neg_cnt = 0
    pos_trades: List[float] = []
    neg_trades: List[float] = []

    if include_pairs:
        lines.extend(["Buy‑Sell pairs and differences", "-" * 60])

    i = 0
    while i < len(df) - 1:
        if df.iloc[i]["Signal"] == "Buy" and df.iloc[i + 1]["Signal"] == "Sell":
            buy, sell = df.iloc[i], df.iloc[i + 1]
            pnl = sell["Price"] - buy["Price"]
            if include_pairs:
                lines.append(
                    f"{buy['DATE'].date()} @ {buy['Price']:.4f} → "
                    f"{sell['DATE'].date()} @ {sell['Price']:.4f} = {pnl:.4f}"
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

    win_rate = (pos_cnt / total_trades * 100) if total_trades else 0.0
    avg_win = sum(pos_trades) / pos_cnt if pos_cnt else 0.0
    avg_loss = abs(sum(neg_trades) / neg_cnt) if neg_cnt else 0.0
    avg_win_loss = (avg_win / avg_loss) if avg_loss else float("inf") if avg_win else 0.0
    profit_factor = (pos_tot / abs(neg_tot)) if neg_tot else float("inf") if pos_tot else 0.0
    expectancy = diff / total_trades if total_trades else 0.0

    if include_pairs:
        lines.append("-" * 60)

    lines.extend(
        [
            f"Number of positive / breakeven trades:     {pos_cnt}",
            f"Number of negative trades:                 {neg_cnt}",
            f"Total positive / breakeven PnL (≥0):       {pos_tot:.4f}",
            f"Total negative PnL (<0):                   {neg_tot:.4f}",
            f"Difference (positive – |negative|):        {diff:.4f}",
            f"Win rate [%]:                              {win_rate:.2f}",
            f"Average win / loss:                        {avg_win_loss:.4f}",
            f"Profit factor:                             {profit_factor:.4f}",
            f"Expectancy per trade:                      {expectancy:.4f}",
        ]
    )
    return lines

# ────────────────────────────────────────────────────────────────────────────────
# Worker function (for parallel execution)
# ────────────────────────────────────────────────────────────────────────────────

def process_file(path: Path, include_pairs: bool) -> Tuple[Path, List[str]]:
    """
    Process a single file.

    Parameters
    ----------
    path : Path
        The path to the file to process.
    include_pairs : bool
        Whether to include the Buy‑Sell pair list in the report.

    Returns
    -------
    Tuple[Path, List[str]]
        The path to the file and the list of report lines.

    Notes
    -----
    - Reads the signals from the file.
    - Analyzes the signals.
    - Writes the report to a file.
    """
    # Read the signals from the file
    data = read_signals(path)

    # Analyse the signals
    report_lines = analyse(data, include_pairs)

    # Write the report to a file
    out_file = path.with_name(path.stem + "-report" + path.suffix)
    out_file.write_text("\n".join(report_lines))

    # Return the path and the report lines
    return path, report_lines

# ────────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main function.

    Returns
    -------
    None

    Notes
    -----
    - Parses command line arguments.
    - Creates output directory if it doesn't exist.
    - Runs sequentially or in parallel.
    - Prints results to console if --print is set.
    """
    args = parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create a “worker” function with most of its parameters already  bound (curried) so that each call only
    # needs the filename.
    #   process_file(...)   – user-defined function that processes one CSV or JSON
    #   include_pairs       – whether to include the Buy‑Sell pair list in the report
    worker = partial(
        process_file,
        include_pairs=args.include_pairs
    )

    # Run sequentially
    if args.jobs == 1 or len(args.files) == 1:
        for _ in map(worker, args.files):
            pass
    # Run in parallel
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map() returns an iterator, not a list
            for _ in executor.map(worker, args.files):
                pass

if __name__ == "__main__":
    main()