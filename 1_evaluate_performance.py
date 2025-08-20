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
import numpy as np
import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict,List,Tuple


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
            help="Root output directory (default: out/)")
    parser.add_argument(
            "--jobs", type=int, default=1,
            help="number of parallel jobs/processes (default 1)")
    parser.add_argument(
            "files", nargs="*",
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
            low_memory=False,         # uncomment if you see dtypes errors
            # na_values=["", "NA", "NaN"]# optional: explicit NA markers (empty fields already -> NaN)
    )


def analyse(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze Buy/Sell signals contained in *signals_df* and compute per‑signal
    performance metrics. Returns a DataFrame with one row per signal column.

    Expected columns
    ----------------
    - PRICE (preferred) or CLOSE – used for entry/exit prices
    - One or more signal columns named 'Sig_<short>_<long>' (case insensitive),
      containing "Buy"/"Sell" on crossover bars and NaN elsewhere.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Input data with prices and signal columns.

    Returns
    -------
    pd.DataFrame
        Index: signal column name (e.g., 'Sig_20_100')
        Columns:
        - "POS_CNT": number of positive trades
        - "NEG_CNT": number of negative trades
        - "POS_PnL": total positive PnL
        - "NEG_PnL": total negative PnL
        - "NET_PnL": difference (positive – |negative|)
        - "WIN_RATE": win rate [%]
        - "AVG_WIN_LOSS": average win / loss
        - "PROFIT_FACTOR": profit factor
        - "EXPECTANCY": expectancy per trade
    
    Notes
    -----
    - Strategy evaluated here is a simple long-only pairing: enter on "Buy",
      exit on the next "Sell". Sells before the first Buy are ignored.
    - Breakeven trades (PnL == 0) are counted as positive for POS_CNT.
    """
    # Always use PRICE column for entry/exit prices
    prices = pd.to_numeric(signals_df["PRICE"], errors="coerce")         # 1-D Series of floats aligned to the DataFrame index
    
    # Detect all signal columns like "Sig_20_100"
    sig_cols = [c for c in signals_df.columns if c.startswith("Sig_")]  # filter only signal columns generated upstream

    # If there are no signal columns, return an empty result table with the right schema.
    if not sig_cols:
        return pd.DataFrame(                                               # create an empty frame with expected columns and no rows
            columns=[
                "POS_CNT", "NEG_CNT", "POS_PnL", "NEG_PnL",
                "NET_PnL", "WIN_RATE", "AVG_WIN_LOSS", "PROFIT_FACTOR", "EXPECTANCY",
            ],
            dtype="float64",                                               # numeric types by default
        )

    # Prepare a list of per-signal metric rows; we will assemble a DataFrame at the end.
    rows: list[dict] = []                                                  # each element will be a dict of metrics for one Sig_* column

    # Compute metrics per signal column.
    # Iterate each "Sig_<short>_<long>" column independently
    for sig_name in sig_cols:                                              
        # Store as pandas StringDtype for safe comparisons incl. NaN
        sig_series = signals_df[sig_name].astype("string")
        open_entry_time = None
        open_entry_price = None
        # Collect PnL for each completed Buy→Sell pair
        trade_pnls: list[float] = []

        # Walk forward in time once; pair each Buy with the *next* Sell.
        # Iterate in chronological index order: (timestamp, "Buy"/"Sell"/<NA>)
        for ts, label in sig_series.items():
            # Skip rows without a signal; nothing to do
            if pd.isna(label):                                             
                continue

            # Entry condition: go long at the Buy bar's price (close/price column)
            if label == "Buy":                                             
                # Only open a new position if we are flat; repeated "Buy" before a "Sell" is ignored.
                if open_entry_time is None:                                # we’re flat — can open a long
                    # Price at the Buy bar; may be NaN (we’ll guard below)
                    entry_price = prices.loc[ts]                           
                    # Only accept valid numeric prices
                    if pd.notna(entry_price):                              
                        # Remember when we entered
                        open_entry_time = ts                               
                        # and at what price
                        open_entry_price = float(entry_price)              
                # else: already long; ignore extra Buy until a Sell closes the trade

            # Exit condition: close long at the Sell bar's price
            elif label == "Sell":                                          
                # Only close if there is an open long
                if open_entry_time is not None:                            
                    # Price at the Sell bar
                    exit_price = prices.loc[ts]                           
                    # Ensure both entry and exit prices are valid
                    if pd.notna(exit_price) and pd.notna(open_entry_price):
                        # Raw PnL in price units (breakeven allowed)
                        pnl = float(exit_price) - float(open_entry_price)  
                        # Store this completed trade’s result
                        trade_pnls.append(pnl)                             
                    # Whether we could compute PnL or not, the position is considered closed on Sell.
                    open_entry_time = None      # flat after a Sell
                    open_entry_price = None     # clear the stored entry price
                # else: Sell without a preceding Buy — ignore (no open position to close)

        # At the end of the series, if we still have an open Buy, it remains unmatched → ignore (no closing Sell).
        # Now compute statistics from the collected trade PnLs for this signal column.
        total_trades = len(trade_pnls)                                     # Number of completed Buy→Sell pairs
        pos_pnls = [p for p in trade_pnls if p >= 0.0]                     # Non-negative outcomes (breakeven counts as a “win”)
        neg_pnls = [p for p in trade_pnls if p <  0.0]                     # Strictly negative outcomes

        pos_cnt = float(len(pos_pnls))                                     # Convert to float for consistent dtype downstream
        neg_cnt = float(len(neg_pnls))
        pos_sum = float(sum(pos_pnls)) if pos_pnls else 0.0                # Sum positive PnL (0.0 if none)
        neg_sum = float(sum(neg_pnls)) if neg_pnls else 0.0                # Sum negative PnL (≤ 0.0; 0.0 if none)
        net_sum = pos_sum + neg_sum                                        # net PnL across all trades

        # Win rate is the fraction of non-negative trades.
        win_rate = (pos_cnt / total_trades * 100.0) if total_trades > 0 else np.nan

        # Average win / loss ratio:
        #   mean(positive PnL) / mean(|negative PnL|)
        avg_win = (pos_sum / pos_cnt) if pos_cnt > 0 else np.nan           # mean win (NaN if no winners)
        avg_loss_abs = (abs(neg_sum) / neg_cnt) if neg_cnt > 0 else np.nan # mean loss magnitude (NaN if no losers)
        if pos_cnt > 0 and neg_cnt > 0:
            avg_win_loss = avg_win / avg_loss_abs                          # finite ratio when both sides exist
        elif pos_cnt > 0 and neg_cnt == 0:
            avg_win_loss = np.inf                                          # no losses → ratio tends to infinity
        elif pos_cnt == 0 and neg_cnt > 0:
            avg_win_loss = 0.0                                             # no wins → ratio is 0
        else:
            avg_win_loss = np.nan                                          # no trades at all

        # Profit factor is Σ wins / |Σ losses|.
        if abs(neg_sum) > 0.0:
            profit_factor = pos_sum / abs(neg_sum)                         # Finite value when both sides exist
        elif pos_sum > 0.0:
            profit_factor = np.inf                                         # Wins but no losses → infinite PF
        elif total_trades > 0:
            profit_factor = 0.0                                            # Only losses or all zero PnL → PF 0
        else:
            profit_factor = np.nan                                         # No trades at all

        # Expectancy per trade is net PnL averaged over the number of trades.
        expectancy = (net_sum / total_trades) if total_trades > 0 else np.nan

        # Assemble the metrics for this signal column into one row.
        rows.append({
            "SIG_x_y": sig_name,                                           # keep the signal column name to set as index later
            "POS_CNT": pos_cnt,
            "NEG_CNT": neg_cnt,
            "POS_PnL": pos_sum,
            "NEG_PnL": neg_sum,
            "NET_PnL": net_sum,
            "WIN_RATE": win_rate,
            "AVG_WIN_LOSS": avg_win_loss,
            "PROFIT_FACTOR": profit_factor,
            "EXPECTANCY": expectancy,
        })

    # Turn the list of dicts into a DataFrame.
    # Shape: (num_signals, 10 columns including "SIG_x_y")
    output_df = pd.DataFrame(rows)
    # Index by signal name (e.g., "Sig_20_100") as requested
    output_df = output_df.set_index("SIG_x_y")

    return output_df     

# ────────────────────────────────────────────────────────────────────────────────
# Worker function (for parallel execution)
# ────────────────────────────────────────────────────────────────────────────────

def process_file(path: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Process a single file.

    Parameters
    ----------
    path : Path
        The path to the file to process.

    Returns
    -------
    Tuple[Path, pd.DataFrame]
        The path to the file and the DataFrame with the report.

    Notes
    -----
    - Reads the signals from the file.
    - Analyzes the signals.
    - Writes the report to a file.
    """
    # Ensure path is a Path object
    path = Path(path)

    # Read the signals from the file
    data = read_signals(path)

    # Analyse the signals
    output_df = analyse(data)

    # Write the report to a file
    out_file = path.with_name(path.stem + "-report" + path.suffix)
    output_df.to_csv(out_file, float_format="%.4f")

    # Return the path and the report lines
    return path, output_df

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
    worker = partial(
        process_file
    )

    # Run sequentially
    if args.jobs == 1 or len(args.files) == 1:
        for _ in map(process_file, args.files):
            pass
    # Run in parallel
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map() returns an iterator, not a list
            for _ in executor.map(process_file, args.files):
                pass

if __name__ == "__main__":
    main()