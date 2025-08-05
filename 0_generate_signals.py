#!/usr/bin/env python3
"""
Golden‑Cross SMA Signal Generator — Batch & Parallel
===================================================
Generate Buy/Sell signals with a Golden‑Cross (short SMA crossing long SMA)
strategy for **one or more** OHLC text files. Results are saved as `.txt`
CSV‑style files; no charts are produced. Supports **parallel execution** via a
`--jobs/-j` switch.

Usage
-----
    python golden_cross_sma.py <SMA_short> <SMA_long> <files...> [options]

Options
~~~~~~~
  -o, --output DIR        Root directory for results (default: .)
  -j, --jobs N            Run up to N files in parallel (default: 1 ⇒ sequential)
  --show-no-signal        Also print tickers whose latest signal is "No signal …"

Example
~~~~~~~
Process many files with 8 parallel workers and skip “No signal” tickers:

    python golden_cross_sma.py 20 100 data/*.txt -o ./out -j 8

Outputs (stored in `<OUTPUT_DIR>/<ticker>/`)
-------------------------------------------
`<base>-<short>-<long>-signals.txt`     – Date, Price, Signal (Buy/Sell)

`<base>` is the input filename without extension (e.g. `slv`).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from functools import partial
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Golden‑Cross SMA Buy/Sell signals in batch mode with optional parallelism."
    )

    parser.add_argument("sma_short", type=int, help="Short SMA period (integer)")
    parser.add_argument("sma_long", type=int, help="Long SMA period (integer)")
    parser.add_argument("files", nargs="+", help="One or more input data files")
    parser.add_argument("--output", default=".", help="Root output directory (default: current)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument(
        "--show-no-signal",
        dest="show_no",
        action="store_true",
        help="Also print tickers whose latest signal is 'No signal …'",
    )

    args = parser.parse_args()

    # --- Argument validation
    if args.sma_short >= args.sma_long:
        parser.error("sma_short must be smaller than sma_long")

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_data_from_file(path: str) -> pd.DataFrame:
    # Read the file (returns data frame)
    df = pd.read_csv(
        path,
        header=0,   # First row is the header
        dtype={"<TICKER>": "string",
               "<DATE>"  : "string",
               "<TIME>"  : "string"}, # Convert ticker to str (without it would be obj)
    )

    # Remove '<>' from header names
    df.columns = df.columns.str.strip("<>")

    # Ensure DATE and TIME is the right length - needed for valid combining
    df["DATE"] = df["DATE"].str.zfill(8)    # YYYYMMDD
    df["TIME"] = df["TIME"].str.zfill(6)    # HHMMSS

    # Combine DATE and TIME to DATETIME
    df["DATETIME"] = pd.to_datetime(
        df["DATE"] + df["TIME"],
        format="%Y%m%d%H%M%S",
        errors="coerce" # turn any malformed rows into NaT so the don't crash the parse
    )

    # Drop the original DATE and TIME columns - replaced by DATETIME
    df = df.drop(columns=["DATE", "TIME"])

    # Columns check (must be before set_index)
    required = {"TICKER", "DATETIME", "CLOSE"}
    if not required.issubset(df.columns):
        raise ValueError(f"File '{path}' missing required columns; found {', '.join(df.columns)}")

    # Set index to DATETIME and sort
    df = df.set_index("DATETIME").sort_index()

    return df


def compute_sma(df: pd.DataFrame, short: int, long: int) -> None:
    """
    Append short- and long-horizon Simple Moving Averages (SMAs) to *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'CLOSE' column.  The DataFrame is modified in place.
    short : int
        Window length for the short SMA (e.g. 20).
    long : int
        Window length for the long SMA (e.g. 50).

    Notes
    -----
    * New columns are named 'SMA_<short>' and 'SMA_<long>'.
    * Values are NaN until the rolling window is fully populated
      (`min_periods` equals the window length).
    """
    df[f"SMA_{short}"] = df["CLOSE"].rolling(window=short, min_periods=short).mean()
    df[f"SMA_{long}"]  = df["CLOSE"].rolling(window=long,  min_periods=long ).mean()


def generate_signals(df: pd.DataFrame, short: int, long: int) -> None:
    """
    Add a 'Signal' column indicating SMA crossovers.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'CLOSE' column.  The DataFrame is modified in place.
    short : int
        Window length for the short SMA (e.g. 20).
    long : int
        Window length for the long SMA (e.g. 50).

    Notes
    -----
    A 'Buy' is recorded when SMA_short crosses above SMA_long,
    A 'Sell' when it crosses below, and other rows get a textual 'No signal (previous was …)' marker.
    The DataFrame is modified in place.
    """

    # Detect where the short SMA is above the long SMA
    # above is a pandas.Series of True / False values that tells, row-by-row,
    # whether the short SMA is currently above the long SMA.
    above = df[f"SMA_{short}"] > df[f"SMA_{long}"]

    # Find every time that Boolean changes value. Replace NaN with 0.
    change = above.astype(int).diff().fillna(0)

    # Map integers {-1, 1} to human-readable labels
    mapping = {1: "Buy", -1: "Sell"}

    # Map to "Buy" ( +1 ) and "Sell" ( -1 ). Replace 0 as NaN,
    labels = change.map(mapping)     # Pandas Series, not list

    # Walk through the "Buy"/"Sell"/NaN series row-by-row,
    # turning stretches of NaN into human-readable "No signal (previous was …)" fillers
    # while remembering the last real action.
    prev = "None"
    final_labels: List[str] = []

    for lbl in labels.to_list():    # Iterate once per bar
        if pd.isna(lbl):            # No crossover on this bar
            lbl_str = f"No signal (previous was {prev})"
        else:                       # Got "Buy" or "Sell"
            lbl_str = lbl
            prev = lbl
        final_labels.append(lbl_str)

    # Add signal column to the DataFrame
    df["Signal"] = final_labels

# ──────────────────────────────────────────────────────────────────────────────
# File‑level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str, sma_short: int, sma_long: int, out_dir: str) -> Optional[Tuple[str, str]]:
    """Process a single file; return (ticker, latest_signal) or None to skip."""
    if os.path.getsize(path) == 0:
        return None  # silently skip empty

    df = load_data_from_file(path)

    compute_sma(df, sma_short, sma_long)

    generate_signals(df, sma_short, sma_long)

    print(list(df.columns))

    # write outputs
    base = os.path.splitext(os.path.basename(path))[0]
    sig_path = os.path.join(out_dir, f"{base}-{sma_short}-{sma_long}-signals.txt")

    df[df["Signal"].isin(["Buy", "Sell"])] [["DATETIME", "CLOSE", "Signal"]] \
        .rename(columns={"CLOSE": "Price"}).to_csv(sig_path, index=False)

    return df["TICKER"].iloc[-1], df["Signal"].iloc[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare data containers
    buys:   List[str] = []
    sells:  List[str] = []
    nos:    List[str] = []

    # Create a “worker” function with most of its parameters already  bound (curried) so that each call only
    # needs the filename.
    #   process_file(...)   – user-defined function that processes one CSV or JSON
    #   sma_short, sma_long – window sizes for simple moving averages
    #   out_dir             – where results should be written
    worker = partial(
        process_file,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        out_dir=args.output
    )

    # Run sequentially or in parallel
    if args.jobs == 1:
        results_iter = map(worker, args.files)
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)
        with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
            results_iter = ex.map(worker, args.files)

    for res in results_iter:
        if not res:
            continue
        ticker, sig = res
        if sig == "Buy":
            buys.append(ticker)
        elif sig == "Sell":
            sells.append(ticker)
        else:
            nos.append(ticker)

    if buys:
        print("Buy: " + " ".join(buys))
    if sells:
        print("Sell: " + " ".join(sells))
    if args.show_no and nos:
        print("No signal: " + " ".join(nos))


if __name__ == "__main__":
    main()
