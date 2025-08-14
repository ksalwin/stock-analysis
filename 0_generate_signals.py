#!/usr/bin/env python3
"""
Golden-Cross SMA Signal Generator — Batch & Parallel
===================================================
Generate Buy/Sell signals using a Golden-Cross strategy (SMA_short crossing
SMA_long) for one or more OHLC CSV text files. Files are processed
sequentially or in parallel.

Usage
-----
    python 0_generate_signals.py \
        --sma-short N --sma-long M \
        [--out-dir DIR] [--jobs N] [--show-no-signal] \
        FILE [FILE ...]

Arguments
---------
  --sma-short N          Window for the short SMA (int, required)
  --sma-long M           Window for the long SMA (int, required)
  --out-dir DIR          Output directory (default: ./out/)
  --jobs N               Process up to N files in parallel (default: 1)
  --show-no-signal       Also print tickers whose latest signal is "No signal …"

Input
-----
CSV with a header row. Required columns are '<TICKER>', '<DATE>', '<TIME>',
and '<CLOSE>' (angle brackets are literally present in the file). DATE is
YYYYMMDD and TIME is HHMMSS. The script removes the angle brackets from column
names, builds a pandas datetime index from DATE+TIME, and sorts by it.

Output
------
For each input file, writes:

    <out_dir>/<base>-<short>-<long>-signals.txt

where <base> is the input file name without extension.

Each output file contains only the crossover rows and two columns:

    Price, Signal

with "Price" taken from CLOSE and "Signal" being "Buy" or "Sell".

Console summary
---------------
After processing, prints one-line summaries like:

    Buy: TKR1 TKR2
    Sell: TKR3

and, if --show-no-signal is given:

    No signal: TKR4 TKR5

Notes
-----
- Empty input files are skipped silently.
- Validates that --sma-short ≤ --sma-long and --jobs ≥ 1.
- Requires NumPy and pandas.
- Uses a process pool when --jobs > 1 to parallelize per-file work.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from functools import partial
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="Generate Golden‑Cross SMA Buy/Sell signals."
    )

    # Common arguments
    parser.add_argument(
            "--jobs", type=int, default=1,
            help="Number of parallel workers (default: 1)")
    parser.add_argument(
            "--out-dir", default="out/",
            help="Root output directory (default: current)")
    parser.add_argument(
            "--show-no-signal",
            dest="show_no",
            action="store_true",
            help="Also print tickers whose latest signal is 'No signal …'")
    parser.add_argument(
            "files", nargs="+",
            help="One or more input data files")

    # Single SMA pair
    parser.add_argument(
            "--sma-short", type=int,
            help="Short SMA period (integer)")
    parser.add_argument(
            "--sma-long", type=int,
            help="Long SMA period (integer)")

    # SMA rage (min, max, step) for both low and high
    parser.add_argument(
            "--sma-short-range", nargs=3, type=int, metavar=("SMIN", "SMAX", "SSTEP"),
            help="Range for SMA short as three ints: min max step")
    parser.add_argument(
            "--sma-long-range", nargs=3, type=int, metavar=("LMIN", "LMAX", "LSTEP"),
            help="Range for SMA long as three ints: min max step")

    args = parser.parse_args()

    # --- Argument validation
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    # Decide mode
    any_single_args_given = (args.sma_short is not None) or \
                            (args.sma_long  is not None)
    any_range_args_given  = (args.sma_short_range is not None) or \
                            (args.sma_long_range  is not None)

    # Both single and range arguments provided - error
    if any_single_args_given and any_range_args_given:
        parser.error("Choose ONE mode: single (--sma-short & --sma-long) "
                     "OR range (--sma-short-range AND --sma-long-range).")

    # Validate single arguments
    if any_single_args_given:
        if args.sma_short is None or args.sma_long is None:
            parser.error("In single mode you must provide BOTH --sma-short and --sma-long.")
        if args.sma_short < 1 or args.sma_long < 1:
            parser.error("--sma-short/--sma-long must be positive integers.")
        if args.sma_short > args.sma_long:
            parser.error("--sma-short must be <= --sma-long.")
        args.mode = "single"

    # Validate range arguments
    elif any_range_args_given:
        if args.sma_short_range is None or args.sma_long_range is None:
            parser.error("In range mode you must provide BOTH --sma-short-range and --sma-long-range.")

        # Unpack input arguments tuple
        (smin, smax, sstep) = args.sma_short_range
        (lmin, lmax, lstep) = args.sma_long_range

        # Validate arguments
        for name, min_val, max_val, step in (("sma-short-range", smin, smax, sstep),
                                             ("sma-long-range",  lmin, lmax, lstep)):
            if min_val < 1 or max_val < 1 or step < 1:
                parser.error(f"--{name}: all values must be positive integers.")
            if min_val > max_val:
                parser.error(f"--{name}: min must be <= max.")

        args.mode = "range"
    else:
        parser.error("Choose a mode: either single (--sma-short & --sma-long) "
                     "or range (--sma-short-range & --sma-long-range).")

    # Normalize output directory
    args.out_dir = os.path.join(args.out_dir, "")

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


def compute_sma(df: pd.DataFrame, sma_range: list[int]) -> None:
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
    start, stop, step = sma_range

    # Calculate SMA window lengths
    sma_windows = range(start, stop + 1, step)

    # Extract close price column
    price = df["CLOSE"]

    # Calculate SMA and is separate off-frame
    result_columns = {
            f"SMA_{window}": price.rolling(window=window, min_periods=window).mean()
            for window in sma_windows
    }

    # Create a new DataFrame holding all computed SMA columns,
    # aligned to the original DataFrame's index for correct joining
    sma_df = pd.DataFrame(result_columns, index=df.index)

    # Join to main data frame
    return df.join(sma_df)


def generate_signals(df: pd.DataFrame,
                     short_range: list[int],
                     long_range:  list[int]
) -> dict[tuple[int, int], str]:
    """
    For every (short, long) pair in the given ranges (with short < long),
    detect SMA crossovers and add a 'Signal_<short>_<long>' column to *df*.

    Each signal column contains:
      - 'Buy' at bars where SMA_short crosses above SMA_long,
      - 'Sell' at bars where SMA_short crosses below SMA_long,
      - 'No signal (previous was …)' everywhere else.

    Returns
    -------
    latest_signal : dict[(short, long) -> str]
        The most recent label for each (short, long) pair, useful for summaries.
    """

    smin, smax, sstep = short_range
    lmin, lmax, lstep = long_range

    latest_signal: dict[tuple[int, int], str] = {}

    for short in range(smin, smax + 1, sstep):
        for long in range(lmin, lmax + 1, lstep):

            # Skip invalid pairs
            if short >= long:
                sys.exit()
                continue
            
            # Names of columns
            s_col = f"SMA_{short}"
            l_col = f"SMA_{long}"

            # Skip pairs if columns not present
            if s_col not in df.columns or l_col not in df.columns:
                sys.exit()
                continue

            # Detect where the short SMA is above the long SMA
            # above is a pandas.Series of True / False values that tells, row-by-row,
            # whether the short SMA is currently above the long SMA.
            short_above_long = df[s_col] > df[l_col]

            # Find every time that Boolean changes value. Replace NaN with 0.
            # +1 when 0->1 (cross up), -1 when 1->0 (cross down), 0 otherwise"
            cross_direction = short_above_long.astype(int).diff().fillna(0)

            # Definition of map: cross direction integers {-1, 1} to human-readable labels 
            cross_direction_to_signal_map = {1: "Buy", -1: "Sell"}

            # Map direction to "Buy" ( +1 ) and "Sell" ( -1 ) according to defined map.
            # Replace 0 as NaN,
            cross_direction_labels = cross_direction.map(cross_direction_to_signal_map)

            print(cross_direction_labels)
            sys.exit(1)

            # Walk through the Buy/Sell/NaN series row-by-row,
            # turning stretches of NaN into human-readable "No signal (previous was …)" fillers
            # while remembering the last real action.
            prev = "None"
            trade_signals: List[str] = []

            for label in cross_direction_labels.tolist():
                if pd.isna(label):
                    trade_signals.append(f"No signal (previous was {prev})")
                else:
                    trade_signals.append(label)
                    prev = label

            sig_col = f"Sig_{short}_{long}"
            df[sig_col] = trade_signals

            latest_signal[(short, long)] = trade_signals[-1] if trade_signals else "No signal (prev was None)"

    return latest_signal

# ──────────────────────────────────────────────────────────────────────────────
# File‑level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str,
                 sma_short_range: list[int],
                 sma_long_range: list[int],
                 out_dir: str
) -> Optional[Tuple[str, str]]:
    """Process a single file; return (ticker, latest_signal) or None to skip."""
    if os.path.getsize(path) == 0:
        return None  # silently skip empty

    df = load_data_from_file(path)

    # Compute SMA short and long
    df = compute_sma(df, sma_short_range)
    df = compute_sma(df, sma_long_range)

    generate_signals(df, sma_short_range, sma_long_range)

    system.exit()

    # Write outputs
    base = os.path.splitext(os.path.basename(path))[0]
    sig_path = os.path.join(out_dir, f"{base}-{sma_short}-{sma_long}-signals.txt")

    # Filter for Buy/Sell signals only
    filtered_df = df[ df["Signal"].isin(["Buy", "Sell"]) ]

    # Keep only columns needed
    filtered_df = filtered_df[["CLOSE", "Signal"]]

    # Rename "CLOSE" to "Price"
    filtered_df = filtered_df.rename(columns={"CLOSE": "Price"})

    # Save to csv
    filtered_df.to_csv(sig_path, index=False)

    # Return last [ticker, signal]
    return df["TICKER"].iloc[-1], df["Signal"].iloc[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare data containers
    buys:   List[str] = []
    sells:  List[str] = []
    nos:    List[str] = []

    # Convert single arguments to range to unify processing
    if args.mode == "single":
        args.sma_short_range = [args.sma_short, args.sma_short, 1]
        args.sma_long_range  = [args.sma_long,  args.sma_long,  1]

    # Create a “worker” function with most of its parameters already  bound (curried) so that each call only
    # needs the filename.
    #   process_file(...)   – user-defined function that processes one CSV or JSON
    #   sma_short_range, sma_long_range – ranges for simple moving averages
    #   out_dir             – where results should be written
    worker = partial(
        process_file,
        sma_short_range=args.sma_short_range,
        sma_long_range=args.sma_long_range,
        out_dir=args.out_dir
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
