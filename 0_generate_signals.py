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

def load_data(path: str) -> pd.DataFrame:
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

    # Drop the original DATE and TIME columns
    df = df.drop(columns=["DATE", "TIME"])

    # Columns check (must be before set_index)
    required = {"TICKER", "DATETIME", "CLOSE"}
    if not required.issubset(df.columns):
        raise ValueError(f"File '{path}' missing required columns; found {', '.join(df.columns)}")

    # Set index to DATETIME and sort
    df = df.set_index("DATETIME").sort_index()

    return df


def compute_sma(df: pd.DataFrame, short: int, long: int) -> None:
    df[f"SMA_{short}"] = df["CLOSE"].rolling(window=short, min_periods=short).mean()
    df[f"SMA_{long}"]  = df["CLOSE"].rolling(window=long,  min_periods=long).mean()


def generate_signals(df: pd.DataFrame, short: int, long: int) -> None:
    above = df[f"SMA_{short}"] > df[f"SMA_{long}"]
    change = above.astype(int).diff().fillna(0)

    mapping = {1: "Buy", -1: "Sell"}

    # Keep unmapped rows as <NA> (not None!) so we can test with pd.isna()
    labels = change.map(mapping)     # Pandas Series, not list

    prev = "Sell"
    final_labels: List[str] = []
    for lbl in labels.to_list():
        if pd.isna(lbl):    # Correctly detects the “no-signal” rows
            lbl_str = f"No signal (previous was {prev})"
        else:
            lbl_str = lbl
            prev = lbl
        final_labels.append(lbl_str)

    df["Signal"] = final_labels


# ──────────────────────────────────────────────────────────────────────────────
# File‑level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str, sma_short: int, sma_long: int, output_root: str) -> Optional[Tuple[str, str]]:
    """Process a single file; return (ticker, latest_signal) or None to skip."""
    if os.path.getsize(path) == 0:
        return None  # silently skip empty

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(output_root, base)

    try:
        df = load_data(path)
    except Exception:
        return None  # skip malformed

    compute_sma(df, sma_short, sma_long)
    generate_signals(df, sma_short, sma_long)

    # write outputs
    os.makedirs(out_dir, exist_ok=True)
    sig_path = os.path.join(out_dir, f"{base}-{sma_short}-{sma_long}-signals.txt")

    df[df["Signal"].isin(["Buy", "Sell"])] [["DATE", "CLOSE", "Signal"]] \
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
    #   process_file(...)    – user-defined function that processes one CSV or JSON
    #   sma_short, sma_long  – window sizes for simple moving averages
    #   output_root          – where results should be written
    worker = partial(
        process_file,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        output_root=args.output
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
