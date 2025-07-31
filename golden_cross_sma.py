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
1. `<base>-<short>-<long>.txt`             – full dataset with SMAs & signals
2. `<base>-<short>-<long>-signals.txt`     – Date, Price, Signal (Buy/Sell)

`<base>` is the input filename without extension (e.g. `SLV`).
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

from py_utils.progress_bar import ProgressBar

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Golden‑Cross SMA Buy/Sell signals in batch mode with optional parallelism."
    )
    p.add_argument("sma_short", type=int, help="Short SMA period (integer)")
    p.add_argument("sma_long", type=int, help="Long SMA period (integer)")
    p.add_argument("files", nargs="+", help="One or more input data files")
    p.add_argument(
        "-o", "--output", default=".", help="Root output directory (default: current)"
    )
    p.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    p.add_argument(
        "--show-no-signal",
        dest="show_no",
        action="store_true",
        help="Also print tickers whose latest signal is 'No signal …'",
    )

    args = p.parse_args()
    if args.sma_short >= args.sma_long:
        p.error("sma_short must be smaller than sma_long")
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_cols(df: pd.DataFrame) -> None:
    df.columns = [str(c).strip().strip("<>").upper() for c in df.columns]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    _sanitize_cols(df)
    required = {"TICKER", "DATE", "CLOSE"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"File '{path}' missing required columns; found {', '.join(df.columns)}"
        )
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d")
    return df


def compute_sma(df: pd.DataFrame, short: int, long: int) -> None:
    df[f"SMA_{short}"] = df["CLOSE"].rolling(window=short, min_periods=1).mean()
    df[f"SMA_{long}"] = df["CLOSE"].rolling(window=long, min_periods=1).mean()


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
    full_path = os.path.join(out_dir, f"{base}-{sma_short}-{sma_long}.txt")
    sig_path = os.path.join(out_dir, f"{base}-{sma_short}-{sma_long}-signals.txt")

    df[["TICKER", "DATE", f"SMA_{sma_short}", f"SMA_{sma_long}", "Signal"]].to_csv(full_path, index=False)

    df[df["Signal"].isin(["Buy", "Sell"])] [["DATE", "CLOSE", "Signal"]] \
        .rename(columns={"CLOSE": "Price"}).to_csv(sig_path, index=False)

    return df["TICKER"].iloc[-1], df["Signal"].iloc[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    buys: List[str] = []
    sells: List[str] = []
    nos: List[str] = []

    pb = ProgressBar(len(args.files))

    worker = partial(process_file, sma_short=args.sma_short, sma_long=args.sma_long, output_root=args.output)

    if args.jobs == 1:
        results_iter = map(worker, args.files)
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)
        with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
            results_iter = ex.map(worker, args.files)

    for res in results_iter:
        pb.update()
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
