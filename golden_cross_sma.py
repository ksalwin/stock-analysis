#!/usr/bin/env python3
"""
Golden‑Cross SMA Signal Generator (Batch‑Capable, Text‑Only)
===========================================================
Generate Buy/Sell signals using the Golden‑Cross strategy (short‑term SMA
crossing long‑term SMA) for **one or more** OHLC data files and write the
results as **plain‑text CSV files with a `.txt` extension**. No charts are
produced.

Usage
-----
    python golden_cross_sma.py <SMA_short> <SMA_long> <files...> [options]

Options
~~~~~~~
  -o, --output DIR       Root directory for results (default: .)
  --show-no-signal       Also print tickers whose latest signal is "No signal…"

Example
~~~~~~~
Process many files and store outputs in ./out (skipping "No signal" tickers):

    python golden_cross_sma.py 20 100 data/*.txt -o ./out

Outputs (per file, saved in `<OUTPUT_DIR>/<ticker>/`)
----------------------------------------------------
1. `<base>-<short>-<long>.txt`             – full dataset with SMAs & signals
2. `<base>-<short>-<long>-signals.txt`     – Date, Price, Signal (Buy/Sell)

`<base>` is the input filename without extension (e.g. `SLV`).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Golden‑Cross SMA Buy/Sell signals in batch mode."
    )
    p.add_argument("sma_short", type=int, help="Short SMA period (integer)")
    p.add_argument("sma_long", type=int, help="Long SMA period (integer)")
    p.add_argument("files", nargs="+", help="One or more input data files")
    p.add_argument(
        "-o", "--output", default=".", help="Root output directory (default: current)"
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


# ──────────────────────────────────────────────────────────────────────────────
# Technical analysis logic
# ──────────────────────────────────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, short: int, long: int) -> None:
    df[f"SMA_{short}"] = df["CLOSE"].rolling(window=short, min_periods=1).mean()
    df[f"SMA_{long}"] = df["CLOSE"].rolling(window=long, min_periods=1).mean()


def generate_signals(df: pd.DataFrame, short: int, long: int) -> None:
    above = np.where(df[f"SMA_{short}"] > df[f"SMA_{long}"], 1, 0)
    change = np.diff(above, prepend=above[0])
    prev = "Sell"
    signals: List[str] = []
    for c in change:
        if c == 1:
            cur = "Buy"
        elif c == -1:
            cur = "Sell"
        else:
            cur = f"No signal (previous was {prev})"
        signals.append(cur)
        if cur in {"Buy", "Sell"}:
            prev = cur
    df["Signal"] = signals


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, out_dir: str, base: str, s: int, l: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    full_path = os.path.join(out_dir, f"{base}-{s}-{l}.txt")
    sig_path = os.path.join(out_dir, f"{base}-{s}-{l}-signals.txt")

    df[["TICKER", "DATE", f"SMA_{s}", f"SMA_{l}", "Signal"]].to_csv(full_path, index=False)

    df[df["Signal"].isin(["Buy", "Sell"])] [["DATE", "CLOSE", "Signal"]] \
        .rename(columns={"CLOSE": "Price"}).to_csv(sig_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Per‑file processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str, args: argparse.Namespace) -> Optional[Tuple[str, str]]:
    # Skip empty files silently
    if os.path.getsize(path) == 0:
        return None

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(args.output, base)

    try:
        df = load_data(path)
    except Exception:
        # Skip unreadable or malformed files silently
        return None

    compute_sma(df, args.sma_short, args.sma_long)
    generate_signals(df, args.sma_short, args.sma_long)

    save_outputs(df, out_dir, base, args.sma_short, args.sma_long)

    ticker = df["TICKER"].iloc[-1]
    latest_sig = df["Signal"].iloc[-1]
    return ticker, latest_sig


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    buys: List[str] = []
    sells: List[str] = []
    nos: List[str] = []

    for f in args.files:
        res = process_file(f, args)
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
        print("No signal:")
        print("  " + " ".join(nos))


if __name__ == "__main__":
    main()
