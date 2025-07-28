#!/usr/bin/env python3
"""
Golden‑Cross SMA Signal Generator (Batch‑Capable)
================================================
Generate Buy/Sell signals using the Golden‑Cross strategy (short‑term SMA
crossing long‑term SMA) for **one or more** OHLC data files.

Usage
-----
    python golden_cross_sma.py <SMA_short> <SMA_long> <files...> [--png] [-o OUTPUT_DIR]

Examples
~~~~~~~~
*Process two files, drop outputs under ./results/, and skip charts*
    python golden_cross_sma.py 20 100 slv.txt gld.txt -o ./results

*Same as above, but save PNG charts, too*
    python golden_cross_sma.py 20 100 slv.txt gld.txt -o ./results --png

Arguments
~~~~~~~~~
Positional
^^^^^^^^^^
SMA_short      Integer – **must be smaller** than *SMA_long*.
SMA_long       Integer – long moving‑average period.
files          One or more input files; each must contain at minimum the
               columns `<TICKER>, <DATE>, <CLOSE>` (header names can be wrapped
               in angle brackets and in any case/whitespace).

Optional
^^^^^^^^
--png, --chart, --plot   Save a price/SMA chart (`.png`) for each file.
-o, --output OUTPUT_DIR  Root directory into which a sub‑folder named after the
                         ticker (file stem) is created. Default is the current
                         working directory.

Outputs (per file, saved in `<OUTPUT_DIR>/<ticker>/`)
----------------------------------------------------
1. `<base>-<short>-<long>.txt`             – full dataset with SMAs & signals
2. `<base>-<short>-<long>-signals.txt`     – Date, Price, Signal (Buy/Sell)
3. `<base>-<short>-<long>.png` *optional*  – chart (only when `--png` supplied)

`<base>` is the input filename without extension (e.g. `slv`).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Parsing & validation
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Golden‑Cross SMA Buy/Sell signals (batch‑mode)"
    )
    parser.add_argument("sma_short", type=int, help="Short SMA period (integer)")
    parser.add_argument("sma_long", type=int, help="Long SMA period (integer)")
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more input data files (CSV/TXT)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Root output directory (default: current directory)",
    )
    parser.add_argument(
        "--png",
        "--chart",
        "--plot",
        dest="make_png",
        action="store_true",
        help="Also save PNG chart per file",
    )

    args = parser.parse_args()

    if args.sma_short >= args.sma_long:
        parser.error("sma_short must be smaller than sma_long")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_columns(df: pd.DataFrame) -> None:
    df.columns = [str(c).strip().strip("<>").upper() for c in df.columns]


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=0)
    except Exception as exc:
        raise ValueError(f"Could not parse file '{path}': {exc}") from exc

    _sanitize_columns(df)

    required = {"TICKER", "DATE", "CLOSE"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"File '{path}' missing required columns; found {', '.join(df.columns)}"
        )

    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Core TA logic
# ──────────────────────────────────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, short: int, long: int) -> None:
    df[f"SMA_{short}"] = df["CLOSE"].rolling(window=short, min_periods=1).mean()
    df[f"SMA_{long}"] = df["CLOSE"].rolling(window=long, min_periods=1).mean()


def generate_signals(df: pd.DataFrame, short: int, long: int) -> None:
    above = np.where(df[f"SMA_{short}"] > df[f"SMA_{long}"], 1, 0)
    change = np.diff(above, prepend=above[0])

    signals: List[str] = []
    prev = "Sell"
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

def save_outputs(df: pd.DataFrame, out_dir: str, base: str, short: int, long: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    full_path = os.path.join(out_dir, f"{base}-{short}-{long}.txt")
    sig_path = os.path.join(out_dir, f"{base}-{short}-{long}-signals.txt")

    df[["TICKER", "DATE", f"SMA_{short}", f"SMA_{long}", "Signal"]].to_csv(
        full_path, index=False
    )

    df[df["Signal"].isin(["Buy", "Sell"])] [["DATE", "CLOSE", "Signal"]] \
        .rename(columns={"CLOSE": "Price"}).to_csv(sig_path, index=False)


def plot_chart(
    df: pd.DataFrame,
    out_dir: str,
    base: str,
    short: int,
    long: int,
    make_png: bool,
) -> None:
    if not make_png:
        return

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(df["DATE"], df["CLOSE"], label="Close")
    plt.plot(df["DATE"], df[f"SMA_{short}"], label=f"SMA {short}")
    plt.plot(df["DATE"], df[f"SMA_{long}"], label=f"SMA {long}")

    buys = df[df["Signal"] == "Buy"]
    sells = df[df["Signal"] == "Sell"]
    plt.scatter(buys["DATE"], buys["CLOSE"], marker="^", s=120, label="Buy")
    plt.scatter(sells["DATE"], sells["CLOSE"], marker="v", s=120, label="Sell")

    plt.title(f"{df['TICKER'].iloc[0]}: Golden‑Cross {short}/{long}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{base}-{short}-{long}.png"))
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main per‑file processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str, args: argparse.Namespace) -> None:
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(args.output, base)

    try:
        df = load_data(path)
    except Exception as exc:
        print(f"[ERROR] {path}: {exc}", file=sys.stderr)
        return

    compute_sma(df, args.sma_short, args.sma_long)
    generate_signals(df, args.sma_short, args.sma_long)

    save_outputs(df, out_dir, base, args.sma_short, args.sma_long)
    plot_chart(df, out_dir, base, args.sma_short, args.sma_long, args.make_png)

    # Print latest signal prefixed by ticker
    print(f"{df['TICKER'].iloc[-1]} {df['Signal'].iloc[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Ensure output root exists
    os.makedirs(args.output, exist_ok=True)

    for f in args.files:
        process_file(f, args)


if __name__ == "__main__":
    main()
