#!/usr/bin/env python3
"""generate_summary.py

Batch‑combine golden‑cross signal **report files** into concise TXT summaries.

Each *input* must be a file named

    <ticker>-<sma_low>-<sma_high>-signals-report.txt

The script **always** writes a combined table

    <ticker>-summary.txt

and, when you pass ``--separate-files``, it **also** writes one file per metric:

    <ticker>-summary-<metric>.txt

All output files live beside the first source file for that ticker and use
lower‑case filenames.

-------------------------------------------------------------------------------
Usage
-----
::

    python generate_summary.py [options] <file> [<file> ...]

Options appear first; then list one or many report files (batch mode).

Options
~~~~~~~
* ``--sort-by METRIC``    Metric used to sort the **combined** summary (default
  ``expectancy_per_trade``).
* ``--separate-files``    Additionally emit one TXT per metric, each sorted by
  that metric’s value.

-------------------------------------------------------------------------------
Metric mapping (raw header → snake_case column)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Number of positive / breakeven trades → ``positive_trades``
* Number of negative trades              → ``negative_trades``
* Total positive / breakeven PnL (≥0)    → ``total_positive_pnl``
* Total negative PnL (<0)                → ``total_negative_pnl``
* Difference (positive – |negative|)     → ``difference``
* Win rate [%]                           → ``win_rate``
* Average win / loss                     → ``average_win_loss``
* Profit factor                          → ``profit_factor``
* Expectancy per trade                   → ``expectancy_per_trade``
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# ---------------- Filename pattern -------------------------------------------

FILE_RE = re.compile(r"^(?P<ticker>[A-Za-z]+)-(?P<sma_low>\d+)-(?P<sma_high>\d+)-signals-report\.txt$")

RAW_TO_COL = {
    "Number of positive / breakeven trades": "positive_trades",
    "Number of negative trades": "negative_trades",
    "Total positive / breakeven PnL (≥0)": "total_positive_pnl",
    "Total negative PnL (<0)": "total_negative_pnl",
    "Difference (positive – |negative|)": "difference",
    "Win rate [%]": "win_rate",
    "Average win / loss": "average_win_loss",
    "Profit factor": "profit_factor",
    "Expectancy per trade": "expectancy_per_trade",
}

COL_ORDER = ["sma_low", "sma_high", *RAW_TO_COL.values()]

# ---------------- Data model --------------------------------------------------

@dataclass
class Record:
    ticker: str
    sma_low: int
    sma_high: int
    positive_trades: int | None = None
    negative_trades: int | None = None
    total_positive_pnl: float | None = None
    total_negative_pnl: float | None = None
    difference: float | None = None
    win_rate: float | None = None
    average_win_loss: float | None = None
    profit_factor: float | None = None
    expectancy_per_trade: float | None = None

    @staticmethod
    def parse_file(path: Path) -> "Record":
        m = FILE_RE.match(path.name)
        if not m:
            raise ValueError(f"Filename does not match expected pattern: {path}")
        ticker = m.group("ticker").lower()
        sma_low = int(m.group("sma_low"))
        sma_high = int(m.group("sma_high"))

        vals: Dict[str, float | int] = {}
        with path.open() as fh:
            for line in fh:
                if ":" not in line:
                    continue
                raw_key, raw_val = [p.strip() for p in line.split(":", 1)]
                if raw_key not in RAW_TO_COL:
                    continue
                key = RAW_TO_COL[raw_key]
                if key.endswith("_trades"):
                    vals[key] = int(raw_val)
                else:
                    raw_val = raw_val.replace(",", ".")
                    try:
                        vals[key] = float(raw_val)
                    except ValueError:
                        vals[key] = None
        return Record(ticker, sma_low, sma_high, **vals)

# ---------------- Helpers -----------------------------------------------------

def _pretty(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def group_by_ticker(files: Iterable[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = FILE_RE.match(f.name)
        if m:
            groups[m.group("ticker").lower()].append(f)
        else:
            print(f"Warning: {f} skipped (filename pattern mismatch)", file=sys.stderr)
    return groups


def _numeric_desc(col: str):
    return lambda r: -(getattr(r, col) or float("-inf"))


def _combined_sort(sort_by: str):
    if sort_by not in COL_ORDER or sort_by in {"sma_low", "sma_high"}:
        return lambda r: (r.sma_low, r.sma_high)
    return _numeric_desc(sort_by)

# ---------------- Writers -----------------------------------------------------

def write_combined(records: list[Record], sort_by: str, out_dir: Path):
    rows = sorted(records, key=_combined_sort(sort_by))
    out_f = (out_dir / f"{records[0].ticker}-summary.txt").resolve()
    out_f.parent.mkdir(parents=True, exist_ok=True)
    with out_f.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=COL_ORDER)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: getattr(r, c) for c in COL_ORDER})
    print(f"Written {_pretty(out_f)}")


def write_metric_files(records: list[Record], out_dir: Path):
    for col in RAW_TO_COL.values():
        rows = sorted(records, key=_numeric_desc(col))
        out_f = (out_dir / f"{records[0].ticker}-summary-{col}.txt").resolve()
        with out_f.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["sma_low", "sma_high", col])
            writer.writeheader()
            for r in rows:
                writer.writerow({"sma_low": r.sma_low, "sma_high": r.sma_high, col: getattr(r, col)})
        print(f"Written {_pretty(out_f)}")

# ---------------- CLI ---------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    pa = argparse.ArgumentParser(description="Combine golden‑cross report files into summaries.")
    # Optional flags first so usage shows [options] before files
    pa.add_argument("--sort-by", default="expectancy_per_trade", help="Metric to sort the combined summary by")
    pa.add_argument("--separate-files", action="store_true", help="Also emit <ticker>-summary-<metric>.txt files")
    # Positional files argument last
    pa.add_argument("files", nargs="+", help="Input report files (batch mode)")
    args = pa.parse_args(argv)

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.is_file():
            pa.error(f"File not found: {p}")

    for ticker, flist in group_by_ticker(paths).items():
        out_dir = flist[0].parent.resolve()
        recs = [Record.parse_file(f) for f in flist]
        write_combined(recs, args.sort_by, out_dir)
        if args.separate_files:
            write_metric_files(recs, out_dir)


if __name__ == "__main__":
    main()
