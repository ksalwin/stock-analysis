#!/usr/bin/env python3
"""generate_summary.py

Batch‑combine golden‑cross signal **report files** into concise, sortable TXT summaries.

Each *input* is an individual back‑test report whose filename matches the pattern::

    <ticker>-<sma_low>-<sma_high>-signals-report.txt

Example: ``slv-40-160-signals-report.txt``

The script groups all files by ``ticker`` and writes either

* ``<ticker>-summary.txt``   – one combined table (default), or
* ``<ticker>-<metric>.txt``  – one file per metric when ``--separate-files`` is set

into the same directory as the **first** source file for that ticker.  Filenames are **always
lowercase** (e.g. ``slv-summary.txt``).

-------------------------------------------------------------------------------
Usage
-----
::

    python generate_summary.py <file> [<file> ...] [options]

You may supply dozens of files at once – that *is* batch mode.

Options
~~~~~~~
* ``--sort-by METRIC``    Metric column to sort rows by (default ``expectancy_per_trade``).
* ``--separate-files``    Emit one TXT per metric instead of a single summary.

-------------------------------------------------------------------------------
Metric mapping (raw header → snake_case column)
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

        values: Dict[str, float | int] = {}
        with path.open() as fh:
            for line in fh:
                if ":" not in line:
                    continue
                raw_key, raw_val = [p.strip() for p in line.split(":", 1)]
                if raw_key not in RAW_TO_COL:
                    continue
                key = RAW_TO_COL[raw_key]
                if key.endswith("_trades"):
                    values[key] = int(raw_val)
                else:
                    raw_val = raw_val.replace(",", ".")
                    try:
                        values[key] = float(raw_val)
                    except ValueError:
                        values[key] = None
        return Record(ticker, sma_low, sma_high, **values)

# ---------------- Helpers -----------------------------------------------------

def _pretty(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def group_by_ticker(files: Iterable[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = FILE_RE.match(f.name)
        if not m:
            print(f"Warning: {f.name} skipped – name doesn’t match pattern", file=sys.stderr)
            continue
        ticker = m.group("ticker").lower()
        groups[ticker].append(f)
    return groups


def _sort_key(sort_by: str):
    if sort_by not in COL_ORDER:
        return lambda r: (r.sma_low, r.sma_high)
    return lambda r: (
        -(getattr(r, sort_by) or float("inf")) if isinstance(getattr(r, sort_by), (int, float)) else getattr(r, sort_by)
    )

# ---------------- Writers -----------------------------------------------------

def write_summary(records: list[Record], sort_by: str, out_dir: Path):
    rows = sorted(records, key=_sort_key(sort_by))
    out_file = (out_dir / f"{records[0].ticker}-summary.txt").resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=COL_ORDER)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: getattr(r, c) for c in COL_ORDER})
    print(f"Written {_pretty(out_file)}")


def write_per_metric(records: list[Record], sort_by: str, out_dir: Path):
    for col in RAW_TO_COL.values():
        rows = sorted(records, key=_sort_key(sort_by if col == sort_by else "sma_low"))
        out_file = (out_dir / f"{records[0].ticker}-{col}.txt").resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["sma_low", "sma_high", col])
            writer.writeheader()
            for r in rows:
                writer.writerow({"sma_low": r.sma_low, "sma_high": r.sma_high, col: getattr(r, col)})
        print(f"Written {_pretty(out_file)}")

# ---------------- Main --------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Combine golden‑cross report files into summaries.")
    parser.add_argument("files", nargs="+", help="Input report files to process.")
    parser.add_argument("--sort-by", default="expectancy_per_trade", help="Metric column to sort by.")
    parser.add_argument("--separate-files", action="store_true", help="Produce one TXT per metric instead of a combined file.")
    args = parser.parse_args(argv)

    file_paths = [Path(f) for f in args.files]
    # Validate files exist
    for p in file_paths:
        if not p.is_file():
            parser.error(f"File not found: {p}")

    for ticker, paths in group_by_ticker(file_paths).items():
        out_dir = paths[0].parent.resolve()
        records = [Record.parse_file(p) for p in paths]
        if args.separate_files:
            write_per_metric(records, args.sort_by, out_dir)
        else:
            write_summary(records, args.sort_by, out_dir)


if __name__ == "__main__":
    main()
