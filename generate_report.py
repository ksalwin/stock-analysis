#!/usr/bin/env python3
"""generate_golden_cross_report.py

Consolidate golden‑cross back‑test signal summaries into compact, sortable **TXT** files (comma‑separated
contents, but a .txt extension). The script scans folders (optionally recursively) or individual files
that match this pattern::

    <ticker>-<sma_low>-<sma_high>-signals-report.txt

For every ticker it finds, it writes either:

* ``<ticker>-summary.txt``  – one combined table (default), **or**
* ``<ticker>-<metric>.txt`` – one file per metric when ``--separate-files`` is set

in the *same directory* as the first source file for that ticker.

-------------------------------------------------------------------------------
Usage
-----
::

    python generate_golden_cross_report.py <path> [<path> ...] [options]

Positional ``<path>`` values may be directories or files.

Options
~~~~~~~
* ``--sort-by METRIC``    Column used to order rows; default ``expectancy_per_trade``.
* ``--separate-files``    Emit one TXT per metric instead of a single summary.
* ``--recursive``         Recurse into sub-directories when scanning a directory.

-------------------------------------------------------------------------------
Column name mapping (raw header → snake_case column)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Number of positive / breakeven trades → ``positive_trades``
* Number of negative trades              → ``negative_trades``
* Total positive / breakeven PnL (≥0)    → ``total_positive_pnl``
* Total negative PnL (<0)                → ``total_negative_pnl``
* Difference (positive – |negative|)     → ``difference``
* Win rate [%]                           → ``win_rate``
* Average win / loss                     → ``average_win_loss``
* Profit factor                          → ``profit_factor``
* Expectancy per trade                   → ``expectancy_per_trade``

-------------------------------------------------------------------------------
Examples
~~~~~~~~
::

    # Scan result folders, sort rows by profit_factor
    python generate_golden_cross_report.py out/slv  out/gld --sort-by profit_factor

    # Include sub‑dirs and write one TXT per metric
    python generate_golden_cross_report.py out/slv --recursive --separate-files
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

# ---------------- Configuration ------------------------------------------------

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

COL_ORDER = [
    "sma_low",
    "sma_high",
    *RAW_TO_COL.values(),
]

# ------------------------------------------------------------------------------

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

    @classmethod
    def from_file(cls, path: Path) -> "Record":
        m = FILE_RE.match(path.name)
        if not m:
            raise ValueError(f"Filename does not match expected pattern: {path}")
        ticker = m.group("ticker").upper()
        sma_low = int(m.group("sma_low"))
        sma_high = int(m.group("sma_high"))

        values: Dict[str, float | int] = {}
        with path.open() as f:
            for line in f:
                if ":" not in line:
                    continue
                raw_key, raw_val = [part.strip() for part in line.split(":", 1)]
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
        return cls(ticker=ticker, sma_low=sma_low, sma_high=sma_high, **values)

# ------------------------------------------------------------------------------


def _pretty_path(p: Path) -> str:
    """Return *p* relative to CWD if possible; otherwise an absolute string."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p.resolve())


def collect_files(paths: Sequence[str | Path], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            glob_pattern = "**/*" if recursive else "*"
            for child in p.glob(glob_pattern):
                if child.is_file() and FILE_RE.match(child.name):
                    files.append(child)
        else:
            print(f"Warning: {p} is not accessible", file=sys.stderr)
    if not files:
        print("No matching files found.", file=sys.stderr)
    return files


def group_by_ticker(files: Iterable[Path]):
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        m = FILE_RE.match(f.name)
        if m:
            ticker = m.group("ticker").upper()
            groups[ticker].append(f)
    return groups


def _sort_key_factory(sort_by: str):
    if sort_by not in COL_ORDER:
        return lambda r: (r.sma_low, r.sma_high)
    return lambda r: (
        -(getattr(r, sort_by) or float("inf")) if isinstance(getattr(r, sort_by), (int, float))
        else getattr(r, sort_by)
    )


def write_combined_txt(records: List[Record], sort_by: str, output_dir: Path):
    sorted_records = sorted(records, key=_sort_key_factory(sort_by))
    out_path = (output_dir / f"{records[0].ticker}-summary.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as txtfile:
        writer = csv.DictWriter(txtfile, fieldnames=COL_ORDER)
        writer.writeheader()
        for rec in sorted_records:
            writer.writerow({k: getattr(rec, k) for k in COL_ORDER})
    print(f"Written {_pretty_path(out_path)}")


def write_separate_txt(records: List[Record], sort_by: str, output_dir: Path):
    for col in RAW_TO_COL.values():
        sorted_records = sorted(records, key=_sort_key_factory(sort_by if col == sort_by else "sma_low"))
        out_path = (output_dir / f"{records[0].ticker}-{col}.txt").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as txtfile:
            writer = csv.DictWriter(txtfile, fieldnames=["sma_low", "sma_high", col])
            writer.writeheader()
            for rec in sorted_records:
                writer.writerow({
                    "sma_low": rec.sma_low,
                    "sma_high": rec.sma_high,
                    col: getattr(rec, col),
                })
        print(f"Written {_pretty_path(out_path)}")

# ------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Aggregate golden‑cross signal reports.")
    parser.add_argument("paths", nargs="+", help="Directories or files to scan.")
    parser.add_argument("--sort-by", default="expectancy_per_trade", help="Column to sort by.")
    parser.add_argument("--separate-files", action="store_true", help="Output one TXT per metric.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories.")
    args = parser.parse_args(argv)

    files = collect_files(args.paths, args.recursive)
    if not files:
        sys.exit(1)

    for ticker, flist in group_by_ticker(files).items():
        output_dir = flist[0].parent.resolve()
        records = [Record.from_file(f) for f in flist]
        if args.separate_files:
            write_separate_txt(records, args.sort_by, output_dir)
        else:
            write_combined_txt(records, args.sort_by, output_dir)


if __name__ == "__main__":
    main()
