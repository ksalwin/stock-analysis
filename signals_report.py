#!/usr/bin/env python3
"""
signals_report.py – analyse **one or many** Buy/Sell signal files and create text summary reports, now with optional **parallel jobs**.

(No plotting – the script only generates output files.)

Generated parameters
--------------------
* **Number of positive / breakeven trades** – count of Buy‑Sell pairs where PnL ≥ 0
* **Number of negative trades** – count of Buy‑Sell pairs where PnL < 0
* **Total positive / breakeven PnL** – sum of all non‑negative trade results
* **Total negative PnL** – sum of all negative trade results
* **Difference (positive − |negative|)** – net PnL across all trades
* **Win rate [%]** – (positive trades / total trades) × 100
* **Average win / loss** – mean positive PnL divided by mean |negative| PnL (ratio > 1 desirable)
* **Profit factor** – Σ positive PnL ÷ |Σ negative PnL|  
  > 2.0 — Very good | 1.5–2.0 — Strong | 1.1–1.5 — OK | ≈1 — Breakeven | <1 — Losing
* **Expectancy per trade** – net PnL ÷ total trades (average profit/loss each trade)

USAGE examples
--------------
# Sequential (1 job) — summary only
python signals_report.py out/*/*-signals.txt

# 8 concurrent jobs, with pairs & console print
python signals_report.py out/*/*-signals.txt --jobs 8 --pairs --print
"""

import argparse
import csv
import os
import py-utils import file_finder
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# Core analytics
# ────────────────────────────────────────────────────────────────────────────────

def read_signals(fname: Path) -> List[dict]:
    """Read a CSV‑like text file into a list of dictionaries."""
    with fname.open(newline="") as f:
        rdr = csv.DictReader(f)
        return [
            {
                "date": datetime.strptime(row["DATE"], "%Y-%m-%d"),
                "price": float(row["Price"]),
                "signal": row["Signal"].strip(),
            }
            for row in rdr
        ]


def analyse(data: List[dict], include_pairs: bool) -> List[str]:
    """Return report lines (pair list optional)."""
    lines: List[str] = []
    pos_tot = neg_tot = 0.0
    pos_cnt = neg_cnt = 0
    pos_trades: List[float] = []
    neg_trades: List[float] = []

    if include_pairs:
        lines.extend(["Buy‑Sell pairs and differences", "-" * 60])

    i = 0
    while i < len(data) - 1:
        if data[i]["signal"] == "Buy" and data[i + 1]["signal"] == "Sell":
            buy, sell = data[i], data[i + 1]
            pnl = sell["price"] - buy["price"]
            if include_pairs:
                lines.append(
                    f"{buy['date'].date()} @ {buy['price']:.4f} → "
                    f"{sell['date'].date()} @ {sell['price']:.4f} = {pnl:.4f}"
                )
            if pnl >= 0:
                pos_tot += pnl
                pos_cnt += 1
                pos_trades.append(pnl)
            else:
                neg_tot += pnl
                neg_cnt += 1
                neg_trades.append(pnl)
            i += 2
        else:
            i += 1

    total_trades = pos_cnt + neg_cnt
    diff = pos_tot - abs(neg_tot)

    win_rate = (pos_cnt / total_trades * 100) if total_trades else 0.0
    avg_win = sum(pos_trades) / pos_cnt if pos_cnt else 0.0
    avg_loss = abs(sum(neg_trades) / neg_cnt) if neg_cnt else 0.0
    avg_win_loss = (avg_win / avg_loss) if avg_loss else float("inf") if avg_win else 0.0
    profit_factor = (pos_tot / abs(neg_tot)) if neg_tot else float("inf") if pos_tot else 0.0
    expectancy = diff / total_trades if total_trades else 0.0

    if include_pairs:
        lines.append("-" * 60)

    lines.extend(
        [
            f"Number of positive / breakeven trades:     {pos_cnt}",
            f"Number of negative trades:                 {neg_cnt}",
            f"Total positive / breakeven PnL (≥0):       {pos_tot:.4f}",
            f"Total negative PnL (<0):                  {neg_tot:.4f}",
            f"Difference (positive – |negative|):        {diff:.4f}",
            f"Win rate [%]:                              {win_rate:.2f}",
            f"Average win / loss:                        {avg_win_loss:.4f}",
            f"Profit factor:                             {profit_factor:.4f}",
            f"Expectancy per trade:                      {expectancy:.4f}",
        ]
    )
    return lines

# ────────────────────────────────────────────────────────────────────────────────
# Worker function
# ────────────────────────────────────────────────────────────────────────────────

def process_file(path: Path, include_pairs: bool) -> Tuple[Path, List[str]]:
    data = read_signals(path)
    report_lines = analyse(data, include_pairs)
    out_file = path.with_name(path.stem + "-report" + path.suffix)
    out_file.write_text("\n".join(report_lines))
    return path, report_lines

# ────────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal reports in parallel (no plotting).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--file-path', default=".", help='Directory to search files in')
    parser.add_argument('--file-pattern', default=".", help='Glob pattern to match files')
    parser.add_argument("--jobs", type=int, default=1, help="number of parallel jobs/processes (default 1)")
    parser.add_argument("--print", dest="do_print", action="store_true", help="print statistics to console")
    parser.add_argument("--pairs", action="store_true", help="include Buy‑Sell pair list in report")
    parser.add_argument("input_files", nargs="+", help="paths to *-signals.txt files")

    args = parser.parse_args()

    """Find all files"""
    file_list = args.input_files or file_finder.find_files(args.file-path, args.file-pattern, true)
    return 0

    if args.jobs == 1 or len(paths) == 1:
        for p in paths:
            path, lines = process_file(p, args.pairs)
            if args.do_print:
                print(f"\n=== {path} ===\n" + "\n".join(lines))
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_file, p, args.pairs): p for p in paths}
            for fut in as_completed(futures):
                path, lines = fut.result()
                if args.do_print:
                    print(f"\n=== {path} ===\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
