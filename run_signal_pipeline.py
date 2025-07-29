#!/usr/bin/env python3
"""
run_signal_pipeline.py

Pipeline:
1. **golden_cross_sma.py** – batch‑processes all supplied ticker text files, producing
   per‑ticker "*-signals.txt" files in a fixed `out/` directory.
2. **signals_report.py** – batch‑reads every generated `*-signals.txt` file, creating
   reports (details depend on that script).

Usage examples
--------------
```
python3 run_signal_pipeline.py gpw.txt pko.txt pkn.txt
python3 run_signal_pipeline.py out-wse_stocks/slv.txt
```
You can pass dozens or hundreds of ticker files in one call.

Design changes
--------------
* **Output directory** is always `out` (created in the current working directory).
* Positional argument is now one or more **ticker text files**, not a directory.
* `--jobs` flag removed (both external scripts are single batch calls).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess
from typing import List

###############################################################################
# Helpers
###############################################################################

def _run_cmd(cmd: List[str], label: str, dry: bool) -> int:
    """Run *cmd* unless *dry*; return exit status."""
    if dry:
        print(" ".join(cmd))
        return 0
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {label}: exited with {e.returncode}", file=sys.stderr)
        return e.returncode if e.returncode else 1
    except OSError as e:
        print(f"[ERROR ] {label}: {e}", file=sys.stderr)
        return 1

###############################################################################
# Main
###############################################################################

def main() -> int:
    p = argparse.ArgumentParser(
        description="Batch golden‑cross analysis followed by batched signal reporting.")

    # Positional – one or more ticker files
    p.add_argument("tickers", nargs="+", type=Path,
                   help="Ticker text files (.txt) to process.")

    # SMA settings
    p.add_argument("--sma-low", type=int, default=20, help="Low SMA (default 20).")
    p.add_argument("--sma-high", type=int, default=100, help="High SMA (default 100).")

    # External scripts
    default_dir = Path(__file__).with_name
    p.add_argument("--golden-script", type=Path,
                   default=default_dir("golden_cross_sma.py"),
                   help="Path to batch-enabled golden_cross_sma.py.")
    p.add_argument("--signals-script", type=Path,
                   default=default_dir("signals_report.py"),
                   help="Path to batch-enabled signals_report.py.")

    # Misc
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")

    args = p.parse_args()

    # ---------------- Validation ------------------------------------------
    if not args.golden_script.exists():
        print(f"Error: golden script not found: {args.golden_script}", file=sys.stderr)
        return 2
    if not args.signals_script.exists():
        print(f"Error: signals script not found: {args.signals_script}", file=sys.stderr)
        return 2
    if args.sma_low <= 0 or args.sma_high <= 0 or args.sma_low >= args.sma_high:
        print("Error: Invalid SMA values (positive, and low < high).", file=sys.stderr)
        return 2

    # Validate ticker files
    tickers: List[Path] = []
    for t in args.tickers:
        if not t.is_file():
            print(f"Warning: {t} is not a file – skipped.", file=sys.stderr)
        else:
            tickers.append(t.resolve())
    if not tickers:
        print("Error: No valid ticker files provided.", file=sys.stderr)
        return 2

    # ---------------- Fixed output directory ------------------------------
    out_dir = Path.cwd() / "out"
    if args.dry_run:
        print(f"[DRY‑RUN] Would create output directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- golden_cross batch ----------------------------------
    golden_cmd: List[str] = [
        sys.executable,
        str(args.golden_script),
        str(args.sma_low),
        str(args.sma_high),
        *[str(t) for t in tickers],
        "-o",
        str(out_dir),
    ]
    gc_exit = _run_cmd(golden_cmd, "golden_cross_batch", args.dry_run)

    # ---------------- Discover "*-signals.txt" ---------------------------
    signal_files = sorted(out_dir.rglob("*-signals.txt"))
    if not signal_files:
        print("Warning: No '*-signals.txt' files found in output directory.")
        return gc_exit

    # ---------------- signals_report batch --------------------------------
    signals_cmd: List[str] = [
        sys.executable,
        str(args.signals_script),
        *[str(sf) for sf in signal_files],
    ]
    rep_exit = _run_cmd(signals_cmd, "signals_report_batch", args.dry_run)

    return max(gc_exit, rep_exit)

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    raise SystemExit(main())
