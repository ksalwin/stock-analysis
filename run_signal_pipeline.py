#!/usr/bin/env python3
"""
run_signal_pipeline.py

Pipeline:
1. **golden_cross_sma.py** – batch‑processes all supplied ticker text files, producing
   per‑ticker "*-<low>-<high>-signals.txt" files inside a fixed `out/` directory
   (each ticker gets its own sub‑directory).
2. **signals_report.py** – is invoked **once per sub‑directory** inside `out/`,
   receiving only the signal files that match the specific SMA pair used for this
   run. This avoids command‑line length limits and ensures we don’t mix results
   from other SMA pairs.

Usage examples
--------------
```
python3 run_signal_pipeline.py --sma-low 10 --sma-high 130 gpw.txt pko.txt
```

Key assumptions
---------------
* Output directory is always `./out`.
* Ticker files are plain text (.txt) with market data.
* Both external scripts support the batch interfaces we agreed on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess
from typing import List, Dict

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
    parser = argparse.ArgumentParser(
        description="Batch golden‑cross analysis followed by per‑folder signal reporting.")

    # SMA settings
    parser.add_argument("--sma-low", type=int, default=20, help="Low SMA (default 20).")
    parser.add_argument("--sma-high", type=int, default=100, help="High SMA (default 100).")

    # External scripts
    default_dir = Path(__file__).with_name
    parser.add_argument("--golden-script", type=Path,
                        default=default_dir("golden_cross_sma.py"),
                        help="Path to batch‑enabled golden_cross_sma.py.")
    parser.add_argument("--signals-script", type=Path,
                        default=default_dir("signals_report.py"),
                        help="Path to batch‑enabled signals_report.py.")

    # Misc
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")

    # Positional – one or more ticker files
    parser.add_argument("tickers", nargs="+", type=Path,
                        help="Ticker text files (.txt) to process.")

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    if not args.golden_script.exists():
        print(f"Error: golden script not found: {args.golden_script}", file=sys.stderr)
        return 2
    if not args.signals_script.exists():
        print(f"Error: signals script not found: {args.signals_script}", file=sys.stderr)
        return 2
    if args.sma_low <= 0 or args.sma_high <= 0 or args.sma_low >= args.sma_high:
        print("Error: Invalid SMA values (positive, and low < high).", file=sys.stderr)
        return 2

    # Validate ticker files exist
    tickers: List[Path] = []
    for t in args.tickers:
        if not t.is_file():
            print(f"Warning: {t} is not a file – skipped.", file=sys.stderr)
        else:
            tickers.append(t.resolve())
    if not tickers:
        print("Error: No valid ticker files provided.", file=sys.stderr)
        return 2

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------
    out_dir = Path.cwd() / "out"
    if args.dry_run:
        print(f"[DRY‑RUN] Would create output directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Run golden_cross_sma.py once for all tickers
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Discover signal files for THIS SMA pair only
    # ---------------------------------------------------------------------
    pattern = f"*-{args.sma_low}-{args.sma_high}-signals.txt"
    signal_files = sorted(out_dir.rglob(pattern))
    if not signal_files:
        print(f"Warning: No '{pattern}' files found in output directory.")
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
