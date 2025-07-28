#!/usr/bin/env python3
"""
Batch‑run golden_cross_sma.py for all tickers and then batch‑run
signals_report.py for all "*-signals.txt" files.

Key points
----------
* **golden_cross_sma.py** (batch‑enabled) is called **once** with *all* input
  ticker files plus `-o <out_dir>`.
* **signals_report.py** (batch‑enabled) is called **once** with *all* discovered
  `-signals.txt` files.
* Output directory is created as `out-<input_dir>` next to the source dir.
* `--jobs` is kept for CLI stability but ignored (both stages are single‑call).
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
        description="Run batch golden-cross analysis and generate batched signal reports.")

    p.add_argument("dir", type=Path, help="Directory containing ticker text files.")

    # SMA settings
    p.add_argument("--sma-low", type=int, default=20, help="Low SMA (default 20).")
    p.add_argument("--sma-high", type=int, default=100, help="High SMA (default 100).")

    # File filtering
    p.add_argument("--ext", default=".txt", help="Ticker file extension (default .txt).")

    # External scripts
    default_dir = Path(__file__).with_name
    p.add_argument("--golden-script", type=Path,
                   default=default_dir("golden_cross_sma.py"),
                   help="Path to batch‑enabled golden_cross_sma.py.")
    p.add_argument("--signals-script", type=Path,
                   default=default_dir("signals_report.py"),
                   help="Path to batch‑enabled signals_report.py.")

    # Misc options
    p.add_argument("--out-prefix", default="out-", help="Prefix for output dir (default out-)")
    p.add_argument("--jobs", type=int, default=1, help="Ignored, kept for backward compatibility.")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")

    args = p.parse_args()

    # ---------------- Validation ------------------------------------------
    if not args.dir.is_dir():
        print(f"Error: {args.dir} is not a directory", file=sys.stderr)
        return 2
    if not args.golden_script.exists():
        print(f"Error: golden script not found: {args.golden_script}", file=sys.stderr)
        return 2
    if not args.signals_script.exists():
        print(f"Error: signals script not found: {args.signals_script}", file=sys.stderr)
        return 2
    if args.sma_low <= 0 or args.sma_high <= 0 or args.sma_low >= args.sma_high:
        print("Error: Invalid SMA values (positive, and low < high).", file=sys.stderr)
        return 2

    # ---------------- Output directory ------------------------------------
    out_dir = args.dir.parent / f"{args.out_prefix}{args.dir.name}"
    if args.dry_run:
        print(f"[DRY‑RUN] Would create output directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Collect tickers -------------------------------------
    tickers = sorted(f for f in args.dir.iterdir() if f.is_file() and f.suffix == args.ext)
    if not tickers:
        msg = f"No files with extension '{args.ext}' found in {args.dir}"
        print(msg)
        return 0

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
