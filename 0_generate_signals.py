#!/usr/bin/env python3
"""
Golden-Cross SMA Signal Generator
===================================================
Generate Buy/Sell signals using a Golden-Cross strategy (SMA_short crossing
SMA_long) for one or more OHLC CSV text files. Files are processed
sequentially or in parallel.

Usage
-----
    python 0_generate_signals.py \
        --sma-short N --sma-long M \
        [--out-dir DIR] [--jobs N] \
        FILE [FILE ...]

Arguments
---------
  --sma-short N          Window for the short SMA (int, required)
  --sma-long M           Window for the long SMA (int, required)
  --out-dir DIR          Output directory (default: ./out/)
  --jobs N               Process up to N files in parallel (default: 1)

Input
-----
CSV with a header row. Required columns are '<TICKER>', '<DATE>', '<TIME>',
and '<CLOSE>' (angle brackets are literally present in the file). DATE is
YYYYMMDD and TIME is HHMMSS. The script removes the angle brackets from column
names, builds a pandas datetime index from DATE+TIME, and sorts by it.

Output
------
For each input file, writes:

    <out_dir>/<base>-<short>-<long>-signals.txt

where <base> is the input file name without extension.

Each output file contains only the crossover rows and two columns: "Price" and "Signal".
"Price" is taken from CLOSE and "Signal" is "Buy" or "Sell".

Console summary
---------------
After processing, prints one-line summaries like:

    Buy: TKR1, TKR2
    Sell: TKR3

Notes
-----
- Empty input files are skipped silently.
- Validates that --sma-short ≤ --sma-long and --jobs ≥ 1.
- Requires NumPy and pandas.
- Uses a process pool when --jobs > 1 to parallelize per-file work.
"""

import argparse
import os
import pandas as pd
import sys
import tomllib

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="Generate Golden‑Cross SMA Buy/Sell signals."
    )
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """

    # Common arguments
    parser.add_argument(
            "--config", default="0_generate_signals.toml",
            help="Path to config file (default: 0_generate_signals.toml)")
    parser.add_argument(
            "--jobs", type=int, default=1,
            help="Number of parallel workers (default: 1)")
    parser.add_argument(
            "--out-dir", default="out/",
            help="Root output directory (default: current)")
    parser.add_argument(
            "files", nargs="*",
            help="One or more input data files")

    # Single SMA pair
    parser.add_argument(
            "--sma-short", type=int,
            help="Short SMA period (integer)")
    parser.add_argument(
            "--sma-long", type=int,
            help="Long SMA period (integer)")

    # SMA rage (min, max, step) for both low and high
    parser.add_argument(
            "--sma-short-range", nargs=3, type=int, metavar=("SMIN", "SMAX", "SSTEP"),
            help="Range for SMA short as three ints: min max step")
    parser.add_argument(
            "--sma-long-range", nargs=3, type=int, metavar=("LMIN", "LMAX", "LSTEP"),
            help="Range for SMA long as three ints: min max step")

    args = parser.parse_args()

    return args


def load_toml_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration from a TOML file and return it as a plain dict.

    Parameters
    ----------
    path : str
        Path to a *.toml file (e.g., '0_generate_signals.toml').

    Returns
    -------
    dict[str, Any]
        A nested dictionary mirroring the TOML structure.
        Example for this project's config:
            {
              "jobs": 16,
              "out_dir": "out/",
              "show_no": false,
              "sma": {
                "short_range": [5, 60, 1],
                "long_range":  [90, 180, 5]
              },
              "inputs": {
                "files": ["wse_stocks/slv.txt"]
              }
            }

    Notes
    -----
    - 'tomllib' requires reading in binary mode ('rb'), not text mode.
    """
    # Open in binary mode as required by 'tomllib'
    with open(path, "rb") as f:
        config: Dict[str, Any] = tomllib.load(f)

    # Return the raw config dict (no mutations here)
    return config

def merge_config_into_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """
    Merge selected keys from TOML config into argparse 'args' *in place*.

    Precedence rule
    ---------------
    - Command‑line arguments **always override** config values.
      (We only copy a config value if the corresponding CLI value is "unset".)

    Supported config keys (TOML)
    ----------------------------
    Top‑level:
      jobs: int
      out_dir: str
      show_no: bool

    Table [sma]:
      short: int                # single mode (optional)
      long: int                 # single mode (optional)
      short_range: [int,int,int]
      long_range:  [int,int,int]

    Table [inputs]:
      files: list[str]

    Notes
    -----
    - We don't validate here; we only copy values into 'args'.
      Validation happens later (after merge), so config‑provided values are
      checked with the same rules as CLI‑provided ones.
    """
    # Helper function to traverse nested dicts, returns default if any key is missing.
    def get_path(d: Dict[str, Any], path: list[str], default_return=None) -> Any:
        # Creates variable cur of type Any, which is a generic type that can hold any value
        # cur is like a pointer as we drill down into the dictionary
        # At the start, cur is assigned the value of d (the input dictionary) 
        # d is the config dictionary
        # Any is a generic type that can hold any value
        cur: Any = d
        # Iterates over each key in the path list
        # path is a list of strings, each string is a key in the dictionary
        # Each loop moves the pointer one step deeper into the dictionary
        for key in path:
            # Checks if cur is not a dictionary or if the current key k is not present in cur
            # If either of these conditions is true, the function returns the default value
            # If cur is a dictionary and k is present, cur is updated to the value associated with k
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # --- Simple scalars (copy only if CLI left the default/empty) ---
    # Check if the user did NOT explicitly set --jobs on the CLI.
    # (In argparse we gave jobs a default = 1. So if it's still 1, it means
    #  the user didn't touch it. If they passed --jobs 8, then args.jobs == 8.)
    if args.jobs == 1:
        val = cfg.get("jobs")
        if isinstance(val, int):
            args.jobs = val

    if args.out_dir == "out/":
        val = cfg.get("out_dir")
        if isinstance(val, str):
            args.out_dir = val

    # 'show_no' is False unless '--show-no-signal' was set.
    # Only copy from config when CLI did not set it.
    if not getattr(args, "show_no", False):
        val = cfg.get("show_no")
        if isinstance(val, bool):
            args.show_no = val

    # --- SMA settings (single or range) ---
    sma = cfg.get("sma", {})
    # Single mode (copy only if CLI didn’t pass them)
    if args.sma_short is None:
        val = sma.get("short")
        if isinstance(val, int):
            args.sma_short = val
    if args.sma_long is None:
        val = sma.get("long")
        if isinstance(val, int):
            args.sma_long = val

    # Range mode (copy only if CLI didn’t pass them)
    if args.sma_short_range is None:
        val = sma.get("short_range")
        if isinstance(val, list) and len(val) == 3 and all(isinstance(x, int) for x in val):
            args.sma_short_range = val
    if args.sma_long_range is None:
        val = sma.get("long_range")
        if isinstance(val, list) and len(val) == 3 and all(isinstance(x, int) for x in val):
            args.sma_long_range = val

    # --- Input files (copy only if CLI omitted them) ---
    if not args.files:
        files = get_path(cfg, ["inputs", "files"])
        if isinstance(files, list) and all(isinstance(x, str) for x in files):
            args.files = files

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate merged CLI+config arguments and finalize derived fields.

    What this function guarantees on return
    --------------------------------------
    - args.jobs            : int >= 1
    - args.mode            : "single" or "range"
    - args.files           : non-empty list[str]
    - args.out_dir         : normalized with trailing os.sep
    - args.sma_short_range : [int,int,int]
    - args.sma_long_range  : [int,int,int]

    Errors
    ------
    Raises SystemExit with a clear message if any rule is violated.
    """
    # 1) --jobs must be >= 1
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    # 2) Decide operating mode (single vs. range)
    any_single_args_given = (args.sma_short is not None) or (args.sma_long is not None)
    any_range_args_given  = (args.sma_short_range is not None) or (args.sma_long_range is not None)

    if any_single_args_given and any_range_args_given:
        raise SystemExit(
            "Choose ONE mode: single (--sma-short & --sma-long) "
            "OR range (--sma-short-range AND --sma-long-range)."
        )

    if any_single_args_given:
        # Single-mode validation
        if args.sma_short is None or args.sma_long is None:
            raise SystemExit("In single mode you must provide BOTH --sma-short and --sma-long.")
        if args.sma_short < 1 or args.sma_long < 1:
            raise SystemExit("--sma-short/--sma-long must be positive integers.")
        if args.sma_short > args.sma_long:
            raise SystemExit("--sma-short must be <= --sma-long.")
        args.mode = "single"

    elif any_range_args_given:
        # Range-mode validation
        if args.sma_short_range is None or args.sma_long_range is None:
            raise SystemExit("In range mode you must provide BOTH --sma-short-range and --sma-long-range.")
        (smin, smax, sstep) = args.sma_short_range
        (lmin, lmax, lstep) = args.sma_long_range
        for name, min_val, max_val, step in (
            ("sma-short-range", smin, smax, sstep),
            ("sma-long-range",  lmin, lmax, lstep),
        ):
            if min_val < 1 or max_val < 1 or step < 1:
                raise SystemExit(f"--{name}: all values must be positive integers.")
            if min_val > max_val:
                raise SystemExit(f"--{name}: min must be <= max.")
        args.mode = "range"
    else:
        raise SystemExit(
            "Choose a mode: either single (--sma-short & --sma-long) "
            "or range (--sma-short-range & --sma-long-range)."
        )

    # 3) Ensure input files exist (from CLI or config)
    if not args.files:
        raise SystemExit(
            "No input FILEs provided. Supply them positionally on the CLI "
            "or set [inputs].files in the TOML config."
        )

    # 4) Normalize output directory to always have a trailing separator
    args.out_dir = os.path.join(args.out_dir, "")

    # 5) Finalize: convert single→range for uniform downstream processing
    if args.mode == "single":
        args.sma_short_range = [args.sma_short, args.sma_short, 1]
        args.sma_long_range  = [args.sma_long,  args.sma_long,  1]

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ohlc_from_file(path: str) -> pd.DataFrame:
    """
    Load OHLC data from a file.

    Parameters
    ----------
    path : str
        Path to the input data file.

    Returns
    -------
    pandas.DataFrame
        Loaded OHLC data.

    Notes
    -----
    - Reads the file (returns data frame).
    - Verifies if input data has all needed columns.
    - Removes '<>' from header names.
    - Ensures DATE and TIME is the right length.
    - Combines DATE and TIME to DATETIME.
    - Sets index to DATETIME and sorts.
    """

    # Read the file (returns data frame)
    df = pd.read_csv(
        path,
        header=0,   # First row is the header
        dtype={"<TICKER>": "string",
               "<DATE>"  : "string",
               "<TIME>"  : "string"}, # Convert ticker to str (without it would be obj)
    )

    # Verify if input data has all needed columns
    verify_input_data(df, path)

    # Remove '<>' from header names
    df.columns = df.columns.str.strip("<>")

    # Ensure DATE and TIME is the right length - needed for valid combining
    df["DATE"] = df["DATE"].str.zfill(8)    # YYYYMMDD
    df["TIME"] = df["TIME"].str.zfill(6)    # HHMMSS

    # Combine DATE and TIME to DATETIME
    df["DATETIME"] = pd.to_datetime(
        df["DATE"] + df["TIME"],
        format="%Y%m%d%H%M%S",
        errors="coerce" # turn any malformed rows into NaT so the don't crash the parse
    )

    # Drop the original DATE and TIME columns - replaced by DATETIME
    df = df.drop(columns=["DATE", "TIME"])

    # Set index to DATETIME and sort
    df = df.set_index("DATETIME").sort_index()

    return df


def compute_sma(df: pd.DataFrame, sma_range: list[int]) -> pd.DataFrame:
    """
    Append short- and long-horizon Simple Moving Averages (SMAs) to *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'CLOSE' column.  The DataFrame is modified in place.
    short : int
        Window length for the short SMA (e.g. 20).
    long : int
        Window length for the long SMA (e.g. 50).

    Notes
    -----
    - New columns are named 'SMA_<short>' and 'SMA_<long>'.
    - Values are NaN until the rolling window is fully populated
      (`min_periods` equals the window length).
    """
    start, stop, step = sma_range

    # Calculate SMA window lengths
    sma_windows = range(start, stop + 1, step)

    # Extract close price column
    price = df["PRICE"]

    # Calculate SMA and is separate off-frame
    result_columns = {
            f"SMA_{window}": price.rolling(window=window, min_periods=window).mean()
            for window in sma_windows
    }

    # Create a new DataFrame holding all computed SMA columns,
    # aligned to the original DataFrame's index for correct joining
    sma_df = pd.DataFrame(result_columns, index=df.index)

    # Join to main data frame
    return df.join(sma_df)


def add_sma_crossover_signals(df: pd.DataFrame,
                              sma_short_range: list[int],
                              sma_long_range:  list[int]
) -> dict[tuple[int, int], str]:
    """
    Add SMA crossover signals to DataFrame.

    For every (short, long) pair in the given ranges (with short < long),
    detect SMA crossovers and add a 'Signal_<short>_<long>' column to DataFrame.

    Each signal column contains:
      - 'Buy' at bars where SMA_short crosses above SMA_long,
      - 'Sell' at bars where SMA_short crosses below SMA_long,
      - 'No signal (previous was …)' everywhere else.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'CLOSE' column. The DataFrame is modified in place.
    sma_short_range : list[int]
        Range for SMA short as three ints: min max step
    sma_long_range : list[int]
        Range for SMA long as three ints: min max step

    Returns
    -------
    latest_signal : dict[(short, long) -> str]
        The most recent label for each (short, long) pair, useful for summaries.
    """

    # Unpack input arguments tuple
    (smin, smax, sstep) = sma_short_range
    (lmin, lmax, lstep) = sma_long_range

    # Initialize dictionary to store latest signal for each (short, long) pair
    latest_signal: dict[tuple[int, int], str] = {}

    # Initialize dictionary to store signal columns
    signal_columns: dict[str, pd.Series] = {}

    # Iterate over all (short, long) pairs in the given ranges
    for short_window in range(smin, smax + 1, sstep):
        for long_window in range(lmin, lmax + 1, lstep):

            # Skip invalid pairs
            if short_window >= long_window:
                print("[ERROR] short window >= long_window")
                sys.exit()
                continue
            
            # Names of columns
            short_column_name = f"SMA_{short_window}"
            long_column_name  = f"SMA_{long_window}"

            # Skip pairs if columns not present
            if short_column_name not in df.columns or long_column_name not in df.columns:
                print("[ERROR] Columns not present")
                sys.exit()
                continue

            # Calculate difference betwen short and long SMA
            # sma_diff > 0: short SMA is above long SMA
            # sma_diff < 0: short SMA is below long SMA
            # sma_diff ==0: both SMAs are equal
            #
            # The sign of sma_diff tells the state:
            # positive: short above long; negative: short below long
            #
            # When the sign changes from negative to positive: Buy signal
            # When the sign changes from positive to negative: Sell signal
            #
            # If one or both are NaN, then sma_diff is NaN
            #
            # sma_diff is one-dimensional vector of numbers, indexed like DataFrame (df)
            # Keep as series because it is faster for vectorized math
            sma_diff_vector = df[short_column_name] - df[long_column_name]

            # Bar (both SMA data in the same row) is valid if both SMAs exist on that bar (are not NaN)
            # df[[s_col, l_col]] -> Select SMA short and long columns
            # .notna() -> Returns True if value is not NaN; else False
            # .all(axis=1) -> checks across the columns for each row (horizontally)
            # The result is:
            # True:  bar is valid
            # False: bar is invalid
            both_smas_valid = df[[short_column_name, long_column_name]].notna().all(axis=1)

            # Check if current and previous bar are valid (to detect cross, previous and new value is needed)
            # .shift() moves the whole series down by one row (row i to i-1)
            both_smas_valid_prev = both_smas_valid & both_smas_valid.shift(fill_value=False)

            # Detect buy crossover rows. Type is boolean mask (series), that can be used as indexer
            # To get all rows with Buy signals, use: df.loc[buy_mask_vector]
            buy_mask_vector  = (sma_diff_vector > 0) & (sma_diff_vector.shift() <= 0) & both_smas_valid_prev
            # Detect sell crossover rows 
            sell_mask_vector = (sma_diff_vector < 0) & (sma_diff_vector.shift() >= 0) & both_smas_valid_prev

            # Build pandas Series of signals (object dtype, 1-D labeled array) filled with missing values
            signal_series = pd.Series(pd.NA, index=df.index, dtype="object")
            # Set Buy and Sell signals
            signal_series.loc[buy_mask_vector]  = "Buy"
            signal_series.loc[sell_mask_vector] = "Sell"

            '''
            # Optional code to convert NaN to "NaN"
            last_signal = "None"
            trade_signals: list[str] = []

            for label in trade_signals_raw.tolist():
                if pd.isna(label):
                    trade_signals.append("NaN")
                else:
                    trade_signals.append(label)
                    last_signal = label
            '''

            # Store under a descriptive column name
            signal_column_name = f"Sig_{short_window}_{long_window}"
            df[signal_column_name] = signal_series

            # Get the latest non-NA signal for this pair
            last_signal_non_missing = signal_series.dropna()
            latest_signal[(short_window, long_window)] = (
                    last_signal_non_missing.iloc[-1] if not last_signal_non_missing.empty else "NaN"
            )

    # Join signals with main df
    # Attach all signal columns at once (keeps original df order; avoids SettingWithCopy)
    if signal_columns:
        # Create signals data frame
        signals_df = pd.DataFrame(signal_columns, index=df.index)
        # Concatenate signals data frame to main data frame
        df[signals_df.columns] = signals_df

    return latest_signal

def verify_input_data(df: pd.DataFrame, path: str) -> None:
    """
    Check if input data has all needed columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame to check.
    path : str
        Path to the input data file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the required columns are missing.

    Notes
    -----   
    - Required columns are: "<TICKER>", "<DATE>", "<TIME>", "<CLOSE>".
    - If any of the required columns are missing, raises a ValueError.
    """
    required_column = {"<TICKER>", "<DATE>", "<TIME>", "<CLOSE>"}
    if not required_column.issubset(df.columns):
        raise ValueError(f"File '{path}' missing required columns; found {', '.join(df.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# File‑level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: str,
                 sma_short_range: list[int],
                 sma_long_range: list[int],
                 out_dir: str
) -> None:
    """
    Process a single file.

    Parameters
    ----------
    path : str
        Path to the input data file.
    sma_short_range : list[int]
        Range for SMA short as three ints: min max step
    sma_long_range : list[int]
        Range for SMA long as three ints: min max step
    out_dir : str
        Root output directory

    Returns
    -------
    None

    Notes
    -----
    - Empty input files are skipped silently.
    """
    if os.path.getsize(path) == 0:
        return None  # silently skip empty files

    # Read data from file and preprocess it
    df = load_ohlc_from_file(path)

    # Close price is the basis for calculation
    df = df.rename(columns={"CLOSE": "PRICE"})

    # Compute SMA short and long
    df = compute_sma(df, sma_short_range)
    df = compute_sma(df, sma_long_range)

    # Trade signals: add trading signals when SMA short corsses SMA long
    latest_signal = add_sma_crossover_signals(df, sma_short_range, sma_long_range)

    # Drop all `SMA_` columns
    # df = df.drop(df.filter(like="SMA_").columns, axis=1)
    # Drop other columns
    df = df.drop(columns=["PER", "OPEN", "HIGH", "LOW", "VOL", "OPENINT"])

    # Get ticker name - will be used as file prefix
    ticker = os.path.splitext(os.path.basename(path))[0]

    # Store all data to csv
    all_data_file_name = os.path.join(out_dir, f"{ticker}-signals.txt")
    df.to_csv(all_data_file_name, float_format="%.4f")


    # ----- Filter for signals only -----
    # Drop all `SMA_` columns
    #df = df.drop(df.filter(like="SMA_").columns, axis=1)
    # Drop other columns
    #df = df.drop(columns=["PER", "OPEN", "HIGH", "LOW", "VOL", "OPENINT"])

    # Save signals to csv
    #signals_file_name = os.path.join(out_dir, f"{ticker}-signals.txt")
    #df.to_csv(signals_file_name, float_format="%.4f")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main function.

    Returns
    -------
    None

    Notes
    -----
    - Parses command line arguments.
    - Creates output directory.
    - Converts single arguments to range to unify processing.
    - Creates a worker function with most of its parameters already bound.
    - Runs sequentially or in parallel.
    """
    args = parse_args()

    # If --config is provided, load it
    if args.config:
        cfg = load_toml_config(args.config)
        merge_config_into_args(args, cfg)

    # Validate and prepare arguments
    validate_args(args)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Convert single arguments to range to unify processing
    if args.mode == "single":
        args.sma_short_range = [args.sma_short, args.sma_short, 1]
        args.sma_long_range  = [args.sma_long,  args.sma_long,  1]

    # Create a “worker” function with most of its parameters already  bound (curried) so that each call only
    # needs the filename.
    #   process_file(...)   – user-defined function that processes one CSV or JSON
    #   sma_short_range, sma_long_range – ranges for simple moving averages
    #   out_dir             – where results should be written
    worker = partial(
        process_file,
        sma_short_range=args.sma_short_range,
        sma_long_range=args.sma_long_range,
        out_dir=args.out_dir
    )

    # Run sequentially
    if args.jobs == 1 or len(args.files) == 1:
        # map() returns an iterator, not a list
        for _ in map(worker, args.files):
            pass
    # Run in parallel
    else:
        max_workers = min(args.jobs, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map() returns an iterator, not a list
            for _ in executor.map(worker, args.files):
                pass

if __name__ == "__main__":
    main()