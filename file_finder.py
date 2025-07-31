"""
file_finder.py
A small utility library for discovering files that match a glob pattern.

Typical use
-----------
import argparse
from file_finder import add_find_files_args, find_files

parser = argparse.ArgumentParser(description="My data processor")
add_find_files_args(parser)           # adds --dir, --pattern, --recursive
args = parser.parse_args()

for path in find_files(args.dir, args.pattern, args.recursive):
    ...  # do something with the file

You can also call file_finder.main() explicitly from another
module or from `python -m file_finder --help` if you need a quick CLI.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence


# --------------------------------------------------------------------------- #
# Core functionality
# --------------------------------------------------------------------------- #
def find_files(
    directory: str | Path = ".",
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    Return a list of paths beneath *directory* that match *pattern*.

    Parameters
    ----------
    directory : str | Path, default="."
        Starting directory.  `"."` means the current working directory.
    pattern : str, default="*"
        Any valid glob expression, e.g. `"*.txt"` or `"*-signals.txt"`.
    recursive : bool, default=False
        If True, search all sub-directories (Path.rglob); otherwise
        search only *directory* (Path.glob).

    Notes
    -----
    Symbolic links are followed.  Only regular files are returned.
    """
    base = Path(directory).expanduser().resolve()
    searcher: Iterable[Path] = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in searcher if p.is_file()]


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def add_find_files_args(parser: argparse.ArgumentParser) -> None:
    """
    Inject ``--dir``, ``--pattern`` and ``--recursive`` options into *parser*.

    Example
    -------
    >>> import argparse, file_finder
    >>> p = argparse.ArgumentParser()
    >>> file_finder.add_find_files_args(p)
    >>> ns = p.parse_args(["--pattern", "*.txt", "--recursive"])
    """
    parser.add_argument(
        "--dir",
        default=".",
        metavar="PATH",
        help="directory to search (default: current directory)",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*",
        metavar="GLOB",
        help='glob pattern to match (default: "*")',
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="search sub-directories recursively",
    )


def parse_find_files_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Stand-alone parser for callers that *only* need our arguments.

    Returns
    -------
    argparse.Namespace with attributes ``dir``, ``pattern`` and ``recursive``.
    """
    parser = argparse.ArgumentParser(
        prog="file_finder",
        description="List files matching a glob pattern.",
    )
    add_find_files_args(parser)
    return parser.parse_args(argv)

# --------------------------------------------------------------------------- #
# Public re-exports
# --------------------------------------------------------------------------- #
__all__ = [
    "find_files",
    "add_find_files_args",
    "parse_find_files_args",
]
