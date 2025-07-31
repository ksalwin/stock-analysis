from pathlib import Path
import glob

def find_files(
    path: str = ".",           # search here by default
    pattern: str = "*",        # match everything if no pattern given
    recursive: bool = False
) -> list[str]:
    """
    Return a list of files whose names match *pattern* inside *path*.

    Parameters
    ----------
    path : str, optional
        Directory to search.  Defaults to current directory (``"."``).
    pattern : str, optional
        Glob pattern such as ``"*.txt"`` or ``"*-signals.txt"``.  
        Defaults to ``"*"``, which matches **all** files.
    recursive : bool, optional
        Search sub‑directories as well (default ``False``).

    Returns
    -------
    list[str]
        Absolute paths (as strings) of matching files.  
        Empty list if nothing matched.

    Examples
    --------
    >>> find_files(pattern="*.txt")
    ['./a.txt', './b.txt']

    >>> find_files("/tmp/results", "*-signals.txt")
    ['/tmp/results/slv-10-90-signals.txt', ...]

    >>> find_files(".", recursive=True)              # every file under cwd
    [...]
    """
    search_root = Path(path).expanduser()

    if recursive:
        # Path.rglob handles recursive globs natively
        return [str(p) for p in search_root.rglob(pattern)]
    else:
        # str() needed because glob.glob works with strings
        return glob.glob(str(search_root / pattern))
