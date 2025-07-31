# progressbar.py
import sys
import shutil
import time

class ProgressBar:
    """Simple console progress bar.

    Parameters
    ----------
    total : int
        How many units of work make 100 %.
    width : int | None
        Desired width of the bar (characters).  If None, we
        auto-size to the current terminal minus a small margin.
    fill : str
        Character to draw completed portion.
    empty : str
        Character to draw remaining portion.
    show_percent : bool
        Whether to append a '  42 %' style percentage.

    Example use
    -----------
    from progressbar import ProgressBar

    work_units = 50
    pb = ProgressBar(total=work_units, width=40)

    for i in range(work_units):
        // some delay
        pb.update()         # advance one unit

    # pb.finish() is optional—it's called automatically at 100 %
    """
    def __init__(self, total, width=None, *, fill="#", empty=" ", show_percent=True):
        if total <= 0:
            raise ValueError("total must be > 0")
        self.total = total
        self.current = 0
        self.fill = fill
        self.empty = empty
        self.show_percent = show_percent

        term_cols = shutil.get_terminal_size(fallback=(80, 20)).columns
        reserved = 8 if show_percent else 0      # space for '100 %'
        self.width = min(width or (term_cols - 10), term_cols - reserved - 2)
        if self.width < 4:
            self.width = 4

        # Draw the empty bar immediately
        self._draw()

    # ------------------------------------------------------------------ public
    def update(self, step=1):
        """Advance the bar by *step* units (default = 1)."""
        self.current = min(self.current + step, self.total)
        self._draw()
        if self.current == self.total:
            self.finish()

    def finish(self):
        """Fill the bar, print newline, and disable further updates."""
        if self.current < self.total:
            self.current = self.total
            self._draw()
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.update = lambda *a, **kw: None   # make further calls no-ops

    # ------------------------------------------------------------------ private
    def _draw(self):
        ratio = self.current / self.total
        filled_len = int(self.width * ratio)
        bar = (
            "[" +
            self.fill * filled_len +
            self.empty * (self.width - filled_len) +
            "]"
        )
        if self.show_percent:
            bar += f" {ratio:6.1%}"
        # \r returns to start of line; end='' avoids extraneous newline
        sys.stdout.write("\r" + bar)
        sys.stdout.flush()

