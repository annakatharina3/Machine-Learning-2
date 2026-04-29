"""Append-only CSV logger with a fixed schema."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable


COLUMNS = ("episode", "return", "max_tile", "steps", "eps_or_alpha",
           "loss_or_td_error", "wallclock_s")


class CSVLogger:
    def __init__(self, path: str | Path, columns: Iterable[str] = COLUMNS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = tuple(columns)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.columns)

    def log(self, **row) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self.columns])
