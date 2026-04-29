"""Abstract Agent interface. All agents accept the same legal-mask interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, board: np.ndarray, legal_mask: np.ndarray, greedy: bool = False) -> int:
        ...

    def save(self, path: str | Path) -> None:
        return None

    def load(self, path: str | Path) -> None:
        return None
