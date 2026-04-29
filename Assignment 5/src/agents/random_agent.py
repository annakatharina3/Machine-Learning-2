"""Random baseline: uniform over legal moves."""
from __future__ import annotations
import numpy as np
from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, board, legal_mask, greedy=False):
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            return 0
        return int(self.rng.choice(legal))
