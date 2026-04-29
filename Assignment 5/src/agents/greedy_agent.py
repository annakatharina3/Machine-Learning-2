"""Greedy heuristic baseline: argmax over legal moves of a hand-crafted score."""
from __future__ import annotations
import numpy as np
from .base import Agent
from ..moves import apply_move


def _monotonicity(board: np.ndarray) -> float:
    """Higher = rows/cols are more monotonically ordered (a common 2048 heuristic)."""
    score = 0.0
    for arr in (board, board.T):
        for row in arr:
            inc = sum(int(row[i + 1]) - int(row[i]) for i in range(3) if row[i + 1] >= row[i])
            dec = sum(int(row[i]) - int(row[i + 1]) for i in range(3) if row[i] >= row[i + 1])
            score -= min(inc, dec)
    return float(score)


def _empty_cells(board: np.ndarray) -> int:
    return int((board == 0).sum())


def _corner_bonus(board: np.ndarray) -> float:
    mx = int(board.max())
    if mx == 0:
        return 0.0
    if board[0, 0] == mx or board[0, 3] == mx or board[3, 0] == mx or board[3, 3] == mx:
        return float(mx)
    return 0.0


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, alpha: float = 1.0, beta: float = 5.0, gamma: float = 1.0,
                 seed: int | None = None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    def act(self, board, legal_mask, greedy=False):
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            return 0
        best_a, best_s = -1, -np.inf
        for a in legal:
            new_board, r, _ = apply_move(board, int(a))
            s = (float(r)
                 + self.alpha * _monotonicity(new_board)
                 + self.beta * _empty_cells(new_board)
                 + self.gamma * _corner_bonus(new_board))
            if s > best_s:
                best_s = s
                best_a = int(a)
        return best_a if best_a >= 0 else int(self.rng.choice(legal))
