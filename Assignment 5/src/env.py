"""Game2048Env + VectorGame2048Env with a tiny Gym-style API.

We avoid a hard dep on gymnasium — the surface is small enough to roll our own.
"""
from __future__ import annotations
import numpy as np

from .moves import (apply_move, ACTION_UP, ACTION_RIGHT, ACTION_DOWN,
                     ACTION_LEFT, N_ACTIONS)


ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTIONS = (ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT)


class Game2048Env:
    """Single-board 2048 environment.

    Observation: ``np.ndarray((4,4), int8)`` of log2 values (0 = empty).
    Reward: sum of log2 values of newly-formed tiles on the move.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int8)
        self.score = 0
        self.steps = 0
        self.done = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board[:] = 0
        self._spawn()
        self._spawn()
        self.score = 0
        self.steps = 0
        self.done = False
        return self.board.copy()

    def _spawn(self) -> None:
        empties = np.argwhere(self.board == 0)
        if len(empties) == 0:
            return
        idx = empties[self.rng.integers(len(empties))]
        self.board[idx[0], idx[1]] = 1 if self.rng.random() < 0.9 else 2

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for a in range(N_ACTIONS):
            _, _, changed = apply_move(self.board, a)
            mask[a] = changed
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Cannot step a terminated env. Call reset().")
        new_board, reward, changed = apply_move(self.board, action)
        info: dict = {"changed": bool(changed)}
        if not changed:
            info["illegal"] = True
            self.done = self._is_terminal()
            return self.board.copy(), 0.0, self.done, info
        self.board = new_board
        self.score += reward
        self.steps += 1
        self._spawn()
        self.done = self._is_terminal()
        info.update(score=self.score, steps=self.steps,
                    max_tile=int(2 ** int(self.board.max())) if self.board.max() > 0 else 0)
        return self.board.copy(), float(reward), self.done, info

    def _is_terminal(self) -> bool:
        return not bool(self.legal_actions().any())

    def render(self, mode: str = "ansi"):
        if mode == "ansi":
            lines = []
            for row in self.board:
                lines.append(" ".join(f"{(2**v if v > 0 else 0):>5d}" for v in row))
            return "\n".join(lines)
        if mode == "rgb":
            arr = np.zeros((4, 4, 3), dtype=np.uint8)
            for v in range(16):
                hue = int(255 * v / 15)
                arr[self.board == v] = (hue, max(0, 200 - hue // 2), 100)
            return arr
        raise ValueError(f"Unknown mode {mode}")


class VectorGame2048Env:
    """Vectorized 2048 environment for batched DQN rollouts."""

    def __init__(self, num_envs: int, seed: int | None = None):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((num_envs, 4, 4), dtype=np.int8)
        self.scores = np.zeros(num_envs, dtype=np.int64)
        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.dones = np.ones(num_envs, dtype=bool)

    def reset(self) -> np.ndarray:
        self.boards[:] = 0
        self.scores[:] = 0
        self.steps[:] = 0
        self.dones[:] = False
        for i in range(self.num_envs):
            self._spawn(i)
            self._spawn(i)
        return self.boards.copy()

    def _spawn(self, i: int) -> None:
        empties = np.argwhere(self.boards[i] == 0)
        if len(empties) == 0:
            return
        idx = empties[self.rng.integers(len(empties))]
        self.boards[i, idx[0], idx[1]] = 1 if self.rng.random() < 0.9 else 2

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros((self.num_envs, N_ACTIONS), dtype=bool)
        for i in range(self.num_envs):
            for a in range(N_ACTIONS):
                _, _, changed = apply_move(self.boards[i], a)
                mask[i, a] = changed
        return mask

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            if self.dones[i]:
                continue
            nb, r, changed = apply_move(self.boards[i], int(actions[i]))
            if changed:
                self.boards[i] = nb
                self.scores[i] += r
                self.steps[i] += 1
                rewards[i] = float(r)
                self._spawn(i)
            self.dones[i] = self._is_terminal(i)
        return self.boards.copy(), rewards, self.dones.copy(), {}

    def _is_terminal(self, i: int) -> bool:
        for a in range(N_ACTIONS):
            _, _, changed = apply_move(self.boards[i], a)
            if changed:
                return False
        return True

    def reset_done(self) -> None:
        for i in range(self.num_envs):
            if self.dones[i]:
                self.boards[i] = 0
                self.scores[i] = 0
                self.steps[i] = 0
                self.dones[i] = False
                self._spawn(i)
                self._spawn(i)
