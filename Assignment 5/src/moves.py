"""Bitboard moves for 2048: precomputed row-shift/merge lookup tables.

The board is a (4, 4) ``int8`` of log2 values (0 = empty, k = tile 2**k).
Each row packs into a 16-bit unsigned int (4 nibbles), so the full set of
2**16 = 65536 possible rows fits in a flat lookup table.

The reward for a move is the sum of log2 values of *newly formed* tiles
(equivalently, the sum of the original 2048 score increments).
"""
from __future__ import annotations
import numpy as np


ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
N_ACTIONS = 4


def _int_to_row(x: int) -> tuple[int, int, int, int]:
    return ((x >> 12) & 0xF, (x >> 8) & 0xF, (x >> 4) & 0xF, x & 0xF)


def _row_to_int(row) -> int:
    return (int(row[0]) << 12) | (int(row[1]) << 8) | (int(row[2]) << 4) | int(row[3])


def _shift_merge_left(row):
    nz = [v for v in row if v > 0]
    out: list[int] = []
    reward = 0
    i = 0
    while i < len(nz):
        if i + 1 < len(nz) and nz[i] == nz[i + 1]:
            v = min(nz[i] + 1, 15)
            out.append(v)
            reward += v
            i += 2
        else:
            out.append(nz[i])
            i += 1
    out.extend([0] * (4 - len(out)))
    return tuple(out), reward


def _build_luts() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(65536, dtype=np.uint16)
    rew = np.zeros(65536, dtype=np.uint16)
    for r in range(65536):
        new_row, reward = _shift_merge_left(_int_to_row(r))
        left[r] = _row_to_int(new_row)
        rew[r] = reward
    return left, rew


LEFT_LUT, REWARD_LUT = _build_luts()


def _pack_rows(board: np.ndarray) -> np.ndarray:
    b = board.astype(np.uint16)
    return (b[:, 0] << 12) | (b[:, 1] << 8) | (b[:, 2] << 4) | b[:, 3]


def _unpack_rows(rows: np.ndarray) -> np.ndarray:
    out = np.empty((rows.shape[0], 4), dtype=np.int8)
    out[:, 0] = (rows >> 12) & 0xF
    out[:, 1] = (rows >> 8) & 0xF
    out[:, 2] = (rows >> 4) & 0xF
    out[:, 3] = rows & 0xF
    return out


def _reverse_packed(rows: np.ndarray) -> np.ndarray:
    return ((rows & 0xF) << 12) | ((rows & 0xF0) << 4) | ((rows & 0xF00) >> 4) | ((rows & 0xF000) >> 12)


def apply_move(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Apply a single move. Returns (new_board, reward, changed)."""
    if action == ACTION_LEFT:
        rows = _pack_rows(board)
        reward = int(REWARD_LUT[rows].sum())
        new_board = _unpack_rows(LEFT_LUT[rows])
    elif action == ACTION_RIGHT:
        rev = _reverse_packed(_pack_rows(board))
        reward = int(REWARD_LUT[rev].sum())
        new_board = _unpack_rows(_reverse_packed(LEFT_LUT[rev]))
    elif action == ACTION_UP:
        rows = _pack_rows(board.T)
        reward = int(REWARD_LUT[rows].sum())
        new_board = _unpack_rows(LEFT_LUT[rows]).T.copy()
    elif action == ACTION_DOWN:
        rev = _reverse_packed(_pack_rows(board.T))
        reward = int(REWARD_LUT[rev].sum())
        new_board = _unpack_rows(_reverse_packed(LEFT_LUT[rev])).T.copy()
    else:
        raise ValueError(f"Unknown action {action}")
    new_board = new_board.astype(np.int8)
    changed = not np.array_equal(new_board, board)
    return new_board, reward, changed


def apply_move_batch(boards: np.ndarray, actions: np.ndarray):
    """boards: (N,4,4), actions: (N,) -> (new_boards, rewards, changed)."""
    N = boards.shape[0]
    new = boards.copy()
    rew = np.zeros(N, dtype=np.int32)
    for i in range(N):
        nb, r, _ = apply_move(boards[i], int(actions[i]))
        new[i] = nb
        rew[i] = r
    changed = ~np.all(new == boards, axis=(1, 2))
    return new, rew, changed
