"""Board encodings: one-hot for DQN, packed integer indices for n-tuple lookup."""
from __future__ import annotations
import numpy as np


def one_hot_log2(board: np.ndarray, n_buckets: int = 16) -> np.ndarray:
    """(4,4) int -> (16*n_buckets,) float32 flat one-hot."""
    flat = np.clip(board.reshape(-1).astype(np.int64), 0, n_buckets - 1)
    out = np.zeros((flat.size, n_buckets), dtype=np.float32)
    out[np.arange(flat.size), flat] = 1.0
    return out.reshape(-1)


def one_hot_log2_batch(boards: np.ndarray, n_buckets: int = 16) -> np.ndarray:
    """(N,4,4) -> (N, 16*n_buckets) float32."""
    N = boards.shape[0]
    flat = np.clip(boards.reshape(N, -1).astype(np.int64), 0, n_buckets - 1)
    out = np.zeros((N, flat.shape[1], n_buckets), dtype=np.float32)
    np.put_along_axis(out, flat[:, :, None], 1.0, axis=2)
    return out.reshape(N, -1)


def tuple_index(board: np.ndarray, cells) -> int:
    """Pack the log2 values at ``cells`` into a base-16 integer key."""
    idx = 0
    for r, c in cells:
        idx = (idx << 4) | int(board[r, c])
    return idx


def tuple_index_batch(boards: np.ndarray, cells) -> np.ndarray:
    """(N,4,4) -> (N,) int64 of packed indices."""
    idx = np.zeros(boards.shape[0], dtype=np.int64)
    for r, c in cells:
        idx = (idx << 4) | boards[:, r, c].astype(np.int64)
    return idx
