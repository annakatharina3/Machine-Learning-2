"""N-tuple network with TD(0) on afterstates (Szubert & Jaśkowski, CIG 2014).

Patterns: two 6-tuples + two 4-tuples, each augmented with the 8-fold
dihedral symmetry group (4 rotations x 2 reflections). Weight tables are
shared within a symmetry group, so each board lookup involves
``sum(len(g) for g in sym_groups) ~ 32`` table reads.

The value V is defined on **afterstates** (the deterministic outcome of an
action, before the random spawn). Update rule:
    V(a_t) <- V(a_t) + alpha * (r_{t+1} + V(a_{t+1}) - V(a_t))
where r_{t+1} is the reward of the next greedy action from the spawn-perturbed
state s_{t+1}, and a_{t+1} is the resulting afterstate.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .base import Agent
from ..moves import apply_move


# Szubert & Jaśkowski 2014 layout — two 6-tuples + two 4-tuples.
_PATTERN_A = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1))   # axe (top row + 2)
_PATTERN_B = ((1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1))   # second-row axe
_PATTERN_C = ((0, 0), (0, 1), (0, 2), (0, 3))                   # row
_PATTERN_D = ((0, 0), (0, 1), (1, 0), (1, 1))                   # 2x2 square

PATTERNS = (_PATTERN_A, _PATTERN_B, _PATTERN_C, _PATTERN_D)


def _dihedral(cells, n: int = 4):
    """All distinct images of `cells` under the 8-element dihedral group on an nxn board."""
    out: list[tuple] = []
    cur = tuple(cells)
    for _ in range(4):
        out.append(cur)
        out.append(tuple((r, n - 1 - c) for (r, c) in cur))   # reflect over vertical axis
        cur = tuple((c, n - 1 - r) for (r, c) in cur)         # rotate 90° clockwise
    seen, uniq = set(), []
    for t in out:
        key = tuple(sorted(t))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


class NTupleNetwork:
    def __init__(self, patterns=PATTERNS):
        self.patterns = patterns
        self.sym_groups = [tuple(_dihedral(p)) for p in patterns]
        self.weights = [np.zeros(16 ** len(p), dtype=np.float32) for p in patterns]
        self.num_lookups = sum(len(g) for g in self.sym_groups)

    @staticmethod
    def _packed_idx(board: np.ndarray, cells) -> int:
        idx = 0
        for r, c in cells:
            idx = (idx << 4) | int(board[r, c])
        return idx

    def value(self, board: np.ndarray) -> float:
        v = 0.0
        for w, sym in zip(self.weights, self.sym_groups):
            for cells in sym:
                v += float(w[self._packed_idx(board, cells)])
        return v

    def update(self, board: np.ndarray, delta: float, alpha: float) -> None:
        """Add ``alpha * delta`` to every weight visited by ``board``."""
        for w, sym in zip(self.weights, self.sym_groups):
            for cells in sym:
                w[self._packed_idx(board, cells)] += alpha * delta

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"w{i}": w for i, w in enumerate(self.weights)})

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"w{i}"]


class NTupleAgent(Agent):
    name = "ntuple"

    def __init__(self, alpha: float = 0.1, network: NTupleNetwork | None = None):
        self.network = network or NTupleNetwork()
        self.alpha_base = float(alpha)
        # Effective per-weight learning rate so that one TD update changes the
        # board's value by roughly `alpha * delta` (each weight is touched once
        # per lookup; total update across all lookups = alpha * delta).
        self.alpha = self.alpha_base / self.network.num_lookups
        self.episode_count = 0

    def best_afterstate(self, board: np.ndarray, legal_mask: np.ndarray):
        """Greedy choice. Returns ``(action, reward, afterstate, value)``."""
        best_a, best_after, best_r, best_s = -1, None, 0.0, -np.inf
        for a in np.flatnonzero(legal_mask):
            after, r, _ = apply_move(board, int(a))
            s = float(r) + self.network.value(after)
            if s > best_s:
                best_s = s
                best_after = after
                best_a = int(a)
                best_r = float(r)
        return best_a, best_r, best_after, best_s

    def act(self, board, legal_mask, greedy=False):
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            return 0
        a, _, _, _ = self.best_afterstate(board, legal_mask)
        return int(a) if a >= 0 else int(legal[0])

    def save(self, path):
        self.network.save(path)

    def load(self, path):
        self.network.load(path)
