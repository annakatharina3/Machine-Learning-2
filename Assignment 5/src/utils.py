"""Common utilities: paths, RNG seeding, and torch device pick (mps > cuda > cpu)."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the env-only path
    torch = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"
EVAL_DIR = DATA_DIR / "eval"
GIF_DIR = DATA_DIR / "gifs"

for _d in (CKPT_DIR, LOG_DIR, EVAL_DIR, GIF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
