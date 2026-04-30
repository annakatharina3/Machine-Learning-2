"""Run an agent for n_games and produce evaluation metrics."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Optional
import numpy as np

from ..env import Game2048Env


THRESHOLDS = (512, 1024, 2048, 4096, 8192)


def run_eval(agent, n_games: int = 1000, seed: int = 0,
             train_log: Optional[str | Path] = None) -> dict:
    env = Game2048Env(seed=seed)
    scores, max_tiles, all_steps = [], [], []
    for g in range(n_games):
        board = env.reset(seed=seed + g)
        legal = env.legal_actions()
        done = False
        while not done:
            a = agent.act(board, legal, greedy=True)
            board, _, done, _ = env.step(a)
            legal = env.legal_actions() if not done else np.zeros(4, dtype=bool)
        scores.append(env.score)
        max_tiles.append(int(2 ** int(board.max())) if board.max() > 0 else 0)
        all_steps.append(env.steps)
    s = np.array(scores)
    mt = np.array(max_tiles)
    st = np.array(all_steps)
    metrics = {
        "agent": getattr(agent, "name", agent.__class__.__name__),
        "n_games": n_games,
        "mean_score": float(s.mean()),
        "median_score": float(np.median(s)),
        "max_score": int(s.max()),
        "mean_max_tile": float(mt.mean()),
        "max_max_tile": int(mt.max()),
        "pct_reaching": {str(t): float((mt >= t).mean()) for t in THRESHOLDS},
        "mean_steps": float(st.mean()),
    }
    if train_log is not None and Path(train_log).exists():
        first1024 = first2048 = None
        with open(train_log) as f:
            for row in csv.DictReader(f):
                try:
                    mt_val = int(row.get("max_tile") or 0)
                    ep = int(row.get("episode"))
                except (TypeError, ValueError):
                    continue
                if first1024 is None and mt_val >= 1024:
                    first1024 = ep
                if first2048 is None and mt_val >= 2048:
                    first2048 = ep
        metrics["episodes_to_first_1024"] = first1024
        metrics["episodes_to_first_2048"] = first2048
    return metrics


def save_eval(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
