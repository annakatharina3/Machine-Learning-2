"""N-tuple TD(0) training loop on afterstates."""
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
import numpy as np

from ..env import Game2048Env
from ..agents.ntuple import NTupleAgent
from .logger import CSVLogger


def train_ntuple(agent: NTupleAgent, episodes: int, log_path, ckpt_dir,
                 ckpt_every: int = 2000, seed: int = 42, verbose: bool = True):
    env = Game2048Env(seed=seed)
    logger = CSVLogger(log_path)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling: deque = deque(maxlen=100)
    best_avg = -np.inf
    t0 = time.time()
    for ep in range(episodes):
        board = env.reset()
        ep_return = 0.0
        ep_max_log = 0
        steps = 0
        td_errs = []
        legal = env.legal_actions()
        if not legal.any():
            continue
        a, r, after, _ = agent.best_afterstate(board, legal)
        while True:
            board2, _, done, _ = env.step(a)
            ep_return += r
            steps += 1
            ep_max_log = max(ep_max_log, int(board2.max()))
            if done:
                td = 0.0 - agent.network.value(after)
                agent.network.update(after, td, agent.alpha)
                td_errs.append(abs(td))
                break
            legal2 = env.legal_actions()
            a2, r2, after2, _ = agent.best_afterstate(board2, legal2)
            target = r2 + agent.network.value(after2)
            td = target - agent.network.value(after)
            agent.network.update(after, td, agent.alpha)
            td_errs.append(abs(td))
            a, r, after = a2, r2, after2
        agent.episode_count += 1
        rolling.append(ep_return)
        avg = float(np.mean(rolling))
        max_tile = int(2 ** ep_max_log) if ep_max_log > 0 else 0
        logger.log(episode=ep + 1, **{
            "return": ep_return,
            "max_tile": max_tile,
            "steps": steps,
            "eps_or_alpha": agent.alpha_base,
            "loss_or_td_error": round(float(np.mean(td_errs)), 5) if td_errs else "",
            "wallclock_s": round(time.time() - t0, 1),
        })
        if (ep + 1) % ckpt_every == 0:
            agent.save(ckpt_dir / "latest.npz")
        if ep >= 100 and avg > best_avg:
            best_avg = avg
            agent.save(ckpt_dir / "best.npz")
        if verbose and (ep + 1) % 500 == 0:
            print(f"ep {ep+1:6d} | avg100 {avg:8.1f} | max_tile {max_tile:5d} | alpha {agent.alpha_base}")
    agent.save(ckpt_dir / "latest.npz")
    return ckpt_dir / "latest.npz"
