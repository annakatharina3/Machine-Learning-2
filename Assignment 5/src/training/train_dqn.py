"""DQN training loop on Game2048Env."""
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
import numpy as np

from ..env import Game2048Env
from ..agents.dqn import DQNAgent
from .logger import CSVLogger


def train_dqn(agent: DQNAgent, episodes: int, log_path, ckpt_dir,
              ckpt_every: int = 2000, learn_every: int = 4,
              warmup_steps: int = 1000, seed: int = 42, verbose: bool = True):
    env = Game2048Env(seed=seed)
    logger = CSVLogger(log_path)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling: deque = deque(maxlen=100)
    best_avg = -np.inf
    t0 = time.time()
    total_env_steps = 0
    for ep in range(episodes):
        board = env.reset()
        ep_return = 0.0
        ep_max_log = 0
        steps = 0
        losses = []
        legal = env.legal_actions()
        done = False
        while not done:
            a = agent.act(board, legal, greedy=False)
            next_board, r, done, info = env.step(a)
            next_legal = env.legal_actions() if not done else np.zeros(4, dtype=bool)
            agent.remember(board, a, r, next_board, done, next_legal)
            ep_return += r
            ep_max_log = max(ep_max_log, int(next_board.max()))
            board = next_board
            legal = next_legal
            steps += 1
            total_env_steps += 1
            if total_env_steps > warmup_steps and total_env_steps % learn_every == 0:
                lo = agent.learn()
                if lo is not None:
                    losses.append(lo)
        agent.episode_count += 1
        rolling.append(ep_return)
        avg = float(np.mean(rolling))
        max_tile = int(2 ** ep_max_log) if ep_max_log > 0 else 0
        logger.log(episode=ep + 1, **{
            "return": ep_return,
            "max_tile": max_tile,
            "steps": steps,
            "eps_or_alpha": round(agent.epsilon, 4),
            "loss_or_td_error": round(float(np.mean(losses)), 5) if losses else "",
            "wallclock_s": round(time.time() - t0, 1),
        })
        if (ep + 1) % ckpt_every == 0:
            agent.save(ckpt_dir / "latest.pt")
        if ep >= 100 and avg > best_avg:
            best_avg = avg
            agent.save(ckpt_dir / "best.pt")
        if verbose and (ep + 1) % 100 == 0:
            print(f"ep {ep+1:6d} | avg100 {avg:8.1f} | max_tile {max_tile:5d} | eps {agent.epsilon:.3f}")
    agent.save(ckpt_dir / "latest.pt")
    return ckpt_dir / "latest.pt"
