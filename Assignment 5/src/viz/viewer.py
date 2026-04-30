"""FuncAnimation-based 2048 viewer.

play_game(agent, ckpt) renders an episode as a GIF and returns
(path, anim) so a notebook can also call ``HTML(anim.to_jshtml())``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ..env import Game2048Env, ACTION_NAMES
from ..utils import GIF_DIR


_TILE_COLORS = {
    0:  "#cdc1b4", 1:  "#eee4da", 2:  "#ede0c8", 3:  "#f2b179",
    4:  "#f59563", 5:  "#f67c5f", 6:  "#f65e3b", 7:  "#edcf72",
    8:  "#edcc61", 9:  "#edc850", 10: "#edc53f", 11: "#edc22e",
    12: "#3c3a32", 13: "#3c3a32", 14: "#3c3a32", 15: "#3c3a32",
}


def _draw_board(ax, board, score, steps, action_name, agent_name):
    ax.clear()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(4):
        for c in range(4):
            v = int(board[r, c])
            tile = 2 ** v if v > 0 else 0
            color = _TILE_COLORS.get(min(v, 15), "#3c3a32")
            ax.add_patch(plt.Rectangle((c, r), 0.96, 0.96, facecolor=color,
                                        edgecolor="#bbada0", linewidth=2))
            if tile > 0:
                fontsize = 24 if tile < 1000 else 18
                txt_color = "#776e65" if v <= 2 else "#f9f6f2"
                ax.text(c + 0.48, r + 0.48, str(tile), ha="center", va="center",
                        fontsize=fontsize, fontweight="bold", color=txt_color)
    ax.set_title(f"{agent_name} | score={score} | step={steps} | last={action_name}",
                 fontsize=11)


def play_game(agent, checkpoint_path: Optional[str | Path] = None, save_gif: bool = True,
              fps: int = 4, seed: int = 42, max_steps: int = 5000,
              out_dir: Optional[Path] = None):
    if checkpoint_path is not None and hasattr(agent, "load"):
        try:
            agent.load(checkpoint_path)
        except Exception as e:
            print(f"warning: could not load checkpoint {checkpoint_path}: {e}")
    env = Game2048Env(seed=seed)
    board = env.reset(seed=seed)
    legal = env.legal_actions()
    history = [(board.copy(), 0, 0, "—")]
    done = False
    while not done and len(history) < max_steps:
        a = agent.act(board, legal, greedy=True)
        board, _, done, _ = env.step(a)
        legal = env.legal_actions() if not done else np.zeros(4, dtype=bool)
        history.append((board.copy(), env.score, env.steps, ACTION_NAMES[a]))

    fig, ax = plt.subplots(figsize=(5, 5.5))
    agent_name = getattr(agent, "name", agent.__class__.__name__)

    def update(i):
        b, s, st, an = history[i]
        _draw_board(ax, b, s, st, an, agent_name)

    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                    interval=1000 // fps, repeat=False)
    out_path: Optional[Path] = None
    if save_gif:
        out_dir = Path(out_dir) if out_dir is not None else GIF_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{agent_name}_{seed}.gif"
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
        except Exception as e:
            print(f"warning: gif save failed: {e}")
            out_path = None
    plt.close(fig)
    return out_path, anim


def compare_side_by_side(agent_a, agent_b, ckpt_a=None, ckpt_b=None,
                          sync_seed: int = 42, fps: int = 4, max_steps: int = 5000,
                          out_dir: Optional[Path] = None):
    if ckpt_a is not None:
        agent_a.load(ckpt_a)
    if ckpt_b is not None:
        agent_b.load(ckpt_b)
    env_a = Game2048Env(seed=sync_seed)
    env_b = Game2048Env(seed=sync_seed)
    ba = env_a.reset(seed=sync_seed)
    bb = env_b.reset(seed=sync_seed)
    history_a = [(ba.copy(), 0, 0, "—")]
    history_b = [(bb.copy(), 0, 0, "—")]
    done_a = done_b = False
    while (not done_a or not done_b) and len(history_a) < max_steps:
        if not done_a:
            la = env_a.legal_actions()
            aa = agent_a.act(ba, la, greedy=True)
            ba, _, done_a, _ = env_a.step(aa)
            history_a.append((ba.copy(), env_a.score, env_a.steps, ACTION_NAMES[aa]))
        if not done_b:
            lb = env_b.legal_actions()
            ab = agent_b.act(bb, lb, greedy=True)
            bb, _, done_b, _ = env_b.step(ab)
            history_b.append((bb.copy(), env_b.score, env_b.steps, ACTION_NAMES[ab]))

    n = max(len(history_a), len(history_b))
    while len(history_a) < n:
        history_a.append(history_a[-1])
    while len(history_b) < n:
        history_b.append(history_b[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    name_a = getattr(agent_a, "name", agent_a.__class__.__name__)
    name_b = getattr(agent_b, "name", agent_b.__class__.__name__)

    def update(i):
        ba_, sa, sta, ana = history_a[i]
        bb_, sb, stb, anb = history_b[i]
        _draw_board(axes[0], ba_, sa, sta, ana, name_a)
        _draw_board(axes[1], bb_, sb, stb, anb, name_b)

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 // fps, repeat=False)
    out_dir = Path(out_dir) if out_dir is not None else GIF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{name_a}_vs_{name_b}.gif"
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception as e:
        print(f"warning: gif save failed: {e}")
        out_path = None
    plt.close(fig)
    return out_path, anim
