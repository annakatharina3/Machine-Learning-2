"""DQN agent: 256-d one-hot input, 3-layer MLP, action-masked target.

Hyperparameters follow the assignment spec:
  - Adam, lr=5e-4, gamma=0.99, smooth-L1 loss
  - target net hard sync every 1000 grad steps
  - epsilon linearly 1.0 -> 0.05 over 50k episodes
  - illegal Q-values clamped to -1e9 in BOTH online action selection and target bootstrap
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent
from ..encoding import one_hot_log2


class QNetwork(nn.Module):
    def __init__(self, in_dim: int = 256, hidden1: int = 512, hidden2: int = 256, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Ring buffer over preallocated NumPy arrays."""

    def __init__(self, capacity: int = 100_000, obs_dim: int = 256):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.idx = 0
        self.size = 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.legal2 = np.zeros((capacity, 4), dtype=np.float32)

    def add(self, s, a, r, s2, d, legal2):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.legal2[i] = legal2
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device):
        ix = np.random.randint(0, self.size, size=batch)

        def to_t(arr, dtype):
            return torch.from_numpy(arr[ix]).to(device=device, dtype=dtype)

        return (to_t(self.s, torch.float32),
                to_t(self.a, torch.long),
                to_t(self.r, torch.float32),
                to_t(self.s2, torch.float32),
                to_t(self.d, torch.float32),
                to_t(self.legal2, torch.float32))


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, device=None, lr: float = 5e-4, gamma: float = 0.99,
                 buffer_size: int = 100_000, batch_size: int = 512,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay_episodes: int = 50_000,
                 target_sync_steps: int = 1000):
        self.device = device or torch.device("cpu")
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.target_sync_steps = target_sync_steps
        self.episode_count = 0
        self.grad_steps = 0
        self.last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.episode_count / max(1, self.eps_decay_episodes))
        return float(self.eps_start + (self.eps_end - self.eps_start) * frac)

    def act(self, board, legal_mask, greedy=False):
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            return 0
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(legal))
        with torch.no_grad():
            x = torch.from_numpy(one_hot_log2(board)).unsqueeze(0).to(self.device)
            q = self.online(x).squeeze(0).cpu().numpy()
        q[~legal_mask] = -1e9
        return int(np.argmax(q))

    def remember(self, board, action, reward, next_board, done, legal_next_mask):
        s = one_hot_log2(board)
        s2 = one_hot_log2(next_board)
        self.buffer.add(s, action, reward, s2, float(done), legal_next_mask.astype(np.float32))

    def learn(self) -> float | None:
        if self.buffer.size < self.batch_size:
            return None
        s, a, r, s2, d, legal2 = self.buffer.sample(self.batch_size, self.device)
        with torch.no_grad():
            q2 = self.target(s2)
            q2 = q2.masked_fill(legal2 < 0.5, -1e9)
            max_q2 = q2.max(dim=1).values
            target = r + self.gamma * max_q2 * (1.0 - d)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "episode_count": self.episode_count,
            "grad_steps": self.grad_steps,
            "gamma": self.gamma,
            "eps_decay_episodes": self.eps_decay_episodes,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        try:
            self.opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
        self.episode_count = ckpt.get("episode_count", 0)
        self.grad_steps = ckpt.get("grad_steps", 0)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.eps_decay_episodes = ckpt.get("eps_decay_episodes", self.eps_decay_episodes)
