# Reinforcement-learning agents for 2048: deep nets vs. n-tuple TD (temporal-difference learning)

**Florian Robrecht, Janin Jankovski, Anna Hartmann — Machine Learning 2, Assignment 5**

---

## 1. The game and the question

**2048** is a single-player tile-sliding puzzle on a 4×4 grid. The rules fit on a postcard:

- Every cell is either empty or holds a tile with a value that is a power of two (2, 4, 8, 16, 32, …).
- On each turn you swipe in one of four directions: **Up, Down, Left, Right**. All tiles slide as far as they can in that direction.
- When two tiles with the **same number** collide, they merge into one tile of **double** the value. You score points equal to the new tile.
- After every swipe a new tile (a 2, or rarely a 4) appears in a random empty cell.
- You **win** if you ever build the 2048 tile. You **lose** when the board fills up and no merge is possible.

Random play almost never gets past the 256 tile. Good players keep the biggest tile pinned in a corner and build a "snake" of decreasing tiles next to it. The strategy looks simple and is in fact quite subtle.

2048 has a long history with classical machine learning. The strongest non-search agents on record use a method from 2014 called **n-tuple temporal-difference learning** (Szubert & Jaśkowski, *CIG 2014*). That method predates the deep-RL boom by years and uses no neural network at all.

The obvious 2026 move is to test whether a deep neural network can beat the n-tuple method. So our question is:

> Given the same compute and the same game interface, can a generic **deep Q-network (DQN)** with a one-hot board encoding match the classical **n-tuple** agent on 2048?

---

## 2. What "reinforcement learning" means here

Reinforcement learning (RL) is the branch of machine learning where an **agent** learns by **trial and error**:

1. It looks at the current **state** (the board).
2. It picks an **action** (a swipe direction).
3. It gets a **reward** (the points scored on that swipe).
4. It observes the new state and repeats.

After many games, the agent should learn which swipe is good in which situation. The technical question is: **how should the agent remember "good moves in good situations" when no two boards are ever quite the same?**

A 2048 board can hold any of ~16 values in 16 cells, giving roughly **16¹⁶ ≈ 18 quintillion** possible boards. So nobody can just memorise. Each method has to define its own way of saying *"boards that look similar in this way should be treated similarly"*, and the two methods we compare answer that question very differently.

Both methods use the same family of update rule, called **temporal-difference (TD) learning**: after each move, the method nudges its own predictions to be more consistent with what actually happened. The "teacher" is the method's *own* prediction one step into the future, plus the real reward it just observed. That's what lets RL work without anyone labelling.

---

## 3. The four players

We build and compare four agents, each implementing the same `act(board, legal_mask) → swipe` interface so the evaluation harness can swap them freely.

### Random
Picks a legal swipe uniformly at random. Anything worse than random is broken. 

### Greedy heuristic
30 lines of hand-coded rules: for each candidate swipe, score the resulting board with **monotonicity** (rewards rows/columns sorted in one direction), **empty cells** (more empties = more room), and a **corner bonus** (extra points if the biggest tile is in a corner). Pick the swipe with the highest score. No learning at all, only human-domain-knowledge baseline.

### DQN (Deep Q-Network)
The "modern deep RL" approach, popularised by DeepMind's 2013–2015 Atari work. A neural network reads the board and outputs four numbers, one per swipe direction. Each number — a **Q-value** — is the network's prediction of "if I pick this swipe now and play well from here, how many points will I score in total before the game ends?" To play a move, pick the swipe with the highest Q-value.

### N-tuple TD
The classical approach. Instead of a neural network, the agent uses a handful of **lookup tables** indexed by small patches of the board. The value of the whole board is just the sum of a few table reads. The "intelligence" comes from picking the patches well — and the agent then learns the table contents from playing the game.

The rest of this writeup explains how DQN and n-tuple actually function, then shows the results.

---

## 4. DQN (Deep Q-Network)

### Network architecture
A small fully-connected network:
- **Input**: 256 floats. The 4×4 board is encoded as a one-hot tensor — each cell becomes a 16-element vector indicating which power of 2 it holds (or that it's empty) — and flattened to a 256-vector. The network sees no domain-specific structure; it processes 256 numbers.
- **Hidden layers**: 256 → 512 → 256, ReLU activations.
- **Output**: 4 Q-values, one per swipe direction.

The network is twinned: an **online network** that is updated every gradient step, and a **target network** with the same architecture but periodically refreshed weights, used to compute training targets (see below).

### One training step
1. **Collect a transition.** The agent observes board `s`, picks swipe `a`, the environment returns reward `r` and the next board `s'`. The 5-tuple `(s, a, r, s', done)` is appended to a **replay buffer** — a fixed-size memory of recent transitions, implemented as a *ring buffer* (a circular queue: when full, the oldest entry is overwritten by the newest). We use a capacity of 100,000 transitions.
2. **Sample a mini-batch** of 512 random transitions from the buffer. Sampling from a large pool of past experience decorrelates consecutive training samples and is far more stable than training only on the current game's trajectory.
3. **Current prediction.** Compute `Q(s, a)` — the Q-value the online network currently assigns to the action that was actually taken.
4. **TD target.** Compute the value the network *should* have predicted:
   ```
   target = r + γ · max over a' of Q_target(s', a')
   ```
   In words: "the reward just received, plus the discounted best Q-value attainable from the next state." The **discount factor γ ∈ [0, 1]** controls how much the agent values future rewards relative to immediate ones. We use γ = 0.99, the standard choice for episodic tasks of this length: a reward received 100 steps from now is worth `0.99¹⁰⁰ ≈ 0.37` of the same reward received immediately. Lower values make the agent myopic and unable to learn long-horizon plans (e.g. building a snake of tiles for later merges); values too close to 1 destabilise the bootstrap by letting small estimation errors compound over a long horizon.
5. **Gradient step.** Compute the loss (smooth-L1 between `Q(s, a)` and `target`), backpropagate, take an Adam step (learning rate 5e-4).

Across 5,000 games of training, steps 1–5 repeat roughly one million times.

### Training stabilisation mechanisms
Basic DQN is unstable for three reasons: (i) the network bootstraps off its own predictions, so the target moves with every gradient step; (ii) the `max` in the target overestimates Q-values, since the maximum of noisy estimates is biased upward; (iii) rewards span several orders of magnitude, so Q-values do too. We apply the standard fix for each, plus two further mechanisms that address training behaviour rather than numerical stability.

**Target network** (fixes i). Keep a frozen snapshot `Q_target` of the online network and compute the TD target from it. Hard-sync the snapshot every 1,000 gradient steps. The target stays stationary between syncs while the online network catches up.

**Double DQN** (fixes ii). Decouple action *selection* from action *evaluation* in step 4: the **online** network picks the best next action, the **target** network reports its Q-value. The biases of the two largely cancel, producing more accurate targets.

**Reward scaling** (fixes iii). Multiply rewards by 0.1 before storing them in the buffer. Rescales the loss landscape but does not change the optimal policy. (Merging two 2s gives reward 2 (since log₂(4)=2). Merging two 1024s gives reward 11. With scaling rewards are between 0.2 and 1.5.)

**ε-greedy exploration.** With probability ε the agent picks a uniformly random *legal* swipe instead of the greedy one; otherwise the highest-Q action. ε decays linearly from 1.0 to 0.05 over the first 70 % of training. Broad exploration early — when Q-values are essentially random — gives way to exploitation later, once the network has something worth following.

**Action masking.** Typically only 2–3 of the 4 swipes are legal. Illegal Q-values are clamped to −1e9 in both action selection *and* the target bootstrap, so illegal moves are never picked and never appear in the value estimate. We deliberately do **not** penalise illegal moves with negative rewards: a reward-based penalty would leak into the value function and inflate its variance.

---

## 5. N-tuple TD (temporal-difference) learning

### The trick: don't look at the whole board, look at little patches
Where DQN tries to learn a function of the entire 256-dim board, the n-tuple agent says: **decompose the board into a few small patches; if two boards have the same patch contents, treat them similarly there.** It then sums contributions from several patches to value the whole board.

We use four patches (the same set as Szubert & Jaśkowski 2014):

| Pattern | Cells | Shape |
| --- | --- | --- |
| **A** (6 cells) | (0,0)(0,1)(0,2)(0,3)(1,0)(1,1) | "Axe" — top row plus the leftmost two cells of row 1 |
| **B** (6 cells) | (1,0)(1,1)(1,2)(1,3)(2,0)(2,1) | Same axe shape, shifted down one row |
| **C** (4 cells) | (0,0)(0,1)(0,2)(0,3) | The top row |
| **D** (4 cells) | (0,0)(0,1)(1,0)(1,1) | The top-left 2×2 corner |

Each cell can hold one of ~16 log₂ values, so a 6-cell pattern has 16⁶ ≈ 16 million possible contents, and each pattern gets its own lookup table sized accordingly. Big by classical-ML standards, tiny by deep-learning ones — and crucially, **finite and enumerable**.

### Computing the value of a board
```
V(board) = lookup_A[contents of cells in pattern A]
         + lookup_B[contents of cells in pattern B]
         + lookup_C[contents of cells in pattern C]
         + lookup_D[contents of cells in pattern D]
         + (the same thing for every rotated/reflected placement of each pattern — see below)
```

No neural network. The whole value function is four lines of code ([`ntuple.py:72`](../src/agents/ntuple.py#L72)).

### Dihedral symmetry sharing
A 2048 strategy that works in the top-left corner is, by symmetry, equally valid in the other three corners (mirrored or rotated). The square has 8 symmetries (4 rotations × 2 reflections). We exploit this by sharing weights: each pattern is looked up not just at its canonical position but at every dihedral image of itself, and they all hit the *same* weight table. One training update teaches all symmetric variants at once.

Not every pattern has 8 distinct dihedral images, though. Pattern C (the top row) is itself symmetric under the left-right flip, so it only has 4 distinct placements (the 4 rows/columns). Pattern D (the 2×2 corner) is symmetric under the main-diagonal flip, so it also has 4 (the 4 corners). Patterns A and B have no such internal symmetry, so they have 8 each. Total lookups per board: **8 + 8 + 4 + 4 = 24** ([`ntuple.py`](../src/agents/ntuple.py) computes this dynamically as `num_lookups`).

### Afterstate values
A 2048 transition has two halves: (a) the **deterministic** slide-and-merge that the agent controls, and (b) the **random** tile spawn that follows. The board between these two halves is the **afterstate**. We learn the value of *afterstates*, not states — that way the random spawn never appears inside the bootstrap target, and the spawn randomness is averaged out empirically over the many games we play.

### One training step
1. **Pick the move greedily.** From the current board `s_t`, try each legal swipe; score the resulting afterstate by `r + V(after)`; keep the highest. This gives the action `a_t`, its immediate reward `r_t`, and the afterstate `after_t`.
2. **Apply the move.** The environment performs the slide-and-merge (producing `after_t`) and spawns a random tile, giving the next board `s_{t+1}`.
3. **Pick the next move the same way** on `s_{t+1}`, producing `r_{t+1}` and `after_{t+1}`.
4. **TD update.** Nudge `V(after_t)` toward the target `r_{t+1} + V(after_{t+1})`:
   ```
   V(after_t)  ←  V(after_t)  +  α · (r_{t+1}  +  V(after_{t+1})  −  V(after_t))
   ```
   Concretely, every one of the 24 weights that contributed to `V(after_t)` gets `α · δ` added to it, where `δ` is the bracketed TD error. We set the board-level learning rate `α_base = 0.1` and divide by `num_lookups`, so each weight gains `≈ 0.00417 · δ` per update — keeping the *board-level* change in `V` at the chosen `α_base · δ`.

Across 10,000 games of training, this loop runs roughly a million times. No replay buffer, no gradient descent, no batching — each update touches 24 numbers and runs in microseconds, which is why the n-tuple trains so much faster than DQN per step.

---

## 6. One last technical note: the reward unit

We feed the agents a **simplified reward**: on each merge, the reward is the **log₂ of the new tile**, not the new tile itself.

- Two 8s merging into a 16: reward 4 (because log₂(16)=4), not 16.
- Two 1024s merging into a 2048: reward 11, not 2048.

We do this because the standard 2048 score grows exponentially through a game, which is hard for both methods to fit a smooth value function to. The log₂ reward keeps magnitudes small and approximately uniform across the early and late game.

The practical consequence: every score in our results table is roughly **5× smaller** than what a leaderboard would report for the same game. The **max-tile distribution is unit-independent**, so the percentages are directly comparable to anything in the literature.

---

## 7. Results

DQN was trained for **5,000 episodes** (~25 minutes on Apple-MPS; Double-DQN, reward scaling, 70 %-of-training ε decay). The n-tuple was trained for **10,000 episodes** (~55 minutes). Random and greedy don't train — they're evaluated directly.

Evaluation: random and greedy on **1,000 games each**, DQN and n-tuple on **300 games each** (the n-tuple eval is slower because its games are longer — it survives longer). All test seeds disjoint from training. The agents play greedily during evaluation, no exploration noise.

| Agent | Mean reward | Median | Mean max tile | P(≥512) | P(≥1024) | P(≥2048) | P(≥4096) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random            |   298 |   291 |   108 |   0% |   0% |   0% |   0% |
| greedy heuristic  | 1 300 | 1 266 |   468 |  65% |  11% |   0% |   0% |
| DQN (5 k ep)      |   311 |   296 |   109 |   0% |   0% |   0% |   0% |
| **N-tuple (10 k ep)** | **3 270** | **3 178** | **1 222** |  **98%** |  **87%** |  **25%** | **0.7%** |

(Reminder: "mean reward" is in log₂-merge units. Full numbers in [`data/eval/comparison.csv`](../data/eval/comparison.csv).)

First 1,024 tile during n-tuple training: **episode 307**. First 2,048 tile: **episode 1,231**.

Three things to read from this table:

1. **N-tuple wins decisively.** It hits the win-condition tile (2,048) in *one quarter of games* and the 1,024 tile in 87 % — well past the greedy heuristic (which never reaches 2,048) and an order of magnitude above DQN on mean reward (3,270 vs 311).
2. **DQN is essentially random.** After 5,000 episodes of Double-DQN with reward scaling and ε decay, mean reward is 311 vs 298 for the uniform-random policy. The max-tile distribution is also indistinguishable from random: never reaches 512.
3. **Greedy beats DQN by 4×.** The hand-coded heuristic with 1-step lookahead over monotonicity + empty-cells + corner reward outperforms the trained DQN on every metric. A 30-line rule-based policy beats vanilla deep RL with one-hot input on this game.

The training curves ([`data/eval/fig_training_curves.png`](../data/eval/fig_training_curves.png), [`data/eval/training_dashboard.png`](../data/eval/training_dashboard.png)) show this concretely: the n-tuple's rolling-100 return climbs steadily from ~500 to ~3,000 over 10 k episodes; DQN's rolling-100 return rises from ~100 to ~300 over 5 k episodes and then flatlines, while smooth-L1 loss keeps growing as the value head fits the spread of returns it sees but cannot turn into a useful policy.

---

## 8. Why does the linear method win?

Three reasons, all about what the agent gets *for free* before it ever starts learning.

**The n-tuple is already looking at the right thing.** Its 6-cell axe pattern covers exactly the kind of arrangement a good 2048 player thinks about — which tiles are next to which, whether the top row is sorted, what's in the corner. DQN sees 256 raw numbers and has no clue any of that matters. To play as well as the n-tuple, it would have to figure out from scratch that neighbouring cells interact, that corners are special, and that a sorted row is worth more than a jumbled one. With 5,000 games of practice, it doesn't get there.

**The n-tuple gets symmetry for free.** A clever move in the top-left corner is also a clever move in the other three corners, just rotated or mirrored. The n-tuple knows this by construction: an update in one corner automatically applies to the other three. DQN has to relearn each corner from scratch, costing it roughly 4–8× as much experience to reach the same understanding.

**The n-tuple's learning signal has less noise.** It learns the value of *afterstates* — the board after your swipe but before the random tile spawn — so the random spawn never enters its target. DQN learns the value of post-spawn boards, so every target it tries to match includes noise from a tile placement it couldn't have controlled. Same data, noisier learning signal, slower convergence.

The big-picture lesson: **when humans already know what features to look at, baking those features in beats throwing a generic deep net at the raw board — at least at the compute budgets used here.** Deep RL *can* win on 2048, but it needs a much better architecture (a convolutional network, more sophisticated training tricks) and a lot more training. Our experiment lands firmly on the "the right representation wins" side of that trade-off.

---

## 9. Limitations & next steps

- **DQN's training budget was 5 k episodes**, half the n-tuple's. The literature places vanilla DQN with one-hot board input well below the n-tuple even at 10× the budget, but the gap reported here is a lower bound, not the asymptote.
- **DQN architecture is intentionally simple.** It includes Double-Q and reward scaling, but no duelling head, no prioritised replay, no n-step returns, and no convolutional encoder. A CNN over the 4×4×16 one-hot tensor would let the network exploit the same translation structure the n-tuple gets via dihedral symmetry; CIG 2017+ papers report CNN-DQN agents matching or beating S&J.
- **DQN over n-tuple features** — out of scope here — would directly test whether the representation gap explains the result. That's the natural follow-up.
- **Mixed evaluation sample sizes**: DQN and n-tuple are evaluated on 300 games, baselines on 1,000. The confidence intervals on rare-tile probabilities are wider for the 300-game test sets, but the gaps in the table are large enough that no realistic re-evaluation would change the qualitative ranking.

---

## 10. Project structure

If you want to read or modify the code, here's the lay of the land. Everything below is relative to `Assignment 5/`.

### Notebooks (the user-facing entry points)
```
notebooks/
├── 01_environment_and_baselines.ipynb    ← env tests + random & greedy baseline eval
├── 02_dqn.ipynb                          ← train DQN, save checkpoint, evaluate
├── 03_ntuple_td.ipynb                    ← train n-tuple, save checkpoint, evaluate
└── 04_comparison_and_viewer.ipynb        ← build comparison CSV + figures, render replay GIFs
```
Run them in order. Each one writes its outputs to `data/` so the next can pick them up.

### Source code
```
src/
├── env.py             ← Game2048Env + VectorGame2048Env (the 2048 simulator)
├── moves.py           ← bitboard move LUTs — the fast slide/merge core
├── encoding.py        ← one-hot encoding (for DQN) and packed indices (for n-tuple)
├── utils.py           ← paths, RNG seeding, torch device picker (MPS > CUDA > CPU)
│
├── agents/            ← the four players, all sharing the Agent interface
│   ├── base.py        ← abstract Agent class — every agent implements act(board, mask)
│   ├── random_agent.py
│   ├── greedy_agent.py
│   ├── dqn.py         ← QNetwork + ReplayBuffer + DQNAgent (training & inference)
│   └── ntuple.py      ← NTupleNetwork + NTupleAgent (afterstate TD)
│
├── training/          ← the training loops
│   ├── train_dqn.py
│   ├── train_ntuple.py
│   └── logger.py      ← append-only CSV logger with a fixed schema
│
├── eval/              ← evaluation and reporting
│   ├── evaluate.py    ← run an agent for n_games, write eval_<agent>.json
│   └── report.py      ← build comparison.csv and the figures used in this writeup
│
├── viz/               ← visualisation helpers
│   ├── dashboard.py   ← training-dashboard.png panel layout
│   └── viewer.py      ← per-step board renderer for replay GIFs
│
└── tests/
    └── test_env.py    ← 8 unit tests: LUT correctness, spawn distribution, legality, symmetries
```

### Generated artefacts
Everything the notebooks produce lives under `data/`:
```
data/
├── checkpoints/        ← trained model weights
│   ├── dqn/            ← latest.pt, best.pt (PyTorch state dicts)
│   └── ntuple/         ← latest.npz, best.npz (compressed weight tables)
│
├── logs/               ← training logs, one CSV per agent
│   ├── dqn.csv         ← per-episode return, max_tile, loss, ε, wallclock
│   └── ntuple.csv      ← same schema, but with α and TD-error
│
├── eval/               ← evaluation results + headline figures
│   ├── eval_<agent>.json  ← per-agent metrics (mean score, max-tile distribution, etc.)
│   ├── comparison.csv     ← all four agents side-by-side
│   ├── fig_scores.png     ← bar chart of mean reward per agent
│   ├── fig_tile_reach.png ← P(reach tile T) per agent
│   ├── fig_training_curves.png  ← rolling-100 return over training
│   ├── tile_distribution.png    ← histogram of max-tile per agent
│   └── training_dashboard.png   ← multi-panel training overview
│
└── gifs/               ← replay animations
    ├── compare_dqn_vs_ntuple.gif
    └── ntuple_42.gif
```

### Top-level
```
Assignment 5/
├── writeup/analysis.md  ← this document
├── conftest.py          ← pytest setup (puts src/ on the import path)
├── .gitignore
└── .python-version
```

### How to find your way around for common tasks
- **"Where does the game logic live?"** → [`src/moves.py`](../src/moves.py) (the LUTs) and [`src/env.py`](../src/env.py) (the wrapper).
- **"Where's the DQN architecture?"** → `QNetwork` at [`src/agents/dqn.py:20`](../src/agents/dqn.py#L20). The training loop is [`src/training/train_dqn.py`](../src/training/train_dqn.py).
- **"Where are the n-tuple patterns defined?"** → top of [`src/agents/ntuple.py`](../src/agents/ntuple.py) (the `_PATTERN_*` constants). The 4-line value function is `NTupleNetwork.value`.
- **"Where are the result numbers from?"** → JSON files in `data/eval/`, generated by [`src/eval/evaluate.py`](../src/eval/evaluate.py), aggregated into `comparison.csv` by [`src/eval/report.py`](../src/eval/report.py).
- **"Where do I change a hyperparameter?"** → for DQN, the `DQNAgent(...)` constructor call in [`notebooks/02_dqn.ipynb`](../notebooks/02_dqn.ipynb). For n-tuple, the `NTupleAgent(alpha=...)` call in [`notebooks/03_ntuple_td.ipynb`](../notebooks/03_ntuple_td.ipynb). The defaults are in the agent class itself.

---

## 11. Reproducing the figures

```
cd "Assignment 5"
uv run pytest src/tests/                         # 8 env tests, all pass
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/01_environment_and_baselines.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/02_dqn.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/03_ntuple_td.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04_comparison_and_viewer.ipynb
```

All artefacts land under `data/`: `checkpoints/{dqn,ntuple}/{latest,best}.{pt,npz}`, per-agent `eval/eval_<agent>.json`, and the comparison figures shown above.
