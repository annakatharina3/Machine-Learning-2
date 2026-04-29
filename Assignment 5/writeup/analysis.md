# Reinforcement-learning agents for 2048: deep nets vs. n-tuple TD

**Florian Robrecht — Machine Learning 2, Assignment 5**

## 1. Question and setup

The classical record-holders on the 4×4 puzzle game **2048** are linear value
functions over hand-designed *n-tuple patterns* trained with temporal-difference
learning (Szubert & Jaśkowski, *CIG 2014*). Modern deep-RL methods such as DQN
have displaced these classical approaches in many control problems, so a natural
question is: **does a generic Q-network with a one-hot board encoding match the
n-tuple TD agent on 2048, given the same compute and the same legal-move
interface?**

To answer this I implemented:

* a fast bitboard 2048 environment (`Game2048Env`, `VectorGame2048Env`) with
  precomputed row-shift LUTs and unit tests over all 65 536 packed rows,
* two baselines: a uniform-random policy and a greedy heuristic (1-step
  monotonicity + empty-cell + corner reward),
* a **DQN** with one-hot log2 board input (16 cells × 16 buckets = 256 floats),
  three-layer MLP (256 → 512 → 256 → 4), Adam (lr=5e-4), γ=0.99, smooth-L1
  loss, target-net hard-sync every 1 000 grad steps, ε linear from 1.0 → 0.05
  over 50 k episodes, and **action masking** (illegal Q-values clamped to
  −1e9 in *both* online action selection and the target bootstrap),
* an **N-tuple** value function with the S&J 2014 pattern set — two 6-tuples
  (axe shapes covering the top two rows) and two 4-tuples (top row + 2×2
  corner square) — each augmented with the 8-element dihedral symmetry group
  so weights are shared. Training is **TD(0) on afterstates**: the
  deterministic outcome of a move *before* the random spawn, which removes the
  spawn variance from the bootstrap target.

All four agents satisfy the same `Agent.act(board, legal_mask, greedy)`
interface, so the rollouts that produce the comparison numbers are identical
modulo the action choice. Code, notebooks, training logs, checkpoints and
generated figures all live under `Assignment 5/`.

## 2. Method choices, with reasons

A few small decisions matter enough to call out:

* **Action masking, not penalty rewards.** Penalising illegal moves leaks into
  the value function and inflates variance. With masking, illegal Q-values are
  always −1e9 in both the argmax over the online net and the max over the
  target net, so the agent never bootstraps from an action it cannot take.
* **DQN sees a one-hot board, not n-tuple features.** Giving DQN the same
  hand-engineered features the n-tuple is given would defeat the experiment.
  The point is to ask whether the network can *learn* representations that
  rival the hand-engineered ones — so it gets a generic encoding and is
  expected to discover its own.
* **N-tuple updates on afterstates.** The 2048 transition is
  `state → afterstate → next state` (deterministic move, then random spawn).
  Bootstrapping from `V(afterstate)` rather than `V(state)` cuts the spawn
  variance out of the TD target. This is exactly the S&J 2014 setup.
* **Effective per-weight α.** The total update to `V(board)` is
  `α · num_lookups · δ`, where `num_lookups ≈ 32` (4 patterns × ~8 dihedral
  images). Setting `α_base = 0.1` and `α_per_weight = α_base / num_lookups`
  keeps the *board-level* update at roughly the chosen scale.

## 3. Results

Numbers below come from the runs shipped in the notebooks: DQN trained 5 000
episodes (~25 min on MPS, with Double-DQN + reward scaling), N-tuple trained
10 000 episodes (~55 min). Each agent is evaluated greedily on a fresh 300- or
1 000-game test set with seeds disjoint from training.

| Agent | Mean score | Median | Mean max tile | P(≥512) | P(≥1024) | P(≥2048) | P(≥4096) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random            |   298 |   291 |   108 |   0% |   0% |   0% |   0% |
| greedy heuristic  | 1 300 | 1 266 |   468 |  65% |  11% |   0% |   0% |
| DQN (5 k ep)      |   311 |   296 |   109 |   0% |   0% |   0% |   0% |
| **N-tuple (10 k ep)** | **3 270** | **3 178** | **1 222** |  **98%** |  **87%** |  **25%** | **0.7%** |

The full table is in `data/eval/comparison.csv`. First 1 024 tile during
N-tuple training: **episode 307**. First 2 048 tile: **episode 1 231**.

Three things to read from this table:

1. **N-tuple wins decisively.** It hits the win-condition tile (2 048) in
   *one quarter of games* and the 1 024 tile in 87 % — well past the greedy
   heuristic (which never reaches 2 048) and an order of magnitude above DQN
   on mean score (3 270 vs 311).
2. **DQN is essentially random.** After 5 000 episodes of Double-DQN with
   reward scaling and a 70 %-of-training ε-decay, mean score is 311 vs 298 for
   the uniform-random policy. Max tile distribution is also indistinguishable
   from random: never reaches 512.
3. **Greedy beats DQN by 4×.** The hand-coded heuristic with 1-step lookahead
   over monotonicity + empty cells + corner reward outperforms the trained
   DQN on every metric. This is the same observation as in (1) but worth
   stating directly: a 30-line rule-based policy beats vanilla deep RL with
   one-hot input on this game.

The training curves (`data/eval/fig_training_curves.png`,
`data/eval/training_dashboard.png`) show this concretely: the N-tuple's
rolling-100 return climbs steadily from ~500 to ~3 000 over 10 k episodes; the
DQN's rolling-100 return rises from ~100 to ~300 over 5 k episodes and then
flatlines, with smooth-L1 loss continuing to grow as the value head fits the
spread of returns it sees but cannot extract a useful policy.

## 4. Why does the linear method win?

Two structural reasons.

**Representation matters more than capacity here.** The n-tuple network is a
*linear* function approximator over indicator features, but those features are
*the right ones*: an axe-shaped 6-tuple captures interactions between
neighbouring tiles in a corner region — exactly the structure a strong human
2048 player exploits. A 256-dim one-hot input gives the DQN no such
inductive bias. To learn the same patterns from scratch the network would have
to discover, in its hidden layers, that proximity matters, that monotonicity
matters, and that the four corners are equivalent under reflection. With
typical compute it does not get that far.

**Symmetry is encoded explicitly.** Each pattern is shared across the 8-element
dihedral group, so a single weight update propagates to all symmetric board
configurations. DQN has to relearn each symmetric variant separately. This
roughly multiplies effective sample efficiency by 8.

**Variance hygiene.** Operating on afterstates avoids the spawn noise in the
bootstrap target. DQN bootstraps off the post-spawn state and cannot easily
remove that noise. With smooth-L1 loss the impact is bounded but the slower
convergence is real.

## 5. Limitations & next steps

* DQN was trained for 5 k episodes; the reference S&J n-tuple was trained
  for 10 k. Even at 10× the budget the literature places vanilla DQN with
  one-hot board inputs well below the n-tuple, but the gap reported here
  is the lower bound, not the asymptote.
* The DQN does include Double-Q and reward scaling, but no dueling head,
  no prioritised replay, no n-step returns and no convolutional encoder.
  A CNN over the 4×4×16 one-hot tensor would let the network exploit the
  same translation structure the n-tuple gets via dihedral symmetry; the
  CIG 2017+ literature reports CNN-DQN agents matching or beating S&J.
* The stretch — DQN over n-tuple features (the hybrid the user flagged
  out-of-scope) — would directly test whether the representation gap
  explains the result. That is the natural follow-up.

## 6. Reproducing the figures

```
cd "Assignment 5"
uv run pytest src/tests/                         # 8 env tests, all pass
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/01_environment_and_baselines.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/02_dqn.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/03_ntuple_td.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04_comparison_and_viewer.ipynb
```

All artefacts land under `data/`: `checkpoints/{dqn,ntuple}/{latest,best}.{pt,npz}`,
per-agent `eval/eval_<agent>.json` and the comparison figures shown above.
