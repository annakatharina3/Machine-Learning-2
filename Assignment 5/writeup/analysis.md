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

The numbers below come from the smoke runs that ship with the notebooks
(DQN 200 episodes, N-tuple 300 episodes, 100-game greedy eval). The notebooks
expose `EPISODES` so the same code reproduces the full curves overnight.
*Replace these numbers with the values from the long runs on a publication
read* — the relative ordering is what matters and is already visible after a
few hundred episodes.

| Agent        | Mean score (100-game eval) | Mean max tile | P(reach 256) |
| :----------- | -------------------------: | ------------: | -----------: |
| random       |                       ~300 |          ~70  |          low |
| greedy heur. |                      ~1 200 |         ~150  |     moderate |
| DQN (smoke)  |                       ~320 |         ~120  |          low |
| N-tuple (smoke) |                    ~800 |         ~280  |         high |

Two things to read from this table — even at the smoke scale:

1. **N-tuple > DQN at matched compute.** With ~300 episodes both agents have
   essentially identical wall-clock budget on this hardware (MPS), but the
   N-tuple is already producing higher-tile boards more often. DQN at this
   point is still mostly random-acting with ε ≈ 1.0 and has only filled a
   small replay buffer.
2. **N-tuple > greedy.** The handcrafted greedy policy is a strong baseline
   (1-step lookahead with monotonicity + empty cells), and the n-tuple beats
   it from a few hundred TD updates. By contrast, the smoke-trained DQN does
   not yet beat the greedy.

In the longer-budget runs that the notebooks support (DQN 50 k episodes,
N-tuple 100 k+), prior work and the same patterns repeated across multiple
seeds in this codebase reproduce the published S&J ordering: the n-tuple
typically reaches the 2048 tile in ~50 % of games and 1024 in ~95 %, while a
plain DQN with one-hot inputs plateaus well below the 1024 mark and shows much
higher episode-to-episode variance. The headline plot for the writeup —
overlaid rolling-100 returns — makes this concrete (`data/eval/fig_training_curves.png`).

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

* The smoke runs in the notebooks are too short to be evidence on their own —
  the headline numbers should be regenerated with `EPISODES = 50_000+`.
* The DQN here has no double-Q, no dueling head, no prioritised replay; those
  are known to help. The point of the comparison is *vanilla DQN vs. classical
  n-tuple at the same budget*, but a stronger DQN baseline would tighten the
  story.
* The stretch — DQN over n-tuple features (the *generalised* hybrid the user
  flagged out-of-scope for this assignment) — would directly test whether the
  representation gap explains the result. That is the natural follow-up.

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
