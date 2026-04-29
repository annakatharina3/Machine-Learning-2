"""Unit tests for env, moves, spawn distribution, legality, rewards."""
import numpy as np
import pytest

from src.env import Game2048Env
from src.moves import (LEFT_LUT, REWARD_LUT, apply_move,
                        _shift_merge_left, _row_to_int, _int_to_row,
                        ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN)


def _naive_left(row):
    nz = [v for v in row if v > 0]
    out, rew, i = [], 0, 0
    while i < len(nz):
        if i + 1 < len(nz) and nz[i] == nz[i + 1]:
            v = min(nz[i] + 1, 15)
            out.append(v)
            rew += v
            i += 2
        else:
            out.append(nz[i])
            i += 1
    out.extend([0] * (4 - len(out)))
    return out, rew


def test_left_lut_matches_naive_for_all_rows():
    for r in range(65536):
        row = list(_int_to_row(r))
        new, rew = _naive_left(row)
        packed = (new[0] << 12) | (new[1] << 8) | (new[2] << 4) | new[3]
        assert int(LEFT_LUT[r]) == packed, f"LUT mismatch at {r}"
        assert int(REWARD_LUT[r]) == rew


def test_spawn_distribution_chi2():
    env = Game2048Env(seed=0)
    env.reset(seed=0)
    twos = fours = 0
    for _ in range(100_000):
        env.board[:] = 0
        env._spawn()
        v = env.board[env.board > 0][0]
        if v == 1:
            twos += 1
        elif v == 2:
            fours += 1
    n = twos + fours
    p_two = twos / n
    # 99.99% CI for binomial sample at n=100k: ~4 sigma ~ 0.0038
    assert abs(p_two - 0.9) < 0.005, f"got p(2)={p_two:.4f}"


def test_legal_mask_all_filled_no_merges():
    env = Game2048Env(seed=0)
    env.board = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ], dtype=np.int8)
    env.score = 0
    env.steps = 0
    env.done = False
    assert not env.legal_actions().any()


def test_legal_mask_with_one_empty():
    env = Game2048Env(seed=0)
    env.board = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 0],
    ], dtype=np.int8)
    env.score = 0
    env.steps = 0
    env.done = False
    mask = env.legal_actions()
    # at least UP and LEFT can move tiles into the empty corner
    assert mask.any()


def test_reward_equals_log2_sum_of_merges():
    board = np.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    new, r, ch = apply_move(board, ACTION_LEFT)
    assert ch and r == 2 and new[0, 0] == 2

    board = np.array([
        [2, 2, 3, 3],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    new, r, ch = apply_move(board, ACTION_LEFT)
    assert r == 7 and new[0, 0] == 3 and new[0, 1] == 4


def test_left_right_symmetry():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = rng.integers(0, 6, size=(4, 4)).astype(np.int8)
        nb_l, r_l, _ = apply_move(b, ACTION_LEFT)
        nb_r, r_r, _ = apply_move(b[:, ::-1].copy(), ACTION_RIGHT)
        assert np.array_equal(nb_l, nb_r[:, ::-1])
        assert r_l == r_r


def test_up_down_symmetry():
    rng = np.random.default_rng(1)
    for _ in range(50):
        b = rng.integers(0, 6, size=(4, 4)).astype(np.int8)
        nb_u, r_u, _ = apply_move(b, ACTION_UP)
        nb_d, r_d, _ = apply_move(b[::-1].copy(), ACTION_DOWN)
        assert np.array_equal(nb_u, nb_d[::-1])
        assert r_u == r_d


def test_episode_terminates():
    env = Game2048Env(seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    while not env.done and steps < 10000:
        legal = env.legal_actions()
        if not legal.any():
            break
        a = int(rng.choice(np.flatnonzero(legal)))
        env.step(a)
        steps += 1
    assert env.done, "random play should terminate within 10k steps"
