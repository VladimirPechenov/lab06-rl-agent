import gymnasium as gym
import numpy as np
import pytest

from lab06.evaluate import rollout


class ConstantPolicy:
    def predict(self, obs, deterministic):
        assert deterministic
        return np.array(0), None


def test_seeded_rollout_and_termination():
    with gym.make("CartPole-v1") as env:
        first = rollout(ConstantPolicy(), env, 3, 2000)
        second = rollout(ConstantPolicy(), env, 3, 2000)
    assert first == second
    assert [r["seed"] for r in first] == [2000, 2001, 2002]
    assert all(r["reward"] == r["length"] and r["terminated"] for r in first)


def test_time_limit_is_respected():
    with gym.make("CartPole-v1", max_episode_steps=2) as env:
        rows = rollout(ConstantPolicy(), env, 1, 0)
    assert rows[0]["length"] == 2
    assert rows[0]["truncated"]


def test_empty_evaluation_is_rejected():
    with gym.make("CartPole-v1") as env, pytest.raises(ValueError):
        rollout(ConstantPolicy(), env, 0, 0)
