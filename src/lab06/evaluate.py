"""Evaluate final policies on held-out, identical seeded episodes."""

import argparse
import hashlib
import json

import gymnasium as gym
import numpy as np
import torch

from .common import ALGOS, MODELS, RESULTS


def rollout(model, env, n, seed):
    if n < 1:
        raise ValueError("episodes must be positive")
    episodes = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        total, length = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
            total += float(reward)
            length += 1
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed + i,
                "reward": total,
                "length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
    return episodes


def evaluate(algo: str, env_id: str = "CartPole-v1", n: int = 20, seed: int = 2000) -> dict:
    torch.set_num_threads(1)
    path = MODELS / f"{algo}.zip"
    model = ALGOS[algo].load(path, device="cpu")
    env = gym.make(env_id)
    try:
        episodes = rollout(model, env, n, seed)
    finally:
        env.close()
    rewards = [e["reward"] for e in episodes]
    threshold = 450 if env_id == "CartPole-v1" else 200 if env_id == "LunarLander-v3" else None
    out = {
        "algo": algo,
        "env": env_id,
        "episodes": n,
        "seed_start": seed,
        "deterministic": True,
        "checkpoint": "final",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "threshold": threshold,
        "passed": threshold is not None and float(np.mean(rewards)) >= threshold,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episode_results": episodes,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{algo}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{algo}: {out['mean_reward']:.2f} +/- {out['std_reward']:.2f}, passed={out['passed']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2000)
    a = ap.parse_args()
    evaluate(a.algo, a.env, a.episodes, a.seed)
