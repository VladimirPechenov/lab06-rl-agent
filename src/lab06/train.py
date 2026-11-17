"""Real SB3 training with independent validation and portable raw logs."""

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .common import ALGOS, MODELS, RESULTS, RUNS, load_cfg
from .evaluate import rollout


class ValidationCallback(BaseCallback):
    def __init__(self, cfg, folder):
        super().__init__()
        self.cfg, self.folder = cfg, folder
        self.rows = []
        self.last_step = -1

    def measure(self):
        step = self.model.num_timesteps
        if step == self.last_step:
            return
        env = gym.make(self.cfg["env"])
        try:
            episodes = rollout(self.model, env, self.cfg["validation_episodes"], self.cfg["validation_seed"])
        finally:
            env.close()
        rewards = [e["reward"] for e in episodes]
        row = {
            "timesteps": step,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "rewards": rewards,
        }
        self.rows.append(row)
        (self.folder / "validation.json").write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        self.logger.record("validation/mean_reward", row["mean_reward"])
        self.last_step = step
        print(f"validation step={step} reward={row['mean_reward']:.1f}", flush=True)

    def _on_training_start(self):
        self.measure()

    def _on_step(self):
        if self.num_timesteps % self.cfg["eval_freq"] == 0:
            self.measure()
        return True

    def _on_training_end(self):
        self.measure()


def train(cfg: dict):
    folder = RESULTS / cfg["algo"]
    if (folder / "train.json").exists() or (folder / "validation.json").exists():
        raise FileExistsError(f"Archive existing experiment first: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["torch_threads"])
    env = Monitor(gym.make(cfg["env"]), str(folder / "train.monitor.csv"))
    started = time.perf_counter()
    try:
        model = ALGOS[cfg["algo"]](
            cfg["policy"],
            env,
            seed=cfg["seed"],
            verbose=0,
            device=cfg["device"],
            tensorboard_log=str(RUNS),
            **cfg.get("hyperparams", {}),
        )
        model.learn(
            total_timesteps=cfg["total_timesteps"], tb_log_name=cfg["algo"], callback=ValidationCallback(cfg, folder)
        )
        elapsed = time.perf_counter() - started
        MODELS.mkdir(exist_ok=True)
        path = MODELS / f"{cfg['algo']}.zip"
        model.save(path)
        meta = {
            "config": cfg,
            "actual_timesteps": model.num_timesteps,
            "wall_seconds_including_validation": elapsed,
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ["gymnasium", "stable-baselines3", "torch", "numpy"]},
        }
        (folder / "train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{cfg['algo']}: {model.num_timesteps} steps, {elapsed:.1f}s", flush=True)
        return model
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train(load_cfg(ap.parse_args().config))
