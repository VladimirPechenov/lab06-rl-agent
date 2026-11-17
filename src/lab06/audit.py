"""Recompute published metrics from saved episode-level evidence."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from .common import MODELS, RESULTS, load_cfg


def audit(check_models=False):
    for algo in ("ppo", "dqn"):
        folder = RESULTS / algo
        meta = json.loads((folder / "train.json").read_text())
        cfg = load_cfg(folder.parents[1] / "configs" / f"{algo}.yaml")
        assert meta["config"] == cfg, "Config differs from executed experiment"
        assert meta["actual_timesteps"] == cfg["total_timesteps"] == 100352
        frame = pd.read_csv(folder / "train.monitor.csv", comment="#")
        assert (frame.r == frame.l).all()
        assert frame.l.sum() <= meta["actual_timesteps"]
        assert meta["actual_timesteps"] - frame.l.sum() < 500
        points = json.loads((folder / "validation.json").read_text())
        assert points[0]["timesteps"] == 0 and points[-1]["timesteps"] == 100352
        for p in points:
            assert len(p["rewards"]) == 10
            assert np.isclose(np.mean(p["rewards"]), p["mean_reward"])
            assert np.isclose(np.std(p["rewards"]), p["std_reward"])
        result = json.loads((RESULTS / f"eval_{algo}.json").read_text())
        rows = result["episode_results"]
        assert len(rows) == result["episodes"] == 20
        assert [r["seed"] for r in rows] == list(range(2000, 2020))
        assert all(r["length"] == r["reward"] and (r["terminated"] or r["truncated"]) for r in rows)
        assert np.isclose(np.mean([r["reward"] for r in rows]), result["mean_reward"])
        assert np.isclose(np.std([r["reward"] for r in rows]), result["std_reward"])
        assert result["passed"] == (result["mean_reward"] >= 450)
        assert result["model_sha256"] == meta["model_sha256"]
        if check_models:
            assert hashlib.sha256((MODELS / f"{algo}.zip").read_bytes()).hexdigest() == meta["model_sha256"]
        print(f"{algo}: raw logs and reported metrics agree")
    video = json.loads((RESULTS / "video_ppo.json").read_text())
    assert [r["seed"] for r in video["episodes"]] == list(range(3000, 3005))
    assert video["frames"] == sum(r["length"] for r in video["episodes"])
    assert video["model_sha256"] == json.loads((RESULTS / "eval_ppo.json").read_text())["model_sha256"]
    assert json.loads((RESULTS / "eval_ppo.json").read_text())["passed"]
    print("PPO passes the lab threshold; video contains five complete episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    audit(parser.parse_args().models)
