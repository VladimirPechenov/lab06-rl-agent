"""Regenerate comparison from raw Monitor and validation logs."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .common import RESULTS  # noqa: E402


def plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for algo, color in [("ppo", "#2563eb"), ("dqn", "#d97706")]:
        frame = pd.read_csv(RESULTS / algo / "train.monitor.csv", comment="#")
        axes[0].plot(frame.l.cumsum(), frame.r.rolling(100, min_periods=1).mean(), label=algo.upper(), color=color)
        points = json.loads((RESULTS / algo / "validation.json").read_text())
        axes[1].errorbar(
            [p["timesteps"] for p in points],
            [p["mean_reward"] for p in points],
            yerr=[p["std_reward"] for p in points],
            label=algo.upper(),
            color=color,
            marker="o",
            capsize=2,
        )
    for ax in axes:
        ax.axhline(450, color="#64748b", ls="--", label="Lab threshold: 450")
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 600)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("Training: rolling mean, 100 episodes")
    axes[1].set_title("Validation: 10 fixed seeds, mean +/- std")
    fig.suptitle("CartPole-v1 | seed=42 | final budget=100,352 steps")
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot()
