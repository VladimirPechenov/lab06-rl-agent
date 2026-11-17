"""Record five complete, unedited seeded episodes with live counters."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .common import ALGOS, MODELS, RESULTS, ROOT


def record(algo: str, env_id: str = "CartPole-v1", episodes: int = 5, seed: int = 3000):
    torch.set_num_threads(1)
    folder = RESULTS / "videos" / algo
    folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/img").mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{algo}.zip"
    model = ALGOS[algo].load(path, device="cpu")
    font_path = ("C:/Windows/Fonts/arial.ttf" if Path("C:/Windows/Fonts/arial.ttf").exists()
                 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, 20)
    env = gym.make(env_id, render_mode="rgb_array")
    fps = env.metadata["render_fps"]
    rows, frames = [], 0
    output = folder / f"{algo}_5_episodes.mp4"
    try:
        with imageio.get_writer(str(output), fps=fps, codec="libx264", macro_block_size=1) as writer:
            for i in range(episodes):
                obs, _ = env.reset(seed=seed + i)
                reward_sum, steps = 0.0, 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
                    steps += 1
                    reward_sum += float(r)
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 600, 76), fill="#13223b")
                    draw.text((16, 8), f"{algo.upper()} | CartPole-v1 | Episode {i + 1}/{episodes}", font=font, fill="white")
                    draw.text((16, 40), f"seed={seed + i}   step={steps}/500   return={reward_sum:.0f}",
                              font=font, fill="#8ee6c6")
                    writer.append_data(np.asarray(frame))
                    frames += 1
                    if i == 0 and steps == 250:
                        frame.save(ROOT / "docs/img" / f"{algo}_episode.png")
                    if terminated or truncated:
                        break
                rows.append({"episode": i + 1, "seed": seed + i, "reward": reward_sum, "length": steps,
                             "terminated": bool(terminated), "truncated": bool(truncated)})
    finally:
        env.close()
    manifest = {"algo": algo, "env": env_id, "episodes": rows, "fps": fps, "frames": frames,
                "duration_seconds": frames / fps, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "file": output.name,
                "editing": "No cuts or speed changes; original Gymnasium frames with live counters."}
    (RESULTS / f"video_{algo}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3000)
    args = ap.parse_args()
    record(args.algo, episodes=args.episodes, seed=args.seed)
