#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p gif

python - <<'PY'
import os
from utils import load_env, draw_gif_from_seq, step_cost
from doorkey import doorkey_problem

known_envs = [
    "doorkey-5x5-normal",
    "doorkey-6x6-normal",
    "doorkey-6x6-direct",
    "doorkey-6x6-shortcut",
    "doorkey-8x8-normal",
    "doorkey-8x8-direct",
    "doorkey-8x8-shortcut",
]

print("=== Known Map Experiments ===")
for name in known_envs:
    env_path = f"./envs/known_envs/{name}.env"
    env, _ = load_env(env_path)
    seq = doorkey_problem(env)
    total_cost = sum(step_cost(a) for a in seq)
    print(f"{name:22s} | steps={len(seq):3d} | cost={total_cost:3d}")
    draw_gif_from_seq(seq, env, path=f"./gif/{name}.gif")
PY
