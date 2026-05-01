#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p gif

# usage: sh 2_random_maps.sh [num_trials]
NUM_TRIALS="${1:-10}"

python - <<'PY' "$NUM_TRIALS"
import os
import sys
import __main__
from create_env import DoorKey10x10Env
from utils import load_random_env, draw_gif_from_seq, step_cost
from doorkey import doorkey_problem

# pickle로 저장된 커스텀 클래스를 안전하게 복원하기 위한 매핑
__main__.DoorKey10x10Env = DoorKey10x10Env

num_trials = int(sys.argv[1])
env_folder = "./envs/random_envs"

print("=== Random Map Experiments ===")
for k in range(num_trials):
    env, _, env_path = load_random_env(env_folder)
    seq = doorkey_problem(env)
    total_cost = sum(step_cost(a) for a in seq)
    print(f"[{k:02d}] {os.path.basename(env_path):20s} | steps={len(seq):3d} | cost={total_cost:3d}")
    draw_gif_from_seq(seq, env, path=f"./gif/random_{k:02d}.gif")
PY
