#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p gif

# usage:
#   sh 3_start_variations.sh [env_name]
# example:
#   sh 3_start_variations.sh doorkey-8x8-normal
ENV_NAME="${1:-doorkey-8x8-normal}"

python - <<'PY' "$ENV_NAME"
import copy
import sys
from utils import load_env, draw_gif_from_seq, step_cost
from doorkey import doorkey_problem

env_name = sys.argv[1]
base_path = f"./envs/known_envs/{env_name}.env"
base_env, _ = load_env(base_path)

# (x, y, dir): dir 0=Right, 1=Down, 2=Left, 3=Up
start_cases = [
    (2, 1, 0),
    (2, 1, 1),
    (1, 2, 2),
    (3, 2, 3),
]

print(f"=== Start Variations: {env_name} ===")
for idx, (x, y, d) in enumerate(start_cases):
    env = copy.deepcopy(base_env)
    env.unwrapped.agent_pos = (x, y)
    env.unwrapped.agent_dir = d
    env.unwrapped.gen_obs()

    seq = doorkey_problem(env)
    total_cost = sum(step_cost(a) for a in seq)
    print(f"case{idx}: start=({x},{y},{d}) | steps={len(seq):3d} | cost={total_cost:3d}")
    draw_gif_from_seq(seq, env, path=f"./gif/{env_name}_start_case_{idx}.gif")
PY
