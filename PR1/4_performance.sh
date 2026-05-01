#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# usage: sh 4_performance.sh [num_random_trials]
NUM_RANDOM="${1:-20}"

python - <<'PY' "$NUM_RANDOM"
import os
import sys
import time
import statistics
import __main__
from create_env import DoorKey10x10Env
from utils import load_env, load_random_env, step_cost
from doorkey import doorkey_problem

__main__.DoorKey10x10Env = DoorKey10x10Env

num_random = int(sys.argv[1])
known_envs = [
    "doorkey-5x5-normal",
    "doorkey-6x6-normal",
    "doorkey-6x6-direct",
    "doorkey-6x6-shortcut",
    "doorkey-8x8-normal",
    "doorkey-8x8-direct",
    "doorkey-8x8-shortcut",
]

print("=== Performance Summary ===")
print("\n[Known Maps]")
known_times = []
for name in known_envs:
    env, _ = load_env(f"./envs/known_envs/{name}.env")
    t0 = time.perf_counter()
    seq = doorkey_problem(env)
    dt = time.perf_counter() - t0
    known_times.append(dt)
    total_cost = sum(step_cost(a) for a in seq)
    print(f"{name:22s} | time={dt*1000:8.2f} ms | steps={len(seq):3d} | cost={total_cost:3d}")

print("\n[Random Maps]")
random_times = []
random_costs = []
for _ in range(num_random):
    env, _, _ = load_random_env("./envs/random_envs")
    t0 = time.perf_counter()
    seq = doorkey_problem(env)
    dt = time.perf_counter() - t0
    random_times.append(dt)
    random_costs.append(sum(step_cost(a) for a in seq))

print(f"random trials      : {num_random}")
print(f"avg time (ms)      : {statistics.mean(random_times)*1000:.2f}")
print(f"max time (ms)      : {max(random_times)*1000:.2f}")
print(f"avg cost           : {statistics.mean(random_costs):.2f}")
print(f"min / max cost     : {min(random_costs)} / {max(random_costs)}")
print(f"known avg time (ms): {statistics.mean(known_times)*1000:.2f}")
PY
