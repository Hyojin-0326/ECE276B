import heapq
from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Wall

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

# dir integer -> (dx, dy): 0=Right, 1=Down, 2=Left, 3=Up
DIR_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)


def doorkey_problem(env):
    """
    Dijkstra-based DP planner for the DoorKey environment.
    Works for any grid size and handles 1 or 2 doors generically.

    State: (x, y, dir, has_key, doors_tuple)
      - x, y     : agent grid position
      - dir      : facing direction integer (0=R,1=D,2=L,3=U)
      - has_key  : 0 or 1
      - doors_tuple : one bool per door, True = open
    """
    uw = env.unwrapped
    width, height = uw.width, uw.height

    # --- scan the grid once to collect all static map info ---
    walls = set()
    key_pos = None
    door_positions = []   # ordered list of (x,y) for every door found
    door_open_init = []   # matching initial open-status booleans
    goal_pos = None

    for x in range(width):
        for y in range(height):
            cell = uw.grid.get(x, y)
            if isinstance(cell, Wall):
                walls.add((x, y))
            elif isinstance(cell, Key):
                key_pos = (x, y)
            elif isinstance(cell, Door):
                door_positions.append((x, y))
                door_open_init.append(cell.is_open)
            elif isinstance(cell, Goal):
                goal_pos = (x, y)

    init_doors = tuple(door_open_init)

    # build the start state from the environment
    ax, ay = int(uw.agent_pos[0]), int(uw.agent_pos[1])
    adir = int(uw.agent_dir)
    has_key_init = 1 if (uw.carrying is not None) else 0
    start_state = (ax, ay, adir, has_key_init, init_doors)

    # ------------------------------------------------------------------
    # Transition model: given a state + action, return (next_state, cost)
    # or None if the action is physically invalid from that state.
    # ------------------------------------------------------------------
    def get_next_state(state, action):
        x, y, d, hk, doors = state
        dx, dy = DIR_DELTA[d]
        nx, ny = x + dx, y + dy   # cell directly in front of agent

        if action == MF:
            # reject out-of-bounds
            if not (0 <= nx < width and 0 <= ny < height):
                return None
            cell = uw.grid.get(nx, ny)
            if isinstance(cell, Wall):
                return None
            # can't walk into a door that's still closed
            if isinstance(cell, Door):
                di = door_positions.index((nx, ny))
                if not doors[di]:
                    return None
            return ((nx, ny, d, hk, doors), 3)

        elif action == TL:
            return ((x, y, (d - 1) % 4, hk, doors), 1)

        elif action == TR:
            return ((x, y, (d + 1) % 4, hk, doors), 1)

        elif action == PK:
            if hk == 1:
                return None
            if not (0 <= nx < width and 0 <= ny < height):
                return None
            cell = uw.grid.get(nx, ny)
            if isinstance(cell, Key):
                return ((x, y, d, 1, doors), 2)
            return None

        elif action == UD:
            # front cell must be a closed door and agent must hold the key
            if not (0 <= nx < width and 0 <= ny < height):
                return None
            cell = uw.grid.get(nx, ny)
            if isinstance(cell, Door) and hk == 1:
                di = door_positions.index((nx, ny))
                if not doors[di]:   # door is currently locked/closed
                    new_doors = list(doors)
                    new_doors[di] = True
                    return ((x, y, d, hk, tuple(new_doors)), 5)
            return None

        return None

    # Dijkstra over the (finite) state space
    INF = float('inf')
    costs = {start_state: 0}
    parent = {start_state: None}   # maps state -> (prev_state, action)
    pq = [(0, start_state)]
    goal_state = None

    while pq:
        curr_cost, curr_state = heapq.heappop(pq)

        # stale entry in the heap — skip it
        if curr_cost > costs.get(curr_state, INF):
            continue

        # reached the goal tile; no need to expand further
        if (curr_state[0], curr_state[1]) == goal_pos:
            goal_state = curr_state
            break

        for action in [MF, TL, TR, PK, UD]:
            result = get_next_state(curr_state, action)
            if result is None:
                continue
            next_state, action_cost = result
            new_cost = curr_cost + action_cost
            if new_cost < costs.get(next_state, INF):
                costs[next_state] = new_cost
                parent[next_state] = (curr_state, action)
                heapq.heappush(pq, (new_cost, next_state))

    if goal_state is None:
        raise RuntimeError("Dijkstra found no path to the goal — check the map.")

    actions = []
    state = goal_state
    while parent[state] is not None:
        prev_state, act = parent[state]
        actions.append(act)
        state = prev_state
    actions.reverse()

    return actions


def partA():
    known_envs = [
        "doorkey-5x5-normal",
        "doorkey-6x6-normal",
        "doorkey-6x6-direct",
        "doorkey-6x6-shortcut",
        "doorkey-8x8-normal",
        "doorkey-8x8-direct",
        "doorkey-8x8-shortcut",
    ]

    import os
    os.makedirs("./gif", exist_ok=True)

    for name in known_envs:
        env_path = f"./envs/known_envs/{name}.env"
        env, info = load_env(env_path)
        print(f"\n[{name}]")
        print("  map info:", info)

        seq = doorkey_problem(env)
        print(f"  optimal action sequence ({len(seq)} steps): {seq}")

        gif_path = f"./gif/{name}.gif"
        draw_gif_from_seq(seq, load_env(env_path)[0], path=gif_path)


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    print(f"\n[Part B] loaded: {env_path}")
    print("  map info:", info)

    seq = doorkey_problem(env)
    print(f"  optimal action sequence ({len(seq)} steps): {seq}")

    import os
    os.makedirs("./gif", exist_ok=True)
    draw_gif_from_seq(seq, load_random_env(env_folder)[0], path="./gif/random_env.gif")


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()
