import os
from env import DisasterResponseEnv
from models import Action
import numpy as np
import heapq

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "disaster-response-agent")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def astar_move(drone_pos, target_pos, obstacles, grid_size):
    move_map = {
        0: (0, 0), 1: (0, 1), 2: (0, -1),
        3: (1, 0), 4: (-1, 0),
        5: (1, 1), 6: (-1, 1),
        7: (1, -1), 8: (-1, -1)
    }

    pq = [(0, drone_pos)]
    visited = set()
    first_move = {}

    while pq:
        cost, curr = heapq.heappop(pq)

        if curr in visited:
            continue
        visited.add(curr)

        if curr == target_pos:
            return first_move.get(curr, 0)

        for m_idx, delta in move_map.items():
            if m_idx == 0:
                continue

            neighbor = (curr[0] + delta[0], curr[1] + delta[1])

            if (
                0 <= neighbor[0] < grid_size[0]
                and 0 <= neighbor[1] < grid_size[1]
                and neighbor not in obstacles
            ):
                if neighbor not in visited:
                    h = abs(target_pos[0] - neighbor[0]) + abs(target_pos[1] - neighbor[1])
                    f = cost + 1 + h

                    if curr == drone_pos:
                        first_move[neighbor] = m_idx
                    else:
                        first_move[neighbor] = first_move[curr]

                    heapq.heappush(pq, (f, neighbor))

    return 0


def run_agent(task_level="easy"):
    env = DisasterResponseEnv(task_level=task_level)
    obs = env.reset()

    print("START")

    done = False
    while not done:
        active_survivors = [s for s in obs.survivors if s[2] == "alive"]
        moves = []

        if task_level == "hard":
            active_survivors.sort(key=lambda s: abs(s[0]) + abs(s[1]))

        claimed = set()

        for i, pos in enumerate(obs.agent_positions):
            if not active_survivors:
                moves.append(0)
                continue

            target = None
            for s in active_survivors:
                t = (s[0], s[1])
                if t not in claimed:
                    target = t
                    claimed.add(t)
                    break

            if not target:
                target = (active_survivors[0][0], active_survivors[0][1])

            move = astar_move(pos, target, set(obs.obstacles), env.grid_size)
            moves.append(move)

        obs, reward, done = env.step(Action(moves=moves))

        print(f"STEP {env.steps}: {reward.feedback}")

    print("END")


if __name__ == "__main__":
    run_agent("easy")