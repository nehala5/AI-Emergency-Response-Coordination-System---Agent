import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Optional

app = FastAPI()

# ── Environment State ──────────────────────────────────────────────
GRID_SIZE = 10
N_AGENTS = 3

class EnvState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.done = False
        self.agents = [[4, 4], [4, 5], [5, 4]]
        self.survivors = [
            [1, 2, "alive"], [3, 7, "alive"], [6, 1, "alive"],
            [7, 8, "alive"], [2, 5, "alive"]
        ]
        self.water_cells = [[0, 5], [1, 5], [2, 5], [3, 3], [8, 2]]
        self.safe_zones = [[0, 0], [0, 9], [9, 0], [9, 9]]
        self.rescued = 0
        return self._obs()

    def _obs(self):
        return {
            "agent_positions": self.agents,
            "survivors": self.survivors,
            "water_cells": self.water_cells,
            "safe_zones": self.safe_zones,
            "step": self.step_count,
            "rescued": self.rescued
        }

    def step(self, moves):
        self.step_count += 1
        reward_total = -0.1 * N_AGENTS

        for i, move in enumerate(moves[:N_AGENTS]):
            ax, ay = self.agents[i]

            # Movement
            deltas = {
                0: (0, 0),  1: (0, 1),  2: (0, -1),
                3: (1, 0),  4: (-1, 0), 5: (1, 1),
                6: (-1, 1), 7: (1, -1), 8: (-1, -1)
            }
            dr, dc = deltas.get(move, (0, 0))
            nr = max(0, min(GRID_SIZE - 1, ax + dr))
            nc = max(0, min(GRID_SIZE - 1, ay + dc))
            self.agents[i] = [nr, nc]

            # Pick up survivor
            for s in self.survivors:
                if s[2] == "alive" and s[0] == nr and s[1] == nc:
                    if [nr, nc] in self.safe_zones:
                        s[2] = "rescued"
                        self.rescued += 1
                        reward_total += 10
                    else:
                        s[2] = "rescued"
                        self.rescued += 1
                        reward_total += 10

        # Water spreads occasionally
        if self.step_count % 7 == 0:
            if self.water_cells:
                base = self.water_cells[np.random.randint(len(self.water_cells))]
                dr, dc = np.random.choice([-1, 0, 1], 2)
                nr = max(0, min(GRID_SIZE - 1, base[0] + int(dr)))
                nc = max(0, min(GRID_SIZE - 1, base[1] + int(dc)))
                if [nr, nc] not in self.safe_zones and [nr, nc] not in self.water_cells:
                    self.water_cells.append([nr, nc])
                    # Drown survivors caught in new water
                    for s in self.survivors:
                        if s[2] == "alive" and s[0] == nr and s[1] == nc:
                            s[2] = "drowned"
                            reward_total -= 5

        alive = [s for s in self.survivors if s[2] == "alive"]
        self.done = (len(alive) == 0) or (self.step_count >= 200)

        return self._obs(), reward_total, self.done


env = EnvState()


# ── API Endpoints (what the checker calls) ─────────────────────────

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}


class StepRequest(BaseModel):
    moves: list


@app.post("/step")
def step(body: StepRequest):
    obs, reward, done = env.step(body.moves)
    return {
        "observation": obs,
        "reward": {"value": round(reward, 3), "feedback": f"Step {env.step_count}: reward={round(reward,2)}"},
        "done": done,
        "info": {"rescued": env.rescued, "step": env.step_count}
    }


@app.get("/observation")
def observation():
    return {"observation": env._obs()}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
