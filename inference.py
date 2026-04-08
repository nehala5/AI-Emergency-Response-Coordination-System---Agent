from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

app = FastAPI()

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
        self.water_cells = [[0, 5], [1, 5], [2, 5]]
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
        reward_total = -0.1

        deltas = {
            0: (0,0), 1: (0,1), 2: (0,-1),
            3: (1,0), 4: (-1,0), 5: (1,1),
            6: (-1,1), 7: (1,-1), 8: (-1,-1)
        }

        for i, move in enumerate(moves[:N_AGENTS]):
            ax, ay = self.agents[i]
            dr, dc = deltas.get(int(move), (0, 0))
            nr = max(0, min(GRID_SIZE-1, ax+dr))
            nc = max(0, min(GRID_SIZE-1, ay+dc))
            self.agents[i] = [nr, nc]

            for s in self.survivors:
                if s[2] == "alive" and s[0] == nr and s[1] == nc:
                    s[2] = "rescued"
                    self.rescued += 1
                    reward_total += 10

        if self.step_count % 7 == 0 and self.water_cells:
            base = self.water_cells[np.random.randint(len(self.water_cells))]
            dr2, dc2 = int(np.random.choice([-1,0,1])), int(np.random.choice([-1,0,1]))
            nr2 = max(0, min(GRID_SIZE-1, base[0]+dr2))
            nc2 = max(0, min(GRID_SIZE-1, base[1]+dc2))
            if [nr2, nc2] not in self.safe_zones:
                self.water_cells.append([nr2, nc2])
                for s in self.survivors:
                    if s[2] == "alive" and s[0] == nr2 and s[1] == nc2:
                        s[2] = "drowned"
                        reward_total -= 5

        alive = [s for s in self.survivors if s[2] == "alive"]
        self.done = len(alive) == 0 or self.step_count >= 200

        return self._obs(), reward_total, self.done


env = EnvState()


@app.get("/")
def root():
    return {"status": "running", "env": "disaster-response"}


@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}


class StepRequest(BaseModel):
    moves: list = [0, 0, 0]


@app.get("/step")
@app.post("/step")
def step(body: StepRequest = None):
    moves = body.moves if body else [0, 0, 0]
    obs, reward, done = env.step(moves)
    return {
        "observation": obs,
        "reward": {
            "value": round(reward, 3),
            "feedback": f"Step {env.step_count} done"
        },
        "done": done,
        "info": {"rescued": env.rescued}
    }


@app.get("/observation")
def observation():
    return {"observation": env._obs()}
