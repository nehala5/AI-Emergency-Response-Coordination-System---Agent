from pydantic import BaseModel
from typing import List, Tuple

class Observation(BaseModel):
    agent_positions: List[Tuple[int, int]]
    batteries: List[int]
    survivors: List[Tuple[int, int, str]]  # (x, y, status: "alive", "rescued", "drowned")
    flood_map: List[List[float]]  # 0.0 to 1.0 (water level)

class Action(BaseModel):
    moves: List[int]  # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right, 5: Diagonal-Up-Left... (mapped in env.py)

class Reward(BaseModel):
    score: float
    feedback: str
