from pydantic import BaseModel
from typing import List, Tuple

class TaskConfig(BaseModel):
    name: str
    num_drones: int
    num_survivors: int
    grid_size: Tuple[int, int]
    battery_limit: int
    rising_flood: bool
    flood_rate: float
    seed: int

def get_task_config(level: str) -> TaskConfig:
    configs = {
        "easy": TaskConfig(
            name="Task 1: Basic Navigation",
            num_drones=1,
            num_survivors=2,
            grid_size=(10, 10),
            battery_limit=1000,
            rising_flood=False,
            flood_rate=0.0,
            seed=42,
        ),
        "medium": TaskConfig(
            name="Task 2: Coordination",
            num_drones=2,
            num_survivors=5,
            grid_size=(15, 15),
            battery_limit=50,
            rising_flood=False,
            flood_rate=0.0,
            seed=42,
        ),
        "hard": TaskConfig(
            name="Task 3: Flood Rescue",
            num_drones=3,
            num_survivors=10,
            grid_size=(20, 20),
            battery_limit=100,
            rising_flood=True,
            flood_rate=0.05,
            seed=42,
        ),
    }
    return configs.get(level.lower(), configs["easy"])
