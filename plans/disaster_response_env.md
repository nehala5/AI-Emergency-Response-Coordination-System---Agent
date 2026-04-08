# Plan: AI Emergency Response Coordination System (OpenEnv)

## 1. Objective
Build a professional multi-agent reinforcement learning environment for coordinating emergency response operations (drones) to rescue survivors in a dynamic disaster zone.

## 2. Tasks (The 3-Tier Requirement)
- **Task 1 (Easy):** 1 drone, 2 survivors, static map. Goal: Basic navigation and rescue.
- **Task 2 (Medium):** 2 drones, battery constraints, 5 survivors. Goal: Multi-agent coordination.
- **Task 3 (Hard):** 3 drones, rising flood levels, 10 survivors. Goal: Real-time optimization under threat.

## 3. Data Models (Pydantic)
```python
from pydantic import BaseModel
from typing import List, Tuple

class Observation(BaseModel):
    agent_positions: List[Tuple[int, int]]
    batteries: List[int]
    survivors: List[Tuple[int, int, str]] # (x, y, status)
    flood_map: List[List[float]]

class Action(BaseModel):
    moves: List[int] # One move [0-6] for each drone

class Reward(BaseModel):
    score: float
    feedback: str
```

## 4. Deterministic Grading Logic
Final Score (0.0 to 1.0) = `(Rescued / Total)` 
- Penalty: `-0.1` per Drowned survivor.
- Penalty: `-0.01` per 10 steps taken.
- Penalty: `-0.05` per Battery depletion.

## 5. Implementation Steps (Tomorrow Morning)
1.  **Environment Interface:** Implement `reset()`, `step()`, and `state()` using OpenEnv standards.
2.  **Task Factory:** Create a function to initialize the 3 different task levels.
3.  **Deterministic Engine:** Use `np.random.RandomState(seed)` to ensure graders are consistent.
4.  **Reward & Grade:** Implement the incremental reward and the final deterministic grader.
5.  **Metadata & Deployment:**
    - Create `openenv.yaml`.
    - Write `Dockerfile`.
    - Create `README.md` with technical specs.
6.  **Validation:** Run `openenv validate` to ensure compliance.

## 6. Project Structure
- `env.py`: Core logic.
- `models.py`: Pydantic Observation/Action/Reward models.
- `tasks.py`: Task definitions (Easy/Medium/Hard).
- `openenv.yaml`: Metadata.
- `Dockerfile`: Containerization.
- `main.py`: Baseline inference script.
