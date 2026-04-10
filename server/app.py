from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys

# Add parent directory to sys.path so we can import from env and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DisasterResponseEnv
from models import Action, Observation, Reward

# Global environment instance
current_env = DisasterResponseEnv(task_level="easy")

app = FastAPI(title="AI Emergency Response Coordination API")

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool

@app.post("/reset", response_model=Observation)
async def reset_env(task_id: Optional[str] = "easy"):
    global current_env
    try:
        current_env = DisasterResponseEnv(task_level=task_id)
        return current_env.reset()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step_env(action: Action):
    global current_env
    try:
        obs, reward, done = current_env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=Observation)
async def get_state():
    global current_env
    return current_env.state()

@app.get("/")
async def root():
    return {"status": "running", "env": "emergency-response-env"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
