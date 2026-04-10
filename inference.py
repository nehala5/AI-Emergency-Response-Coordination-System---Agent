import os
import requests
import numpy as np
import time

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "disaster-response-agent")
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

def get_action(obs):
    """
    Simple heuristic: move drones towards the nearest alive survivors.
    """
    moves = []
    agent_positions = obs.get("agent_positions", [])
    survivors = obs.get("survivors", [])
    obstacles = set(tuple(o) for o in obs.get("obstacles", []))
    
    active_survivors = [s for s in survivors if s[2] == "alive"]
    
    for i, pos in enumerate(agent_positions):
        if not active_survivors:
            moves.append(0)
            continue
        
        # Find nearest survivor
        target = min(active_survivors, key=lambda s: abs(s[0]-pos[0]) + abs(s[1]-pos[1]))
        tx, ty = target[0], target[1]
        ax, ay = pos[0], pos[1]
        
        # Simple greedy move
        dx = np.sign(tx - ax)
        dy = np.sign(ty - ay)
        
        # Map (dx, dy) to move index (0-8)
        # 0: (0,0), 1: (0,1), 2: (0,-1), 3: (1,0), 4: (-1,0), 
        # 5: (1,1), 6: (-1,1), 7: (1,-1), 8: (-1,-1)
        move_map = {
            (0, 0): 0, (0, 1): 1, (0, -1): 2,
            (1, 0): 3, (-1, 0): 4, (1, 1): 5,
            (-1, 1): 6, (1, -1): 7, (-1, -1): 8
        }
        
        move = move_map.get((dx, dy), 0)
        moves.append(int(move))
        
    return {"moves": moves}

def main():
    print("START")
    
    # 1. Reset Environment
    try:
        response = requests.post(f"{API_BASE_URL}/reset", headers=headers, params={"task_id": "easy"})
        response.raise_for_status()
        obs = response.json()
    except Exception as e:
        print(f"Failed to connect to environment: {e}")
        return

    done = False
    step_count = 0
    
    # 2. Step Loop
    while not done and step_count < 200:
        step_count += 1
        
        action = get_action(obs)
        
        try:
            response = requests.post(f"{API_BASE_URL}/step", json=action, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            obs = data["observation"]
            reward = data["reward"]
            done = data["done"]
            
            print(f"STEP {step_count}: {reward.get('feedback', 'no_feedback')}")
            
        except Exception as e:
            print(f"Error during step: {e}")
            break
            
    print("END")

if __name__ == "__main__":
    main()
