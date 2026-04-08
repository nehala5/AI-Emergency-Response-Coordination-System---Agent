import os
import requests
import numpy as np

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "default")
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

def get_action(obs):
    # simple nearest survivor logic (keep it basic for API)
    moves = []
    agents = obs["agent_positions"]
    survivors = [s for s in obs["survivors"] if s[2] == "alive"]

    for ax, ay in agents:
        if not survivors:
            moves.append(0)
            continue
        
        # nearest survivor
        target = min(survivors, key=lambda s: abs(s[0]-ax)+abs(s[1]-ay))
        tx, ty = target[0], target[1]

        dx = np.sign(tx - ax)
        dy = np.sign(ty - ay)

        move_map = {
            (0, 0): 0,
            (0, 1): 1, (0, -1): 2,
            (1, 0): 3, (-1, 0): 4,
            (1, 1): 5, (-1, 1): 6,
            (1, -1): 7, (-1, -1): 8
        }

        moves.append(move_map[(dx, dy)])

    return {"moves": moves}


def main():
    print("START")

    # 🔹 RESET
    res = requests.post(f"{API_BASE_URL}/reset", headers=headers)
    obs = res.json()

    step = 0

    while True:
        step += 1

        action = get_action(obs)

        # 🔹 STEP
        res = requests.post(
            f"{API_BASE_URL}/step",
            json=action,
            headers=headers
        )
        data = res.json()

        obs = data["observation"]
        done = data["done"]
        feedback = data["reward"]["feedback"]

        print(f"STEP {step}: {feedback}")

        if done:
            break

    print("END")


if __name__ == "__main__":
    main()
