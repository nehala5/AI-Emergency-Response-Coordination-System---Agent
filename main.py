from env import DisasterResponseEnv
from models import Action
import time

def run_baseline(task_level: str = "easy"):
    print(f"--- Running {task_level.upper()} Task ---")
    env = DisasterResponseEnv(task_level=task_level)
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Simple policy: Move towards the first available survivor
        active_survivors = [s for s in obs.survivors if s[2] == "alive"]
        moves = []
        
        for i in range(len(obs.agent_positions)):
            if not active_survivors:
                moves.append(0) # Stay
                continue
            
            # Target the survivor at index 0 (shared by all drones for simplicity in baseline)
            target = active_survivors[0]
            curr_pos = obs.agent_positions[i]
            
            dx = target[0] - curr_pos[0]
            dy = target[1] - curr_pos[1]
            
            # Simplified cardinal movement logic
            if dx > 0: move = 3 # East
            elif dx < 0: move = 4 # West
            elif dy > 0: move = 1 # North
            elif dy < 0: move = 2 # South
            else: move = 0 # Arrived
            
            moves.append(move)

        action = Action(moves=moves)
        obs, reward, done = env.step(action)
        total_reward += reward.score
        
        if task_level == "easy":
            env.render()
            time.sleep(0.1)
        
        if "rescued" in reward.feedback or "drowned" in reward.feedback:
            print(f"Step {env.steps}: {reward.feedback}")

    final_score = env.get_final_score()
    print(f"Final Score: {final_score:.4f}")
    print(f"Total Rescued: {env.rescued_count}/{env.config.num_survivors}")
    return final_score

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")
