from env import DisasterResponseEnv
from models import Action, Observation
import time
import gradio as gr
import sys
from io import StringIO
import numpy as np

def get_best_move(drone_pos, target_pos, obstacles, grid_size):
    """
    Greedy movement with basic obstacle avoidance.
    """
    move_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0), 
                5: (1, 1), 6: (-1, 1), 7: (1, -1), 8: (-1, -1)}
    
    # Preferred directions
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]
    
    # Candidate moves ranked by how much they reduce distance
    candidates = []
    for move_idx, delta in move_map.items():
        new_pos = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])
        
        # Check bounds
        if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
            continue
            
        # Check obstacles
        if new_pos in obstacles:
            continue
            
        # Calculate new distance (Manhattan)
        new_dist = abs(target_pos[0] - new_pos[0]) + abs(target_pos[1] - new_pos[1])
        candidates.append((new_dist, move_idx))
    
    if not candidates:
        return 0 # Stay if no valid moves
        
    # Pick the move that minimizes distance
    candidates.sort()
    return candidates[0][1]

def run_simulation(task_level: str = "easy"):
    # Redirect stdout to capture the grid visualization
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    print(f"--- Running {task_level.upper()} Task ---")
    env = DisasterResponseEnv(task_level=task_level)
    obs = env.reset()
    done = False
    
    # Run simulation
    while not done:
        active_survivors = [s for s in obs.survivors if s[2] == "alive"]
        moves = []
        
        for i in range(len(obs.agent_positions)):
            if not active_survivors:
                moves.append(0)
                continue
            
            curr_pos = obs.agent_positions[i]
            
            # Find nearest survivor for this drone
            dists = [abs(s[0] - curr_pos[0]) + abs(s[1] - curr_pos[1]) for s in active_survivors]
            nearest_idx = np.argmin(dists)
            target = active_survivors[nearest_idx]
            
            # Get best move considering obstacles
            move = get_best_move(curr_pos, (target[0], target[1]), obs.obstacles, env.grid_size)
            moves.append(move)
        
        obs, reward, done = env.step(Action(moves=moves))
        
        # Visualize occasionally or on events
        if task_level == "easy" and env.steps % 5 == 0:
            env.render()
            
        if "rescued" in reward.feedback or "drowned" in reward.feedback:
            print(f"Step {env.steps}: {reward.feedback}")

    final_score = env.get_final_score()
    success_rate = (env.rescued_count / env.config.num_survivors) * 100
    
    print(f"\nFinal Score: {final_score:.4f}")
    print(f"Success Rate: {success_rate:.1f}% ({env.rescued_count}/{env.config.num_survivors})")
    print(f"Total Steps: {env.steps}")
    
    sys.stdout = old_stdout
    return mystdout.getvalue()

def run_all_tasks():
    easy_results = run_simulation("easy")
    medium_results = run_simulation("medium")
    hard_results = run_simulation("hard")
    return f"{easy_results}\n\n{medium_results}\n\n{hard_results}"

# Gradio UI
with gr.Blocks(title="AI Emergency Response Coordination") as demo:
    gr.Markdown("# 🚁 AI Emergency Response Coordination System")
    gr.Markdown("An advanced multi-agent RL environment for disaster response coordination.")
    
    with gr.Row():
        btn = gr.Button("🚀 Run Full Simulation", variant="primary")
    
    out = gr.Textbox(label="Simulation Logs & Results", lines=30, interactive=False)
    
    btn.click(run_all_tasks, outputs=out)
    
    gr.Markdown("### ℹ️ Legend")
    gr.Markdown("- `0, 1, 2` → Drones | `S` → Survivor | `R` → Rescued | `X` → Obstacle | `.` → Empty")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
