from env import DisasterResponseEnv
from models import Action, Observation
import time
import gradio as gr
import sys
from io import StringIO
import numpy as np
import heapq

def astar_move(drone_pos, target_pos, obstacles, grid_size):
    """
    Finds the first move index (0-8) towards target using A* search.
    """
    move_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0), 
                5: (1, 1), 6: (-1, 1), 7: (1, -1), 8: (-1, -1)}
    
    # Priority queue for A* (f_score, pos, move_idx_to_get_there)
    pq = [(0, drone_pos, 0)] # f_score, current_pos, first_move_idx
    visited = set()
    
    # Track the first move to reach each node
    first_move = {}

    while pq:
        f, curr, move_idx = heapq.heappop(pq)
        
        if curr in visited: continue
        visited.add(curr)
        
        if curr == target_pos:
            return first_move.get(curr, 0)
        
        for m_idx, delta in move_map.items():
            if m_idx == 0: continue # Skip 'Stay' for pathfinding
            
            neighbor = (curr[0] + delta[0], curr[1] + delta[1])
            
            # Check bounds and obstacles
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1] and neighbor not in obstacles:
                if neighbor not in visited:
                    # G-score: distance from start (using steps)
                    # H-score: distance to target (using Manhattan or Euclidean)
                    h = abs(target_pos[0] - neighbor[0]) + abs(target_pos[1] - neighbor[1])
                    f_new = f + 1 + h
                    
                    # Store the move that led to this node (only for the very first step)
                    if curr == drone_pos:
                        first_move[neighbor] = m_idx
                    else:
                        first_move[neighbor] = first_move[curr]
                        
                    heapq.heappush(pq, (f_new, neighbor, m_idx))
                    
    return 0 # No path found

def run_simulation(task_level: str = "easy"):
    # Redirect stdout to capture the grid visualization
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    print(f"--- Running {task_level.upper()} Task ---")
    env = DisasterResponseEnv(task_level=task_level)
    obs = env.reset()
    done = False
    
    while not done:
        active_survivors = [s for s in obs.survivors if s[2] == "alive"]
        moves = []
        
        # Sort survivors by "Survival Urgency" (distance from 0,0) for HARD task
        if task_level == "hard":
            active_survivors.sort(key=lambda s: np.sqrt(s[0]**2 + s[1]**2))
            
        claimed_survivors = set()
        for i in range(len(obs.agent_positions)):
            curr_pos = obs.agent_positions[i]
            
            if not active_survivors:
                moves.append(0)
                continue
            
            # Find the best survivor that hasn't been claimed yet
            target = None
            for s in active_survivors:
                s_tuple = (s[0], s[1])
                if s_tuple not in claimed_survivors:
                    target = s_tuple
                    claimed_survivors.add(s_tuple)
                    break
            
            # If all survivors are claimed, help with the closest one
            if not target:
                dists = [abs(s[0] - curr_pos[0]) + abs(s[1] - curr_pos[1]) for s in active_survivors]
                target = (active_survivors[np.argmin(dists)][0], active_survivors[np.argmin(dists)][1])
            
            # Get move via A*
            move = astar_move(curr_pos, target, set(obs.obstacles), env.grid_size)
            moves.append(move)
        
        obs, reward, done = env.step(Action(moves=moves))
        
        # Visualize EASY task frequently
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
    gr.Markdown("Click the button below to run the baseline rescue simulation for all three tasks (Easy, Medium, Hard).")
    
    with gr.Row():
        btn = gr.Button("🚀 Run Full Simulation", variant="primary")
    
    out = gr.Textbox(label="Simulation Logs & Results", lines=30, interactive=False)
    
    btn.click(run_all_tasks, outputs=out)
    
    gr.Markdown("### ℹ️ Legend")
    gr.Markdown("- `0, 1, 2` → Drones | `S` → Survivor | `R` → Rescued | `X` → Obstacle | `.` → Empty")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
