from env import DisasterResponseEnv
from models import Action, Observation
import time
import sys
import heapq
from io import StringIO
import numpy as np
try:
    import gradio as gr
except ImportError:
    gr = None

def astar_move(drone_pos, target_pos, obstacles, grid_size):
    """
    Finds the first move index (0-8) towards target using A* search.
    """
    move_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0), 
                5: (1, 1), 6: (-1, 1), 7: (1, -1), 8: (-1, -1)}
    
    pq = [(0, drone_pos, 0)] 
    visited = set()
    first_move = {}

    while pq:
        f, curr, move_idx = heapq.heappop(pq)
        if curr in visited: continue
        visited.add(curr)
        
        if curr == target_pos:
            return first_move.get(curr, 0)
        
        for m_idx, delta in move_map.items():
            if m_idx == 0: continue
            neighbor = (curr[0] + delta[0], curr[1] + delta[1])
            
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1] and neighbor not in obstacles:
                if neighbor not in visited:
                    h = abs(target_pos[0] - neighbor[0]) + abs(target_pos[1] - neighbor[1])
                    f_new = f + 1 + h
                    if curr == drone_pos: first_move[neighbor] = m_idx
                    else: first_move[neighbor] = first_move[curr]
                    heapq.heappush(pq, (f_new, neighbor, m_idx))
    return 0

def env_to_html(env):
    """
    Generates a color-coded HTML grid with a dark theme for high visibility.
    """
    grid_data = np.full(env.grid_size, ".", dtype=str)
    for ox, oy in env.obstacles: grid_data[ox, oy] = "X"
    for x, y, status in env.survivors:
        if status == "alive": grid_data[x, y] = "S"
        elif status == "rescued": grid_data[x, y] = "R"
        elif status == "drowned": grid_data[x, y] = "D"
    for i, (x, y) in enumerate(env.agent_positions): grid_data[x, y] = str(i)

    # High-contrast styling mapping
    colors = {
        "X": "#555555", # Grey Obstacle
        "S": "#4caf50", # Green Survivor
        "R": "#1b5e20", # Deep Green Rescued
        "D": "#b71c1c", # Deep Red Drowned
        ".": "#333333", # Dark Grey Empty
        "0": "#ff9800", # Orange Drone 0
        "1": "#fbc02d", # Yellow Drone 1
        "2": "#ff5722", # Deep Orange Drone 2
    }

    grid_data = grid_data.T[::-1]
    
    # Dark container for "Command Center" feel
    html = '<div style="display: grid; grid-template-columns: repeat({}, 22px); gap: 3px; background: #212121; padding: 20px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 2px solid #424242; width: fit-content; margin: auto;">'.format(env.grid_size[0])
    
    for row in grid_data:
        for cell in row:
            bg_color = colors.get(cell, colors["."])
            text_color = "white" if cell != "." else "#666666"
            content = f"<b>{cell}</b>" if cell != "." else "·"
                
            html += f'<div style="width: 22px; height: 22px; background: {bg_color}; color: {text_color}; display: flex; align-items: center; justify-content: center; font-family: \'Courier New\', Courier, monospace; font-size: 12px; border-radius: 3px; transition: transform 0.2s;">{content}</div>'
    
    html += '</div>'
    return html

def run_simulation(task_level: str = "easy"):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    env = DisasterResponseEnv(task_level=task_level)
    obs = env.reset()
    done = False
    
    while not done:
        active_survivors = [s for s in obs.survivors if s[2] == "alive"]
        moves = []
        if task_level == "hard": active_survivors.sort(key=lambda s: np.sqrt(s[0]**2 + s[1]**2))
            
        claimed_survivors = set()
        for i in range(len(obs.agent_positions)):
            curr_pos = obs.agent_positions[i]
            if not active_survivors:
                moves.append(0); continue
            
            target = None
            for s in active_survivors:
                s_tuple = (s[0], s[1])
                if s_tuple not in claimed_survivors:
                    target = s_tuple; claimed_survivors.add(s_tuple); break
            
            if not target:
                dists = [abs(s[0] - curr_pos[0]) + abs(s[1] - curr_pos[1]) for s in active_survivors]
                target = (active_survivors[np.argmin(dists)][0], active_survivors[np.argmin(dists)][1])
            
            moves.append(astar_move(curr_pos, target, set(obs.obstacles), env.grid_size))
        
        obs, reward, done = env.step(Action(moves=moves))
        if "rescued" in reward.feedback or "drowned" in reward.feedback:
            print(f"Step {env.steps}: {reward.feedback}")

    final_score = env.get_final_score()
    success_rate = (env.rescued_count / env.config.num_survivors) * 100
    
    stats = {
        "score": f"{final_score:.4f}",
        "success": f"{success_rate:.1f}%",
        "rescued": f"{env.rescued_count}/{env.config.num_survivors}",
        "lost": f"{env.drowned_count}",
        "steps": env.steps
    }
    
    visual_grid = env_to_html(env)
    sys.stdout = old_stdout
    return stats, visual_grid, mystdout.getvalue()

def run_all_tasks():
    e_stats, e_grid, e_log = run_simulation("easy")
    m_stats, m_grid, m_log = run_simulation("medium")
    h_stats, h_grid, h_log = run_simulation("hard")
    
    return (
        e_grid, f"Success: {e_stats['success']} | Score: {e_stats['score']}", e_log,
        m_grid, f"Success: {m_stats['success']} | Score: {m_stats['score']}", m_log,
        h_grid, f"Success: {h_stats['success']} | Score: {h_stats['score']}", h_log
    )

# Gradio Dashboard
if gr:
    with gr.Blocks(title="AI Emergency Response Coordination", css=".gradio-container {background: #000000; color: #ffffff;} .tabs {background: #000000; border: 1px solid #333333; border-radius: 8px;} button.primary {background: #ff9800 !important; border: none !important;} label {color: #bbbbbb !important; font-weight: bold;} .gradio-container {max-width: 100% !important;}") as demo:
        gr.Markdown("# 🚁 <span style='color: #ff9800;'>AI Emergency Response Mission Dashboard</span>")
        gr.Markdown("Visualize coordinates and success metrics for the autonomous rescue drone fleet.")
        
        with gr.Row():
            run_btn = gr.Button("🚀 Launch Mission Simulation", variant="primary", scale=2)
        
        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("Task 1: Easy Rescue"):
                with gr.Row():
                    e_status = gr.Textbox(label="Mission Status", interactive=False)
                with gr.Row():
                    e_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False):
                    e_logs = gr.Textbox(interactive=False, lines=10)
                    
            with gr.Tab("Task 2: Medium Coordination"):
                with gr.Row():
                    m_status = gr.Textbox(label="Mission Status", interactive=False)
                with gr.Row():
                    m_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False):
                    m_logs = gr.Textbox(interactive=False, lines=10)
                    
            with gr.Tab("Task 3: Hard Dynamic Response"):
                with gr.Row():
                    h_status = gr.Textbox(label="Mission Status", interactive=False)
                with gr.Row():
                    h_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False):
                    h_logs = gr.Textbox(interactive=False, lines=10)

        gr.Markdown("### ℹ️ Legend")
        gr.Markdown("- **<span style='color: #ff9800;'>0, 1, 2:</span>** Active Drones | **<span style='color: #4caf50;'>S:</span>** Survivor | **<span style='color: #1b5e20;'>R:</span>** Rescued | **<span style='color: #b71c1c;'>D:</span>** Drowned | **<span style='color: #555555;'>X:</span>** Obstacle")

        run_btn.click(
            run_all_tasks, 
            outputs=[
                e_visual, e_status, e_logs,
                m_visual, m_status, m_logs,
                h_visual, h_status, h_logs
            ]
        )

if __name__ == "__main__":
    if gr:
        demo.launch(server_name="0.0.0.0", server_port=7860)
