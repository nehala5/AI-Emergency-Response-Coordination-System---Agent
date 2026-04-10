from env import DisasterResponseEnv
from models import Action, Observation, Reward
import time
import sys
import heapq
from io import StringIO
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

try:
    import gradio as gr
except ImportError:
    gr = None

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

# Existing simulation logic for Gradio
def astar_move(drone_pos, target_pos, obstacles, grid_size):
    move_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0), 
                5: (1, 1), 6: (-1, 1), 7: (1, -1), 8: (-1, -1)}
    pq = [(0, drone_pos, 0)] 
    visited = set()
    first_move = {}
    while pq:
        f, curr, move_idx = heapq.heappop(pq)
        if curr in visited: continue
        visited.add(curr)
        if curr == target_pos: return first_move.get(curr, 0)
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
    grid_data = np.full(env.grid_size, ".", dtype=str)
    for ox, oy in env.obstacles: grid_data[ox, oy] = "X"
    for x, y, status in env.survivors:
        if status == "alive": grid_data[x, y] = "S"
        elif status == "rescued": grid_data[x, y] = "R"
        elif status == "drowned": grid_data[x, y] = "D"
    for i, (x, y) in enumerate(env.agent_positions): grid_data[x, y] = str(i)
    colors = {"X": "#555555", "S": "#4caf50", "R": "#1b5e20", "D": "#b71c1c", ".": "#333333", "0": "#ff9800", "1": "#fbc02d", "2": "#ff5722"}
    grid_data = grid_data.T[::-1]
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
        claimed_survivors = set()
        for i in range(len(obs.agent_positions)):
            curr_pos = obs.agent_positions[i]
            if not active_survivors:
                moves.append(0)
                continue
            def priority(s):
                dist = abs(s[0] - curr_pos[0]) + abs(s[1] - curr_pos[1])
                flood_risk = obs.flood_map[s[0]][s[1]] if task_level == "hard" else 0
                return dist - (flood_risk * 8)
            available = [s for s in active_survivors if (s[0], s[1]) not in claimed_survivors]
            if available:
                best = min(available, key=priority)
                target = (best[0], best[1])
                claimed_survivors.add(target)
            else:
                best = min(active_survivors, key=priority)
                target = (best[0], best[1])
            move = astar_move(curr_pos, target, set(obs.obstacles), env.grid_size)
            moves.append(move)
        obs, reward, done = env.step(Action(moves=moves))
        print(f"STEP {env.steps}: {reward.feedback}")
    final_score = env.get_final_score()
    success_rate = (env.rescued_count / env.config.num_survivors) * 100
    stats = {"score": f"{final_score:.4f}", "success": f"{success_rate:.1f}%", "rescued": f"{env.rescued_count}/{env.config.num_survivors}", "lost": f"{env.drowned_count}", "steps": env.steps}
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
    with gr.Blocks(title="AI Emergency Response Coordination") as demo:
        gr.Markdown("# 🚁 AI Emergency Response Mission Dashboard")
        run_btn = gr.Button("🚀 Launch Mission Simulation", variant="primary")
        with gr.Tabs():
            with gr.Tab("Task 1: Easy Rescue"):
                e_status = gr.Textbox(label="Mission Status", interactive=False)
                e_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False): e_logs = gr.Textbox(interactive=False, lines=10)
            with gr.Tab("Task 2: Medium Coordination"):
                m_status = gr.Textbox(label="Mission Status", interactive=False)
                m_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False): m_logs = gr.Textbox(interactive=False, lines=10)
            with gr.Tab("Task 3: Hard Dynamic Response"):
                h_status = gr.Textbox(label="Mission Status", interactive=False)
                h_visual = gr.HTML()
                with gr.Accordion("Mission Logs", open=False): h_logs = gr.Textbox(interactive=False, lines=10)
        run_btn.click(run_all_tasks, outputs=[e_visual, e_status, e_logs, m_visual, m_status, m_logs, h_visual, h_status, h_logs])

if __name__ == "__main__":
    if gr:
        app = gr.mount_gradio_app(app, demo, path="/")
    
    uvicorn.run(app, host="0.0.0.0", port=7860)
