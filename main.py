from env import DisasterResponseEnv
from models import Action
import time
import gradio as gr
import sys
from io import StringIO

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
            
            # Coordination: Target different survivors
            # Drone i targets survivor (i % num_survivors)
            target_idx = i % len(active_survivors)
            target = active_survivors[target_idx]
            
            curr_pos = obs.agent_positions[i]
            dx, dy = target[0] - curr_pos[0], target[1] - curr_pos[1]
            
            if dx > 0: move = 3
            elif dx < 0: move = 4
            elif dy > 0: move = 1
            elif dy < 0: move = 2
            else: move = 0
            moves.append(move)
        
        obs, reward, done = env.step(Action(moves=moves))
        if task_level == "easy" and env.steps % 5 == 0:
            env.render()
        if "rescued" in reward.feedback or "drowned" in reward.feedback:
            print(f"Step {env.steps}: {reward.feedback}")

    final_score = env.get_final_score()
    print(f"\nFinal Score: {final_score:.4f}")
    print(f"Total Rescued: {env.rescued_count}/{env.config.num_survivors}")
    
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
    gr.Markdown("- `0, 1, 2` → Drones | `S` → Survivor | `R` → Rescued | `.` → Empty")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
