---
title: AI Emergency Response Coordination System
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# 🚁 AI Emergency Response Coordination System (OpenEnv)

## 🔹 Real-World Motivation
In the immediate aftermath of natural disasters like floods, rapid and coordinated search-and-rescue is critical. This environment simulates autonomous drone coordination in a dynamic disaster zone. Drones must navigate complex grids to rescue survivors while managing limited battery life and responding to **dynamic hazards** like rising flood levels and **static obstacles** (collapsed buildings) that threaten lives in real-time.

## 🔹 Advanced Engineering Features
- **Dynamic Source-Based Flooding:** Unlike simple uniform rising water, the flood in this environment spreads from a specific source (e.g., origin point), requiring agents to prioritize survivors in the immediate path of the wave.
- **Static Obstacles (`X`):** The grid includes collapsed buildings that drones must navigate around, adding a critical pathfinding layer to the coordination problem.
- **Multi-Agent Coordination Baseline:** The included baseline demonstrates a coordinated strategy where multiple drones target different survivors simultaneously to maximize rescue efficiency.
- **Deterministic Evaluation:** Uses a fixed random seed (`np.random.RandomState`) to ensure all evaluations are reproducible and consistent across different environments.

## 🔹 Grid Legend
- `0, 1, 2` → Active Rescue Drones
- `S` → Survivor (Alive & Waiting)
- `R` → Survivor (Successfully Rescued)
- `D` → Survivor (Drowned/Lost)
- `X` → Static Obstacle (Collapsed Building)
- `.` → Empty Cell

## 🔹 Tasks (The 3-Tier Requirement)
- **Task 1: easy_rescue (Easy)**
  - 1 Drone, 2 Survivors, 10x10 grid.
  - Focus: Basic navigation and pathfinding around obstacles.
- **Task 2: medium_coordination (Medium)**
  - 2 Drones, 5 Survivors, 15x15 grid, limited battery.
  - Focus: Multi-agent coordination and energy-efficient path planning.
- **Task 3: hard_dynamic_response (Hard)**
  - 3 Drones, 10 Survivors, 20x20 grid, **Dynamic Flood Spread**.
  - Focus: Real-time risk assessment and rescue optimization under lethal environmental pressure.

##🔹 Final Results (Optimized Multi-Agent System)
- Easy Task: Success 100% | Score: 0.99
- Medium Task: Success 100% | Score: 0.97
- Hard Task: Success 100% | Score: 0.97
🔹 Baseline Results (Before Optimization)
- Hard Task: 0.5500 (demonstrates initial difficulty)
  <img width="1240" height="841" alt="image" src="https://github.com/user-attachments/assets/a92d50bb-e0f6-4f9d-b3ee-c78127c44128" />

The system achieves 100% success across all tasks due to deterministic evaluation and optimized multi-agent coordination strategies, while score differences reflect efficiency under increasing complexity.

## 🔹 Installation & Submission
1. **Local Install:**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
2. **Validation:**
   ```bash
   openenv validate
   ```
3. **Containerization:**
   ```bash
   docker build -t emergency-response .
   ```
