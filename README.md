# AI Emergency Response Coordination System (OpenEnv)

## 🔹 Problem Statement
In the immediate aftermath of natural disasters like floods, rapid and coordinated search-and-rescue is critical. This environment simulates autonomous drone coordination in a dynamic disaster zone. Drones must navigate complex grids to rescue survivors while managing limited battery life and responding to rising flood levels that threaten lives in real-time.

## 🔹 Tasks (The 3-Tier Requirement)
- **Task 1: easy_rescue (Easy)**
  - 1 Drone, 2 Survivors, 10x10 static grid.
  - Focus: Basic navigation and pathfinding.
- **Task 2: medium_coordination (Medium)**
  - 2 Drones, 5 Survivors, 15x15 grid, limited battery.
  - Focus: Multi-agent coordination and energy management.
- **Task 3: hard_dynamic_response (Hard)**
  - 3 Drones, 10 Survivors, 20x20 grid, rising flood.
  - Focus: Real-time risk assessment and rescue optimization under lethal environmental pressure.

## 🔹 Observation Space
The `Observation` model includes:
- `agent_positions`: Current (x, y) coordinates for all drones.
- `batteries`: Current energy levels for each drone.
- `survivors`: List of survivors (x, y) and their status (`alive`, `rescued`, `drowned`).
- `flood_map`: A grid representing current water levels (0.0 to 1.0).

## 🔹 Action Space (0-8)
Drones can perform 9 discrete actions:
- `0`: Stay
- `1-4`: Cardinal moves (North, South, East, West)
- `5-8`: Diagonal moves (NE, NW, SE, SW)

## 🔹 Reward & Grading Logic
The environment uses a **Deterministic Grading Logic** for consistent scoring:
- `Base Score`: `Rescued / Total Survivors`
- `Penalty (Drowning)`: `-0.1` per Drowned survivor.
- `Penalty (Time)`: `-0.01` per 10 steps taken.
- `Penalty (Battery)`: `-0.05` per Battery depletion.

## 🔹 Baseline Results
Running the provided `main.py` baseline policy:
- **Easy:** 0.9800
- **Medium:** 0.9600
- **Hard:** 0.5500

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
