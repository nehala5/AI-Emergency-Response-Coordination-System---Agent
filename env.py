import numpy as np
from typing import List, Tuple, Dict
from models import Observation, Action, Reward
from tasks import TaskConfig, get_task_config

class DisasterResponseEnv:
    def __init__(self, task_level: str = "easy"):
        self.config: TaskConfig = get_task_config(task_level)
        self.rng = np.random.RandomState(self.config.seed)
        self.reset()

    def reset(self) -> Observation:
        self.grid_size = self.config.grid_size
        self.num_drones = self.config.num_drones
        self.batteries = [self.config.battery_limit] * self.num_drones
        
        # Initial positions
        self.agent_positions = [(0, 0) for _ in range(self.num_drones)]
        
        # Survivors: (x, y, status)
        self.survivors = []
        for _ in range(self.config.num_survivors):
            x = self.rng.randint(0, self.grid_size[0])
            y = self.rng.randint(0, self.grid_size[1])
            self.survivors.append([x, y, "alive"])
            
        # Flood map (0.0 to 1.0)
        self.flood_map = np.zeros(self.grid_size)
        self.steps = 0
        self.rescued_count = 0
        self.drowned_count = 0
        
        return self.get_observation()

    def state(self) -> Observation:
        return self.get_observation()

    def render(self):
        grid = np.full(self.grid_size, ".", dtype=str)
        # Add survivors
        for x, y, status in self.survivors:
            if status == "alive":
                grid[x, y] = "S"
            elif status == "rescued":
                grid[x, y] = "R"
            elif status == "drowned":
                grid[x, y] = "D"
        # Add drones
        for i, (x, y) in enumerate(self.agent_positions):
            grid[x, y] = str(i)
        
        print(f"\nStep: {self.steps} | Rescued: {self.rescued_count} | Drowned: {self.drowned_count}")
        for row in grid.T[::-1]: # Flip for proper Y-axis
            print(" ".join(row))
        print("-" * (self.grid_size[0] * 2))

    def get_observation(self) -> Observation:
        survivors_list = [(s[0], s[1], s[2]) for s in self.survivors]
        return Observation(
            agent_positions=self.agent_positions,
            batteries=self.batteries,
            survivors=survivors_list,
            flood_map=self.flood_map.tolist()
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        self.steps += 1
        reward_score = 0.0
        feedback = []

        # 1. Update Battery and Moves
        # Moves: 0: Stay, 1: N, 2: S, 3: E, 4: W, 5: NE, 6: NW, 7: SE, 8: SW
        move_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0), 
                    5: (1, 1), 6: (-1, 1), 7: (1, -1), 8: (-1, -1)}

        for i in range(self.num_drones):
            if self.batteries[i] <= 0:
                feedback.append(f"Drone {i} battery depleted.")
                continue

            move = move_map.get(action.moves[i], (0, 0))
            new_x = max(0, min(self.grid_size[0] - 1, self.agent_positions[i][0] + move[0]))
            new_y = max(0, min(self.grid_size[1] - 1, self.agent_positions[i][1] + move[1]))
            
            self.agent_positions[i] = (new_x, new_y)
            self.batteries[i] -= 1
            if self.batteries[i] == 0:
                reward_score -= 0.05

        # 2. Rescue Logic
        for i in range(self.num_drones):
            drone_pos = self.agent_positions[i]
            for s_idx, s in enumerate(self.survivors):
                if s[2] == "alive" and (s[0], s[1]) == drone_pos:
                    s[2] = "rescued"
                    self.rescued_count += 1
                    reward_score += (1.0 / self.config.num_survivors)
                    feedback.append(f"Survivor rescued at {drone_pos}!")

        # 3. Flood Expansion
        if self.config.rising_flood:
            # Simple flood logic: random spread or rising from bottom
            # For simplicity: increase flood level across whole map or specific points
            self.flood_map += (self.config.flood_rate / 10.0) # Slow rise
            self.flood_map = np.clip(self.flood_map, 0.0, 1.0)
            
            # Check for drowning
            for s in self.survivors:
                if s[2] == "alive" and self.flood_map[s[0], s[1]] > 0.8:
                    s[2] = "drowned"
                    self.drowned_count += 1
                    reward_score -= 0.1
                    feedback.append(f"Survivor drowned at ({s[0]}, {s[1]}).")

        # 4. Grading logic penalties
        if self.steps % 10 == 0:
            reward_score -= 0.01

        # Check Done
        done = (self.rescued_count + self.drowned_count == self.config.num_survivors) or \
               (all(b <= 0 for b in self.batteries)) or \
               (self.steps >= 200)

        reward = Reward(score=reward_score, feedback="; ".join(feedback) if feedback else "Continuing...")
        return self.get_observation(), reward, done

    def get_final_score(self) -> float:
        # deterministic grading logic
        base_score = self.rescued_count / self.config.num_survivors
        penalties = (self.drowned_count * 0.1) + \
                    ( (self.steps // 10) * 0.01 ) + \
                    ( sum(1 for b in self.batteries if b <= 0) * 0.05 )
        return max(0.0, base_score - penalties)
