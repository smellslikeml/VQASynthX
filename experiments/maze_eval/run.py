import json

class MazeEnvironment:
    """
    A Python implementation of the maze navigation environment from
    https://github.com/haffi112/maze-navigation-game.
    This environment is designed to test an agent's non-visual spatial
    reasoning and sequential decision-making.
    """
    def __init__(self, maze_data):
        self.width = maze_data['width']
        self.height = maze_data['height']
        self.start_pos = tuple(maze_data['start'])
        self.goal_pos = tuple(maze_data['goal'])
        self.cells = maze_data['cells']

        # Game state
        self.agent_pos = self.start_pos
        self.moves = 0
        self.max_moves = self.width * self.height * 2 # A reasonable limit
        self.visited = {self.agent_pos: 1}
        self.is_finished = False
        self.path = [self.start_pos]

    def get_observation(self):
        """
        Calculates the agent's current limited visibility, mimicking game.js.
        Returns a dictionary of how far the agent can "see" in each direction.
        """
        if self.is_finished:
            return None

        x, y = self.agent_pos
        visibility = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        directions = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }

        for name, (dx, dy) in directions.items():
            dist = 0
            cx, cy = x, y
            while True:
                # Check bounds first
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    break
                
                current_cell_walls = self.cells[cy][cx]['walls']
                if current_cell_walls[name]:
                    break
                
                # Move to next cell in this direction for visibility check
                cx += dx
                cy += dy
                dist += 1
            visibility[name] = dist
        
        return {
            "current_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "visibility": visibility,
            "moves_made": self.moves,
        }

    def step(self, action: str):
        """
        Takes an action ('north', 'south', 'east', 'west') and updates the state.
        Returns observation, reward, done, info.
        """
        if self.is_finished:
            raise Exception("Game is already finished.")

        action = action.lower()
        directions = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }

        if action not in directions:
            raise ValueError(f"Invalid action: {action}")

        x, y = self.agent_pos
        current_cell_walls = self.cells[y][x]['walls']

        if current_cell_walls[action]:
            # This case should ideally be prevented by a smart agent
            # but we handle it for robustness.
            self.moves += 1
            self.is_finished = True
            return self.get_observation(), -10, True, {"status": "hit_wall"}

        dx, dy = directions[action]
        new_pos = (x + dx, y + dy)
        
        self.moves += 1
        self.agent_pos = new_pos
        self.path.append(new_pos)
        self.visited[new_pos] = self.visited.get(new_pos, 0) + 1

        reward = -1 # Cost for each move
        done = False
        status = "in_progress"
        
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            status = "success"
        elif self.moves >= self.max_moves:
            reward = -10
            done = True
            status = "max_moves_exceeded"
            
        self.is_finished = done
        return self.get_observation(), reward, done, {"status": status}

class GreedyAgent:
    """A simple greedy agent that tries to reduce Manhattan distance to the goal."""
    def __init__(self, goal):
        self.goal = goal

    def act(self, observation):
        pos = observation['current_pos']
        visibility = observation['visibility']
        
        best_action = None
        min_dist = float('inf')

        possible_moves = {
            'north': (pos[0], pos[1] - 1),
            'south': (pos[0], pos[1] + 1),
            'east': (pos[0] + 1, pos[1]),
            'west': (pos[0] - 1, pos[1]),
        }
        
        # Find the valid move that gets closest to the goal
        for action, is_visible in visibility.items():
            if is_visible > 0:
                next_pos = possible_moves[action]
                dist = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
                if dist < min_dist:
                    min_dist = dist
                    best_action = action

        return best_action

def get_test_maze():
    """A simple 3x3 maze for testing. Optimal path is 4 moves: S, S, E, E."""
    # S . .
    # | . .
    # . _ G
    return {
        "width": 3, "height": 3,
        "start": [0, 0], "goal": [2, 2],
        "cells": [
            # y=0
            [
                {"walls": {'north': True, 'south': False, 'east': True, 'west': True}},
                {"walls": {'north': True, 'south': False, 'east': False, 'west': True}},
                {"walls": {'north': True, 'south': False, 'east': True, 'west': False}},
            ],
            # y=1
            [
                {"walls": {'north': False, 'south': False, 'east': True, 'west': True}},
                {"walls": {'north': False, 'south': True, 'east': False, 'west': True}},
                {"walls": {'north': False, 'south': False, 'east': True, 'west': False}},
            ],
            # y=2
            [
                {"walls": {'north': False, 'south': True, 'east': False, 'west': True}},
                {"walls": {'north': True, 'south': True, 'east': False, 'west': False}},
                {"walls": {'north': False, 'south': True, 'east': True, 'west': False}},
            ],
        ],
    }

def main():
    """
    Main function to run the maze navigation experiment.
    This demonstrates the environment with a simple rule-based agent.
    """
    print("--- Maze Navigation Benchmark ---")
    maze_data = get_test_maze()
    env = MazeEnvironment(maze_data)
    agent = GreedyAgent(env.goal_pos)
    
    obs = env.get_observation()
    done = False
    
    while not done:
        print(f"\n--- Turn {env.moves + 1} ---")
        print(f"Agent at {obs['current_pos']}. Visibility: {obs['visibility']}")
        
        # Get action from agent (this is where an LLM call would go)
        action = agent.act(obs)
        print(f"Agent action: {action}")

        obs, reward, done, info = env.step(action)
        
    print("\n--- Experiment Finished ---")
    print(f"Status: {info['status']}")
    print(f"Total Moves: {env.moves}")
    print(f"Final Position: {env.agent_pos}")
    print(f"Path taken: {env.path}")

    success = info['status'] == 'success'
    print(f"\nSuccess: {success}")


if __name__ == "__main__":
    main()
