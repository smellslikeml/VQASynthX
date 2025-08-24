import json

# Maze data structure adapted from maze_data.js
# This represents the first 5x5 maze from the source.
MAZE_DATA = {
    "width": 5,
    "height": 5,
    "start": [0, 2],
    "goal": [4, 2],
    "cells": [
        [
            {"walls": {"north": True, "south": False, "east": True, "west": True}},
            {"walls": {"north": True, "south": True, "east": True, "west": True}},
            {"walls": {"north": True, "south": False, "east": True, "west": True}},
            {"walls": {"north": True, "south": True, "east": True, "west": True}},
            {"walls": {"north": True, "south": False, "east": True, "west": True}},
        ],
        [
            {"walls": {"north": False, "south": False, "east": False, "west": True}},
            {"walls": {"north": True, "south": False, "east": True, "west": False}},
            {"walls": {"north": False, "south": False, "east": False, "west": True}},
            {"walls": {"north": True, "south": False, "east": True, "west": False}},
            {"walls": {"north": False, "south": False, "east": True, "west": True}},
        ],
        [
            {"walls": {"north": False, "south": False, "east": False, "west": True}},
            {"walls": {"north": False, "south": False, "east": False, "west": False}},
            {"walls": {"north": False, "south": False, "east": False, "west": False}},
            {"walls": {"north": False, "south": False, "east": False, "west": False}},
            {"walls": {"north": False, "south": False, "east": True, "west": False}},
        ],
        [
            {"walls": {"north": False, "south": True, "east": False, "west": True}},
            {"walls": {"north": False, "south": True, "east": True, "west": False}},
            {"walls": {"north": False, "south": False, "east": False, "west": True}},
            {"walls": {"north": False, "south": True, "east": True, "west": False}},
            {"walls": {"north": False, "south": True, "east": True, "west": True}},
        ],
        [
            {"walls": {"north": True, "south": True, "east": False, "west": True}},
            {"walls": {"north": True, "south": True, "east": False, "west": False}},
            {"walls": {"north": False, "south": True, "east": False, "west": False}},
            {"walls": {"north": True, "south": True, "east": False, "west": False}},
            {"walls": {"north": True, "south": True, "east": True, "west": False}},
        ],
    ],
}


class MazeEnvironment:
    """
    A Python implementation of the maze navigation environment from the SOURCE repo.
    This environment provides text-based observations for an LLM agent to navigate a maze.
    """

    def __init__(self, maze_data):
        self.width = maze_data["width"]
        self.height = maze_data["height"]
        self.start_pos = tuple(maze_data["start"])
        self.goal_pos = tuple(maze_data["goal"])
        self.cells = maze_data["cells"]

        self.player_pos = self.start_pos
        self.moves = 0
        self.visited = {self.player_pos}
        self.done = False

    def reset(self):
        """Resets the environment to the initial state."""
        self.player_pos = self.start_pos
        self.moves = 0
        self.visited = {self.player_pos}
        self.done = False
        return self.get_observation()

    def get_observation(self):
        """
        Generates a text-based observation of the current state, mimicking the
        limited visibility from the source game's game.js. The agent can "see"
        how many steps it can travel in each cardinal direction.
        """
        if self.done:
            return f"Goal reached in {self.moves} moves!"

        x, y = self.player_pos
        observation = (
            f"You are at position ({x}, {y}). The goal is at {self.goal_pos}.\n"
        )
        observation += "From your current location, you can see:\n"

        directions = {
            "north": {"dx": 0, "dy": -1},
            "south": {"dx": 0, "dy": 1},
            "east": {"dx": 1, "dy": 0},
            "west": {"dx": -1, "dy": 0},
        }

        view_distances = {}
        for name, move in directions.items():
            dist = 0
            curr_x, curr_y = x, y
            while True:
                current_cell = self.cells[curr_y][curr_x]
                if current_cell["walls"][name]:
                    break

                curr_x += move["dx"]
                curr_y += move["dy"]

                if not (0 <= curr_x < self.width and 0 <= curr_y < self.height):
                    break

                dist += 1

            view_distances[name] = dist

        observation += f"- North: {view_distances['north']} steps\n"
        observation += f"- South: {view_distances['south']} steps\n"
        observation += f"- East: {view_distances['east']} steps\n"
        observation += f"- West: {view_distances['west']} steps\n"

        return observation

    def step(self, action):
        """
        Executes an action, updates the environment state, and returns the new observation.
        :param action: A string, one of ['north', 'south', 'east', 'west'].
        :return: A tuple of (observation, reward, done).
        """
        if self.done:
            return self.get_observation(), 0, self.done

        x, y = self.player_pos
        current_cell = self.cells[y][x]

        action_map = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
        }

        if action not in action_map:
            return (
                f"Invalid action: {action}. Choose from {list(action_map.keys())}",
                -1,
                self.done,
            )

        if not current_cell["walls"][action]:
            dx, dy = action_map[action]
            new_x, new_y = x + dx, y + dy
            self.player_pos = (new_x, new_y)
            self.visited.add(self.player_pos)
            self.moves += 1
        else:
            return (
                f"You hit a wall at position {self.player_pos} trying to move {action}.",
                -1,
                self.done,
            )

        reward = -1  # Cost for each move
        if self.player_pos == self.goal_pos:
            self.done = True
            reward = 100  # Positive reward for reaching the goal

        return self.get_observation(), reward, self.done


if __name__ == "__main__":
    print("--- Initializing Maze Environment ---")
    env = MazeEnvironment(MAZE_DATA)
    obs = env.reset()
    print("Initial Observation:")
    print(obs)

    # This demonstrates how an agent (e.g., an LLM) would interact with the environment.
    # The agent receives an observation and chooses an action.
    actions = ["north", "north", "east", "east", "south", "south", "east", "east"]

    print("\n--- Starting Agent Simulation ---")
    for i, action in enumerate(actions):
        if env.done:
            print("Agent has reached the goal. Halting simulation.")
            break

        print(f"\nStep {i+1}: Agent chooses to move '{action}'")
        obs, reward, done = env.step(action)

        print(f"Observation after move:")
        print(obs)
        print(f"Reward: {reward}, Done: {done}")

    print("\n--- Simulation Finished ---")
    if env.done:
        print(f"Success! Maze solved in {env.moves} moves.")
    else:
        print(f"Agent did not reach the goal. Final position: {env.player_pos}")
