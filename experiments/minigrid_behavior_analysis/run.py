import gymnasium as gym
import minigrid
from minigrid.core.actions import Actions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from collections import deque

"""
This script integrates behavioral analysis concepts from the 'Minding Motivation'
repository (lyonva/bad-apple) into the VQASynth project. It establishes a
minimal testbed for analyzing agent navigation patterns in a controlled
environment.

The core idea is to run an agent in a MiniGrid environment, record its
trajectory, and visualize it as a heatmap. This provides a tangible way
to assess an agent's exploration strategy and task completion, a methodology
central to the SOURCE repository's analysis of intrinsic motivation.

This initial experiment uses a simple "Oracle" agent that has perfect
knowledge of the environment to find the shortest path, demonstrating the
analysis pipeline.
"""

class OracleAgent:
    """A scripted agent that solves the DoorKey task using BFS to find actions."""
    def __init__(self, env):
        self.env = env
        self.action_plan = []

    def plan_path(self):
        """Creates a full action plan to solve the environment."""
        # Find key, door, and goal positions
        key_pos, door_pos, goal_pos = None, None, None
        for pos, obj in self.env.grid.iter_cells():
            if obj:
                if obj.type == 'key': key_pos = pos
                elif obj.type == 'door': door_pos = pos
                elif obj.type == 'goal': goal_pos = pos

        # Plan path segments
        path_to_key = self._bfs_to_target(self.env.agent_pos, self.env.agent_dir, key_pos)
        # State after picking up key
        state_after_key = (key_pos, path_to_key[-2].new_dir if len(path_to_key) > 1 else self.env.agent_dir)

        path_to_door = self._bfs_to_target(state_after_key[0], state_after_key[1], door_pos, has_key=True)
        # State after opening door
        state_after_door = (door_pos, path_to_door[-2].new_dir if len(path_to_door) > 1 else state_after_key[1])

        path_to_goal = self._bfs_to_target(state_after_door[0], state_after_door[1], goal_pos, has_key=True)

        self.action_plan = (
            [a.action for a in path_to_key] + [Actions.pickup] +
            [a.action for a in path_to_door] + [Actions.toggle] +
            [a.action for a in path_to_goal]
        )
        return self.action_plan

    def _bfs_to_target(self, start_pos, start_dir, target_pos, has_key=False):
        """BFS to find a sequence of actions to reach a target."""
        class Node: 
            def __init__(self, pos, direction, action, parent=None, new_dir=None):
                self.pos = pos; self.dir = direction; self.action = action; self.parent = parent; self.new_dir = new_dir if new_dir is not None else direction

        q = deque([Node(start_pos, start_dir, None)])
        visited = {(tuple(start_pos), start_dir)}

        while q:
            node = q.popleft()

            if tuple(node.pos) == tuple(target_pos):
                path = []
                while node.parent is not None:
                    path.append(node)
                    node = node.parent
                return path[::-1]

            # Try actions: turn left, turn right, move forward
            # Turn Left
            new_dir_left = (node.dir - 1) % 4
            if (tuple(node.pos), new_dir_left) not in visited:
                visited.add((tuple(node.pos), new_dir_left))
                q.append(Node(node.pos, node.dir, Actions.left, node, new_dir=new_dir_left))
            
            # Turn Right
            new_dir_right = (node.dir + 1) % 4
            if (tuple(node.pos), new_dir_right) not in visited:
                visited.add((tuple(node.pos), new_dir_right))
                q.append(Node(node.pos, node.dir, Actions.right, node, new_dir=new_dir_right))

            # Move Forward
            fwd_pos = tuple(node.pos + self.env.dir_vecs[node.dir])
            if (fwd_pos, node.dir) not in visited:
                cell = self.env.grid.get(*fwd_pos)
                if cell is None or cell.can_pass() or (cell.type == 'door' and has_key):
                    visited.add((fwd_pos, node.dir))
                    q.append(Node(fwd_pos, node.dir, Actions.forward, node))
        return []


def run_experiment(env_name, seed, steps, output_dir):
    """Runs the agent, collects data, and generates a heatmap."""
    print(f"Initializing environment '{env_name}' with seed {seed}.")
    env = gym.make(env_name)
    env.reset(seed=seed)

    agent = OracleAgent(env)
    action_plan = agent.plan_path()
    
    print(f"Oracle agent planned a path of {len(action_plan)} actions.")

    # Re-initialize env for execution
    obs, info = env.reset(seed=seed)
    positions = []
    for i, action in enumerate(action_plan):
        if i >= steps: 
            print(f"Reached max steps ({steps})."); break
        
        positions.append(env.agent_pos)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated: 
            print(f"Episode finished after {i+1} steps."); break
    
    positions.append(env.agent_pos)
    env.close()

    if not positions: 
        print("No positions recorded."); return

    # Generate Heatmap
    grid_size_x, grid_size_y = env.width, env.height
    heatmap = np.zeros((grid_size_y, grid_size_x))
    for x, y in positions:
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            heatmap[y, x] += 1

    plt.figure(figsize=(8, 8))
    sns.heatmap(heatmap, cmap="viridis", linewidths=.5, annot=False, cbar=True)
    plt.title(f"Agent Visitation Heatmap\nEnv: {env_name}, Seed: {seed}")
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"heatmap_{env_name.replace('/', '_')}_seed{seed}.png")
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Name of the MiniGrid environment.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the environment.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum number of steps to run the agent.")
    parser.add_argument("--output_dir", type=str, default="experiment_outputs", help="Directory to save the heatmap.")
    args = parser.parse_args()
    
    run_experiment(args.env_name, args.seed, args.steps, args.output_dir)