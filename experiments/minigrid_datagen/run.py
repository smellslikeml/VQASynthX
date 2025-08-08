import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import pandas as pd
import numpy as np
import os
import argparse
from PIL import Image

def generate_trajectory_data(env_name, num_steps, seed, output_dir):
    """
    Generates agent trajectory data from a MiniGrid environment.

    An agent explores the environment for a fixed number of steps, and at each step,
    its observation (image) and position (x, y) are recorded. This is inspired by
    the data collection process in lyonva/bad-apple's oracle.py and test.py,
    which are used to analyze agent behavior.
    """
    print(f"Initializing environment: {env_name} with seed {seed}")
    env = gym.make(env_name)
    # The ImgObsWrapper provides the pixel-based observations needed for VQA.
    # FullyObsWrapper gives the agent a full view of the grid, simplifying navigation.
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, info = env.reset(seed=seed)

    # Create output directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    trajectory_log = []

    print(f"Starting data generation for {num_steps} steps...")
    for step in range(num_steps):
        # For this minimal experiment, we use a random action policy.
        # This simulates exploration to collect diverse spatial data.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # Agent position is crucial for spatial reasoning.
        agent_pos = info['agent_pos']

        # Save the frame as an image file.
        frame_filename = f"frame_{step:05d}.png"
        frame_path = os.path.join(image_output_dir, frame_filename)
        img = Image.fromarray(obs)
        img.save(frame_path)

        trajectory_log.append({
            "step": step,
            "image_path": os.path.join("images", frame_filename),
            "agent_pos_x": int(agent_pos[0]),
            "agent_pos_y": int(agent_pos[1]),
            "action": int(action),
            "terminated": terminated
        })

        if terminated or truncated:
            print(f"Episode finished at step {step}. Resetting environment.")
            obs, info = env.reset()

    # Save trajectory metadata to a CSV file
    df = pd.DataFrame(trajectory_log)
    csv_path = os.path.join(output_dir, "trajectory.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Trajectory log saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory data from MiniGrid.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-DoorKey-16x16-v0",
        help="Name of the MiniGrid environment."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Total number of steps to run the agent."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the environment."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/minigrid_trajectory_data",
        help="Directory to save the output images and CSV."
    )
    args = parser.parse_args()

    generate_trajectory_data(args.env_name, args.num_steps, args.seed, args.output_dir)
