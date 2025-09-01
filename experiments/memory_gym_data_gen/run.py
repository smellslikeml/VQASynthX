import gymnasium as gym
import memory_gym
import os
import json
import numpy as np
from PIL import Image


def generate_episode_data(env_name="MortarMayhem-v0", output_dir="episode_data"):
    """
    Runs one episode of a memory_gym environment and saves observations and metadata.
    """
    print(f"Initializing environment: {env_name}")
    # The environment returns rgb_array observations by default
    env = gym.make(env_name)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    print(f"Resetting environment and starting episode...")
    obs, info = env.reset(seed=42)

    done = False
    truncated = False
    step_count = 0
    metadata = []

    # Save initial observation
    img = Image.fromarray(obs)
    img.save(os.path.join(frames_dir, f"frame_{step_count:04d}.png"))
    metadata.append(info)

    while not done and not truncated:
        action = env.action_space.sample()  # Use a random agent for now
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        # Save frame as PNG
        img = Image.fromarray(obs)
        img.save(os.path.join(frames_dir, f"frame_{step_count:04d}.png"))

        # Convert numpy types in info for JSON serialization
        serializable_info = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                serializable_info[k] = v.tolist()
            elif isinstance(
                v,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                serializable_info[k] = int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                serializable_info[k] = float(v)
            else:
                serializable_info[k] = v
        metadata.append(serializable_info)

        if (step_count) % 50 == 0:
            print(f"  ... step {step_count}")

    env.close()

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode finished after {step_count} steps.")
    print(f"Data saved to {output_dir}")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Metadata: {metadata_path}")


if __name__ == "__main__":
    generate_episode_data()
