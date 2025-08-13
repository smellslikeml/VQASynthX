import os
import json
import numpy as np
import cv2
import gymnasium as gym

# This script leverages the pre-built environments from the RoboManipBaselines repository.
# To run this, you must first install its dependencies.
# The SOURCE repo (RoboManipBaselines) provides a rich set of simulated environments
# that are perfect for generating ground-truth data for VQASynth.
# We will use the 'MujocoUR5eCableEnv' as it's a well-defined task.

# Ensure the necessary package from the source repo is available
try:
    from robo_manip_baselines.envs.mujoco.ur5e import MujocoUR5eCableEnv
except ImportError:
    print("Error: 'robo_manip_baselines' is not installed.")
    print(
        "Please install it, e.g., via 'pip install git+https://github.com/isri-aist/RoboManipBaselines.git'"
    )
    exit(1)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def generate_simulation_data(output_dir="output/sim_dataset", num_frames=1):
    """
    Initializes a MuJoCo environment from RoboManipBaselines, captures observation data,
    and saves the image and ground-truth state to disk.

    Args:
        output_dir (str): Directory to save the generated data.
        num_frames (int): Number of frames/observations to capture.
    """
    print("Initializing MuJoCo environment: MujocoUR5eCable-v0")

    # The environment is instantiated just like any Gymnasium environment.
    # 'render_mode="rgb_array"' is crucial for capturing images without a display.
    env = gym.make("MujocoUR5eCable-v0", render_mode="rgb_array")

    print("Resetting environment...")
    obs, _ = env.reset()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving data to {output_dir}")

    for i in range(num_frames):
        if i > 0:
            # Take a random action to get a different state
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

        # The observation 'obs' is a dictionary containing the full environment state.
        # It includes the camera image and ground-truth simulation data like joint positions ('qpos').
        # This is the key insight: we get perfect 3D data aligned with 2D images.
        image = obs["image"]

        # The rest of the observation is the ground-truth state vector.
        # We separate the image for saving as a PNG and serialize the rest as JSON.
        state_data = {key: value for key, value in obs.items() if key != "image"}

        frame_id = f"frame_{i:03d}"
        image_path = os.path.join(output_dir, f"{frame_id}.png")
        json_path = os.path.join(output_dir, f"{frame_id}.json")

        # Save the image (OpenCV expects BGR, but env provides RGB, so we convert)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save the ground-truth state data
        with open(json_path, "w") as f:
            json.dump(state_data, f, cls=NumpyEncoder, indent=4)

        print(f"- Saved {image_path} and {json_path}")

    env.close()
    print("\nData generation complete.")
    print("This data can now be used as input for a VQASynth pipeline stage.")


if __name__ == "__main__":
    generate_simulation_data(num_frames=1)
