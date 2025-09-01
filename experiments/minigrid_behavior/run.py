import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import imageio
from tqdm import tqdm

# --- Configuration ---
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
TOTAL_TIMESTEPS = 30000
N_ENVS = 4  # Parallel environments
MODEL_PATH = "ppo_minigrid_doorkey.zip"
HEATMAP_PATH = "agent_trajectory_heatmap.png"
GIF_PATH = "agent_trajectory.gif"
EVAL_EPISODES = 50


def train_agent():
    """Trains a PPO agent on the MiniGrid environment."""
    print(f"--- Training agent on {ENV_NAME} for {TOTAL_TIMESTEPS} timesteps ---")

    # Create vectorized environments
    # FullyObsWrapper gives a grid-like observation instead of a partial view
    env = make_vec_env(ENV_NAME, n_envs=N_ENVS, wrapper_class=FullyObsWrapper)

    # PPO model from stable-baselines3
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log="./ppo_minigrid_tensorboard/"
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"--- Model saved to {MODEL_PATH} ---")
    return model


def evaluate_and_visualize(model):
    """Evaluates the trained agent and generates a heatmap and GIF of its trajectory."""
    print(f"--- Evaluating agent for {EVAL_EPISODES} episodes ---")

    # Create a single environment for evaluation with rendering
    eval_env = gym.make(ENV_NAME, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)

    grid_size = (eval_env.unwrapped.width, eval_env.unwrapped.height)
    visitation_counts = np.zeros(grid_size)

    frames = []

    for episode in tqdm(range(EVAL_EPISODES), desc="Evaluating Episodes"):
        obs, _ = eval_env.reset()
        done = False

        # Record frames for GIF only for the first episode to save time
        is_first_episode = episode == 0
        if is_first_episode:
            frames.append(eval_env.render())

        while not done:
            # Record agent's position from the unwrapped environment
            agent_pos = eval_env.unwrapped.agent_pos
            visitation_counts[agent_pos[0], agent_pos[1]] += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            if is_first_episode:
                frames.append(eval_env.render())

    eval_env.close()

    # Generate and save heatmap
    print(f"--- Generating heatmap at {HEATMAP_PATH} ---")
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        visitation_counts.T,  # Transpose to match grid layout (x, y)
        cmap="viridis",
        annot=False,
        cbar=True,
    )
    ax.set_title(
        f"Agent State Visitation Frequency\n{ENV_NAME} - {EVAL_EPISODES} episodes"
    )
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    plt.savefig(HEATMAP_PATH)
    plt.close()

    # Generate and save GIF
    print(f"--- Generating GIF at {GIF_PATH} ---")
    imageio.mimsave(GIF_PATH, frames, fps=10)

    print("--- Evaluation and visualization complete ---")


def main():
    """Main function to run training and evaluation."""
    # Create a directory for outputs if it doesn't exist
    os.makedirs("experiments/minigrid_behavior/outputs", exist_ok=True)
    global MODEL_PATH, HEATMAP_PATH, GIF_PATH
    MODEL_PATH = os.path.join(
        "experiments/minigrid_behavior/outputs", os.path.basename(MODEL_PATH)
    )
    HEATMAP_PATH = os.path.join(
        "experiments/minigrid_behavior/outputs", os.path.basename(HEATMAP_PATH)
    )
    GIF_PATH = os.path.join(
        "experiments/minigrid_behavior/outputs", os.path.basename(GIF_PATH)
    )

    if os.path.exists(MODEL_PATH):
        print(f"--- Loading pre-trained model from {MODEL_PATH} ---")
        model = PPO.load(MODEL_PATH)
    else:
        model = train_agent()

    evaluate_and_visualize(model)


if __name__ == "__main__":
    main()
