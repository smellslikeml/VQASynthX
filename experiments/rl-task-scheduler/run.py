import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from robot_env import RobotTaskEnv

def main():
    """
    Trains and evaluates a PPO agent on the RobotTaskEnv.
    This script is inspired by the PPO implementation in CallCentreProject.ipynb.
    """
    print("Initializing Robot Task Scheduling Environment...")
    env = RobotTaskEnv(num_task_types=3, max_queue_size=10)
    
    print("Checking environment compatibility...")
    try:
        check_env(env, warn=True)
        print("Environment check passed.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    log_dir = "./rl_scheduler_logs/"
    model_dir = "./rl_scheduler_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Instantiating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    training_timesteps = 20000
    print(f"Starting training for {training_timesteps} timesteps...")
    model.learn(total_timesteps=training_timesteps, progress_bar=True)
    print("Training complete.")

    model_path = os.path.join(model_dir, "ppo_robot_scheduler.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    print("\nEvaluating trained policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation Results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    print("\n--- Demonstrating loaded model ---")
    del model
    
    loaded_model = PPO.load(model_path, env=env)
    obs = env.reset()
    for i in range(200):
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if i % 20 == 0:
            env.render()
        if dones:
            obs = env.reset()
    
    print("\nExperiment finished successfully.")

if __name__ == '__main__':
    main()
