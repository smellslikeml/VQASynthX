import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

"""
This script introduces a prototype for an RL-based task planner, inspired by the 
reinforcement learning concepts in nonstopronald/Reinforcement-Learning. The goal
is to create a simple, simulated environment where an agent learns to manage a 
queue of tasks, analogous to how the source repository optimizes a call center.

This serves as a foundational experiment for integrating decision-making agents 
into the vqasynth ecosystem, which currently focuses on data generation for
spatial awareness. The next step after awareness is action and planning.

The environment `TaskQueueEnv` is a simplified version of `CallCentreEnvironment.py`,
and the training loop uses PPO from `stable-baselines3`, similar to what is
demonstrated in `CallCentreProject.ipynb`.
"""


class TaskQueueEnv(gym.Env):
    """
    A simple task queue environment for a robot.
    - State: The number of tasks currently in the queue.
    - Actions: The agent can choose to process one task.
    - Reward: The agent is rewarded for keeping the queue short and penalized
      for having it overflow.
    - Dynamics: New tasks are added to the queue at each step with a certain probability.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, max_queue_size=20, arrival_prob=0.5):
        super(TaskQueueEnv, self).__init__()

        self.max_queue_size = max_queue_size
        self.arrival_prob = arrival_prob

        # Action: 0 = do nothing, 1 = process a task
        self.action_space = spaces.Discrete(2)

        # Observation: a single value representing the number of tasks in the queue.
        self.observation_space = spaces.Box(
            low=0, high=self.max_queue_size, shape=(1,), dtype=np.float32
        )

        # Initialize state
        self.tasks_in_queue = 0
        self.current_step = 0
        self.max_steps = 200  # An episode lasts for 200 steps

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.tasks_in_queue = 0
        self.current_step = 0
        return np.array([self.tasks_in_queue], dtype=np.float32)

    def step(self, action):
        """
        Execute one time step within the environment
        """
        self.current_step += 1

        # Action 1: Process one task if available
        if action == 1 and self.tasks_in_queue > 0:
            self.tasks_in_queue -= 1
            reward = 5.0  # Reward for completing a task
        elif action == 1 and self.tasks_in_queue == 0:
            reward = -2.0  # Penalty for trying to process a non-existent task
        else:  # Action 0: Do nothing
            reward = 0.0

        # Simulate new tasks arriving
        if np.random.uniform(0, 1) < self.arrival_prob:
            new_tasks = np.random.poisson(1)  # Average of 1 new task
            self.tasks_in_queue += new_tasks

        # Penalty for a long queue
        reward -= self.tasks_in_queue * 0.1

        # Check for overflow (failure condition)
        if self.tasks_in_queue > self.max_queue_size:
            self.tasks_in_queue = self.max_queue_size
            reward -= 50  # Large penalty for overflowing the queue
            done = True
        else:
            done = False

        # Check if episode is over
        if self.current_step >= self.max_steps:
            done = True

        # Cap queue at max size for observation space consistency
        obs = np.array(
            [min(self.tasks_in_queue, self.max_queue_size)], dtype=np.float32
        )

        info = {}

        return obs, reward, done, info

    def render(self, mode="human", close=False):
        """
        Render the environment to the screen
        """
        print(f"Step: {self.current_step}, Tasks in queue: {self.tasks_in_queue}")


def main():
    """
    Main function to create, train, and evaluate the RL agent.
    """
    print("Initializing environment...")
    env = TaskQueueEnv()

    # It's a good practice to check if the environment follows the gym interface
    print("Checking environment compatibility...")
    check_env(env, warn=True)

    # Wrap the environment in a vectorized environment
    env = DummyVecEnv([lambda: env])

    # PPO is a good, general-purpose model-free algorithm.
    # The policy 'MlpPolicy' is suitable for this simple observation space.
    print("Creating PPO agent...")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64, gamma=0.99)

    # Train the agent
    print("Training agent for 20000 timesteps...")
    model.learn(total_timesteps=20000)
    print("Training complete.")

    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    obs = env.reset()
    total_reward = 0
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        env.render()
        if dones:
            break
    print(f"Evaluation finished. Total reward: {total_reward[0]:.2f}")


if __name__ == "__main__":
    main()
