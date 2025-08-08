import gym
import numpy as np
from gym import spaces

class GridWorldEnv(gym.Env):
    """A simple grid world environment, inspired by the structure of CallCentreEnvironment.py."""
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observations: Agent's (row, col) position
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        self.agent_pos = None
        self.goal_pos = None
        self.obstacle_pos = None
        self.max_steps = 50
        self.current_step = 0
        
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        self.goal_pos = np.random.randint(0, self.grid_size, size=2)
        self.obstacle_pos = np.random.randint(0, self.grid_size, size=2)

        # Ensure start, goal, and obstacle are not at the same position
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.grid_size, size=2)
        while np.array_equal(self.agent_pos, self.obstacle_pos) or np.array_equal(self.goal_pos, self.obstacle_pos):
            self.obstacle_pos = np.random.randint(0, self.grid_size, size=2)
        
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        return self.agent_pos

    def step(self, action):
        """Executes one time step within the environment."""
        if action == 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2:  # Left
            self.agent_pos[1] -= 1
        elif action == 3:  # Right
            self.agent_pos[1] += 1

        # Clip to stay within grid boundaries
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        self.current_step += 1
        done = False
        reward = -0.1  # Small penalty for each step to encourage efficiency

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            done = True
        elif np.array_equal(self.agent_pos, self.obstacle_pos):
            reward = -5.0
            done = True
        
        if self.current_step >= self.max_steps:
            done = True
        
        info = {'goal_pos': self.goal_pos, 'obstacle_pos': self.obstacle_pos}

        return self._get_observation(), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        
        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        grid[tuple(self.goal_pos)] = 'G'
        grid[tuple(self.obstacle_pos)] = 'X'
        grid[tuple(self.agent_pos)] = 'A'
        
        for row in grid:
            print(' '.join(row))
        print('\n')

if __name__ == '__main__':
    # Example of running the environment with a random agent
    env = GridWorldEnv()
    for episode in range(3):
        obs = env.reset()
        done = False
        total_reward = 0
        print(f"--- Episode {episode + 1} ---")
        env.render()
        
        while not done:
            action = env.action_space.sample() # Random action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            print(f"Action: {action}, Reward: {reward:.1f}, Done: {done}")

        print(f"Episode finished! Total Reward: {total_reward:.1f}\n")
