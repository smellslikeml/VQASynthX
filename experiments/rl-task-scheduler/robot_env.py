import gym
from gym import spaces
import numpy as np
from collections import deque

class RobotTaskEnv(gym.Env):
    """
    A simplified environment for a robot learning to schedule tasks,
    inspired by the CallCentreEnvironment from the SOURCE repository.

    - State: A vector representing the number of pending tasks for each type.
    - Action: A discrete action to choose which task type to work on next.
    - Reward: Positive reward for completing a task, negative for being idle
              while tasks are pending.
    """
    def __init__(self, num_task_types=3, max_queue_size=10, arrival_prob=0.3):
        super(RobotTaskEnv, self).__init__()

        self.num_task_types = num_task_types
        self.max_queue_size = max_queue_size
        self.arrival_prob = arrival_prob

        # Action: choose which task type to work on. Action 0 means be idle.
        self.action_space = spaces.Discrete(self.num_task_types + 1)
        
        # Observation: number of tasks in each queue
        self.observation_space = spaces.Box(
            low=0, high=self.max_queue_size, 
            shape=(self.num_task_types,), dtype=np.float32
        )

        # Task processing times (mean, std_dev)
        self.task_times = {i: (5 + i*2, 1) for i in range(self.num_task_types)}
        
        self.reset()

    def _get_obs(self):
        return np.array([len(q) for q in self.task_queues], dtype=np.float32)

    def reset(self):
        # Queues for each task type
        self.task_queues = [deque() for _ in range(self.num_task_types)]
        
        self.robot_busy_until = 0
        self.current_time = 0
        self.total_reward = 0
        
        # Pre-populate with a few tasks
        for i in range(self.num_task_types):
            self.task_queues[i].append(self.current_time)

        return self._get_obs()

    def step(self, action):
        self.current_time += 1
        reward = 0
        done = False

        # 1. Handle new task arrivals
        if np.random.uniform() < self.arrival_prob:
            task_type = np.random.randint(0, self.num_task_types)
            if len(self.task_queues[task_type]) < self.max_queue_size:
                self.task_queues[task_type].append(self.current_time)

        # 2. Check if the robot is idle
        is_idle = self.current_time >= self.robot_busy_until
        
        # 3. Process the chosen action if the robot is idle
        if is_idle:
            # Action 0 is 'wait/idle'
            if action == 0:
                # Penalize for being idle if tasks are waiting
                if sum(len(q) for q in self.task_queues) > 0:
                    reward -= 1.0
            
            # Actions 1 to N correspond to task types 0 to N-1
            else:
                task_choice = action - 1
                if task_choice < self.num_task_types and len(self.task_queues[task_choice]) > 0:
                    # Start the task
                    self.task_queues[task_choice].popleft()
                    
                    # Calculate service time
                    mean_time, std_dev = self.task_times[task_choice]
                    service_time = int(np.random.normal(mean_time, std_dev))
                    self.robot_busy_until = self.current_time + max(1, service_time)
                    
                    # Reward for starting a task
                    reward += 10.0
                else:
                    # Penalize for choosing an unavailable task
                    reward -= 5.0
        
        # Small penalty for time passing
        reward -= 0.1

        # Check for termination condition (e.g., time limit)
        if self.current_time > 1000:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        queues_str = ', '.join([f'T{i}:{len(q)}' for i, q in enumerate(self.task_queues)])
        status = "IDLE" if self.current_time >= self.robot_busy_until else f"BUSY (until T={self.robot_busy_until})"
        print(f"Time: {self.current_time} | Queues: [{queues_str}] | Robot Status: {status}")
