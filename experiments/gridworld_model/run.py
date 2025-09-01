import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm

# --- Configuration ---
GRID_SIZE = 10
NUM_OBSTACLES = 5
NUM_EPISODES_COLLECT = 500
MAX_STEPS_PER_EPISODE = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PLANNING_HORIZON = 8
PLANNING_CANDIDATES = 64


class GridWorld:
    """A simple 2D GridWorld environment."""

    def __init__(self, size, num_obstacles):
        self.size = size
        self.num_obstacles = num_obstacles
        self.action_space = 4  # 0: up, 1: down, 2: left, 3: right

    def reset(self):
        """Resets the environment to a new random configuration."""
        self.grid = np.zeros((self.size, self.size))

        # Place agent, goal, and obstacles
        positions = random.sample(range(self.size * self.size), 2 + self.num_obstacles)
        self.agent_pos = np.unravel_index(positions[0], (self.size, self.size))
        self.goal_pos = np.unravel_index(positions[1], (self.size, self.size))

        self.grid[self.agent_pos] = 0.8  # Agent marker
        self.grid[self.goal_pos] = 1.0  # Goal marker

        for i in range(self.num_obstacles):
            obstacle_pos = np.unravel_index(positions[2 + i], (self.size, self.size))
            self.grid[obstacle_pos] = 0.4  # Obstacle marker

        return self.get_state()

    def step(self, action):
        """Executes an action and returns the new state."""
        y, x = self.agent_pos
        if action == 0:
            y -= 1  # Up
        elif action == 1:
            y += 1  # Down
        elif action == 2:
            x -= 1  # Left
        elif action == 3:
            x += 1  # Right

        # Check boundaries and obstacles
        if 0 <= y < self.size and 0 <= x < self.size and self.grid[y, x] != 0.4:
            self.grid[self.agent_pos] = 0.0
            self.agent_pos = (y, x)
            self.grid[self.agent_pos] = 0.8

        return self.get_state()

    def get_state(self):
        """Returns the current grid state as a tensor."""
        return torch.from_numpy(self.grid).float().unsqueeze(0)


class WorldModel(nn.Module):
    """A simple CNN to predict the next state given a state and action."""

    def __init__(self, grid_size, num_actions):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, 4)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Project action embedding to match spatial dimensions of CNN output
        self.action_proj = nn.Linear(4, 32 * grid_size * grid_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output values between 0 and 1
        )

    def forward(self, state, action):
        # state: (B, 1, H, W), action: (B,)
        cnn_out = self.cnn(state)

        action_emb = self.action_embedding(action)
        action_map = self.action_proj(action_emb).view(cnn_out.shape)

        combined = torch.cat([cnn_out, action_map], dim=1)

        predicted_next_state = self.decoder(combined)
        return predicted_next_state


def collect_data(env, num_episodes, max_steps):
    """Collects state-action-next_state transitions using a random policy."""
    transitions = []
    print(f"Collecting data from {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        state = env.reset()
        for _ in range(max_steps):
            action = random.randint(0, env.action_space - 1)
            next_state = env.step(action)
            transitions.append((state, torch.tensor([action]), next_state))
            state = next_state
    return transitions


def plan_actions(world_model, start_state, goal_pos, env):
    """Uses the world model to plan a sequence of actions via random shooting."""
    best_action_sequence = None
    best_score = float("inf")

    start_pos = tuple(
        np.unravel_index(start_state.numpy().argmax(), start_state.shape[1:])
    )
    print(f"\nPlanning from {start_pos} to {goal_pos}...")

    for _ in range(PLANNING_CANDIDATES):
        action_sequence = [
            random.randint(0, env.action_space - 1) for _ in range(PLANNING_HORIZON)
        ]

        # 'Imagine' the future using the world model
        imagined_state = start_state.clone()
        for action in action_sequence:
            action_tensor = torch.tensor([action])
            with torch.no_grad():
                imagined_state = world_model(
                    imagined_state.unsqueeze(0), action_tensor
                ).squeeze(0)

        # Score the imagined outcome by distance to goal
        # The agent's position is the brightest spot on the grid
        imagined_pos = np.unravel_index(
            imagined_state.numpy().argmax(), imagined_state.shape[1:]
        )
        dist = np.linalg.norm(np.array(imagined_pos) - np.array(goal_pos))

        if dist < best_score:
            best_score = dist
            best_action_sequence = action_sequence

    print(f"Best imagined sequence leads to distance {best_score:.2f} from goal.")
    return best_action_sequence


if __name__ == "__main__":
    # 1. Setup Environment and Model
    env = GridWorld(GRID_SIZE, NUM_OBSTACLES)
    world_model = WorldModel(grid_size=GRID_SIZE, num_actions=env.action_space)
    optimizer = optim.Adam(world_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 2. Collect Experience Data
    transitions = collect_data(env, NUM_EPISODES_COLLECT, MAX_STEPS_PER_EPISODE)
    states, actions, next_states = zip(*transitions)
    dataset = TensorDataset(
        torch.cat(states), torch.cat(actions), torch.cat(next_states)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Train the World Model
    print(f"\nTraining world model for {NUM_EPOCHS} epochs...")
    world_model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for state_batch, action_batch, next_state_batch in dataloader:
            optimizer.zero_grad()
            pred_next_state = world_model(state_batch, action_batch.squeeze(-1))
            loss = criterion(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")

    # 4. Evaluate Planning with the Trained Model
    world_model.eval()
    start_state = env.reset()
    goal_position = env.goal_pos

    plan = plan_actions(world_model, start_state.squeeze(0), goal_position, env)

    if plan:
        print(f"Executing planned sequence: {plan}")
        current_state = start_state
        for action in plan:
            # Take action in the *real* environment
            env.step(action)

        final_agent_pos = env.agent_pos
        print(
            f"Final agent position: {final_agent_pos}, Goal position: {goal_position}"
        )
        if final_agent_pos == goal_position:
            print("\nSuccess! The agent reached the goal using the planned path.")
        else:
            print("\nFailure. The agent did not reach the goal.")
    else:
        print("Could not find a valid plan.")
