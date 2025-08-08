import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import numpy as np
import argparse
from tqdm import tqdm
import warnings

# Suppress Opacus UserWarnings about insecure PRNG
warnings.filterwarnings("ignore", category=UserWarning, module='opacus')

# --- 1. Q-Learning Agent ---
# A simple implementation of a Q-learning agent to control the DP noise multiplier.
# This is a simplified version of the SAC hyper-policy from the SOURCE repo (RLDP).
class QLearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.5, exploration_decay=0.99):
        self.q_table = np.zeros((states, len(actions)))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.actions = actions

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(len(self.actions))  # Explore (action index)
        else:
            return np.argmax(self.q_table[state, :])  # Exploit (action index)

    def update_q_table(self, state, action_idx, reward, next_state):
        old_value = self.q_table[state, action_idx]
        next_max = np.max(self.q_table[next_state, :])
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action_idx] = new_value

    def decay_exploration(self):
        self.epsilon *= self.epsilon_decay

# --- 2. Simple Model & Data ---
# A minimal model and dummy dataset to test the training loop.
# In a real scenario, this would be a VLM and VQA data from the TARGET repo.
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1) # Dummy model: 10 input features, 1 output

    def forward(self, x):
        return self.fc(x)

def get_dummy_data(num_samples=200, features=10, batch_size=20):
    X = torch.randn(num_samples, features)
    # Simple linear relationship for the model to learn
    y = X[:, 0] * 2 + X[:, 1] * -3 + torch.randn(num_samples) * 0.1
    y = y.unsqueeze(1)
    dataset = TensorDataset(X, y)
    # drop_last=True is important for Opacus to ensure fixed batch sizes
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)

# --- 3. State & Reward Definition ---
# Discretize the loss into states for the Q-agent.
def get_state_from_loss(loss, loss_bins):
    if loss is None:
        return 0 # Initial state
    for i, bin_edge in enumerate(loss_bins):
        if loss < bin_edge:
            return i
    return len(loss_bins) # Return the highest state if loss is very high

# --- 4. Main Training Loop ---
def main(args):
    print("Starting experiment: Dynamic DP noise scheduling with a Q-learning agent.")
    print(f"This experiment is inspired by RLDP (SOURCE) and applies its core idea to a mock training task relevant to VQASynth (TARGET).")

    # --- Setup ---
    # Agent setup
    noise_multipliers = [0.5, 1.0, 2.0, 4.0]
    num_actions = len(noise_multipliers)
    
    # States are discretized loss values
    loss_bins = [0.1, 0.5, 1.0] # e.g., loss < 0.1 is state 0, 0.1 <= loss < 0.5 is state 1, etc.
    num_states = len(loss_bins) + 1
    
    agent = QLearningAgent(states=num_states, actions=noise_multipliers)

    dataloader = get_dummy_data(batch_size=args.batch_size)
    n_total_steps = len(dataloader)

    # RL-DP Training Loop
    # The loop is structured into "RL intervals" as described in the RLDP paper.
    # At each interval, the agent picks a new noise multiplier.
    current_loss = None
    for interval in range(args.rl_intervals):
        print(f"\n--- RL Interval {interval + 1}/{args.rl_intervals} ---")
        
        # 1. Agent chooses action based on current state (loss)
        state = get_state_from_loss(current_loss, loss_bins)
        action_idx = agent.choose_action(state)
        current_noise = agent.actions[action_idx]
        print(f"Current State (loss bucket): {state}. Agent chose Action: {action_idx} (Noise Multiplier: {current_noise}). Epsilon: {agent.epsilon:.3f}")
        
        # 2. Setup DP training for this interval
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, interval_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=current_noise,
            max_grad_norm=args.max_grad_norm,
        )
        
        # 3. Train for one "epoch" (or interval)
        model.train()
        total_loss = 0
        for x, y in tqdm(interval_dataloader, desc=f"Training with noise={current_noise}"):
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.MSELoss()(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / n_total_steps
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        print(f"Interval complete. Avg Loss: {avg_loss:.4f}. Privacy spent (\u03b5, \u03b4): ({epsilon:.2f}, {args.delta})")

        # 4. Calculate reward and update Q-table
        # Simple reward: negative loss. The agent's goal is to minimize loss.
        reward = -avg_loss
        next_state = get_state_from_loss(avg_loss, loss_bins)
        agent.update_q_table(state, action_idx, reward, next_state)
        
        # Update state for next interval
        current_loss = avg_loss
        agent.decay_exploration()
        privacy_engine.detach()

    print("\n--- Experiment Complete ---")
    print("Final Q-Table:")
    print(f"States (rows) are loss buckets: <{loss_bins[0]}, <{loss_bins[1]}, <{loss_bins[2]}, >={loss_bins[2]}")
    print(f"Actions (columns) are noise multipliers: {noise_multipliers}")
    print(np.round(agent.q_table, 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-DP Minimal Experiment")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for the model optimizer")
    parser.add_argument("--rl_intervals", type=int, default=20, help="Number of times the RL agent updates its policy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="DP clipping threshold")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--delta", type=float, default=1e-3, help="DP delta (must be smaller than 1/num_samples)")
    
    args = parser.parse_args()
    main(args)
