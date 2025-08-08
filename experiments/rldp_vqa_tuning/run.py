import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
import numpy as np
import torch.nn.functional as F
from collections import namedtuple

# --- Configuration ---
CONFIG = {
    "epochs": 2,
    "lr": 0.05,
    "batch_size": 64,
    "val_batch_size": 256,
    "noise_multiplier": 1.0,
    "delta": 1e-5,
    "rl_interval": 5,  # Number of training steps between RL agent actions
    "log_interval": 10,
    "n_features": 10,
    "n_classes": 2,
}

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# --- 1. Simple Reinforcement Learning Agent (Actor-Critic) ---
# This agent learns to select the clipping threshold for DP-SGD.
class RLAgent(nn.Module):
    def __init__(self):
        super(RLAgent, self).__init__()
        # State: [normalized_step, last_validation_loss]
        self.affine1 = nn.Linear(2, 128)
        
        # Actor head: chooses an action (clipping threshold)
        self.action_head = nn.Linear(128, 3)  # Actions: [0.8, 1.0, 1.2]
        
        # Critic head: evaluates the state
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state):
        probs, state_value = self.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def update(self):
        if not self.rewards:
            return

        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        # Calculate discounted rewards
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # gamma = 0.99
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        del self.rewards[:]
        del self.saved_actions[:]


# --- 2. Dummy Model and Data (simulating VQA model and data) ---
class DummyVQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CONFIG["n_features"], 16)
        self.fc2 = nn.Linear(16, CONFIG["n_classes"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_data_loaders():
    X_train = torch.randn(2048, CONFIG["n_features"])
    y_train = torch.randint(0, CONFIG["n_classes"], (2048,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"])

    X_val = torch.randn(512, CONFIG["n_features"])
    y_val = torch.randint(0, CONFIG["n_classes"], (512,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["val_batch_size"])
    return train_loader, val_loader

# --- 3. Main Experiment Loop ---
def run_experiment():
    print("--- Starting RLDP-inspired Experiment ---")
    
    model = DummyVQA()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["lr"])
    train_loader, val_loader = get_data_loaders()
    
    # Action space for the RL agent
    clipping_thresholds = [0.8, 1.0, 1.2]
    current_clipping_threshold = clipping_thresholds[1] # Start with the middle value

    # Opacus Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=CONFIG["noise_multiplier"],
        max_grad_norm=current_clipping_threshold,
        poisson_sampling=False,
    )
    
    rl_agent = RLAgent()
    max_steps = len(train_loader) * CONFIG["epochs"]
    global_step = 0
    last_val_loss = float('inf')

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            # RL agent action step
            if global_step % CONFIG["rl_interval"] == 0:
                # 1. Get state
                state = torch.tensor([
                    global_step / max_steps, 
                    last_val_loss if last_val_loss != float('inf') else 1.0
                ], dtype=torch.float32)

                # 2. Select and apply action
                action_idx = rl_agent.select_action(state)
                new_clipping_threshold = clipping_thresholds[action_idx]
                
                if new_clipping_threshold != current_clipping_threshold:
                    # In a real scenario, this is how you'd update the clipper
                    optimizer.max_grad_norm = new_clipping_threshold
                    current_clipping_threshold = new_clipping_threshold
                    print(f"[Step {global_step}] RL Agent set max_grad_norm to: {current_clipping_threshold}")

                # 3. Calculate reward from previous action (based on validation loss)
                # and update agent
                if global_step > 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_data, val_target in val_loader:
                            output = model(val_data)
                            val_loss += F.cross_entropy(output, val_target, reduction='sum').item()
                    val_loss /= len(val_loader.dataset)
                    
                    # Reward is negative validation loss
                    reward = -val_loss
                    rl_agent.rewards.append(reward)
                    rl_agent.update()
                    last_val_loss = val_loss
                    model.train()

            # VQA model training step
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % CONFIG["log_interval"] == 0:
                epsilon = privacy_engine.get_epsilon(delta=CONFIG["delta"])
                print(
                    f"Epoch: {epoch} | Step: {global_step}/{max_steps} | "
                    f"Loss: {loss.item():.4f} | Epsilon: {epsilon:.2f} | "
                    f"Current Clip: {current_clipping_threshold}"
                )

    print("--- Experiment Finished ---")
    final_epsilon = privacy_engine.get_epsilon(delta=CONFIG['delta'])
    print(f"Final privacy budget: (ε = {final_epsilon:.2f}, δ = {CONFIG['delta']})")

if __name__ == "__main__":
    run_experiment()
