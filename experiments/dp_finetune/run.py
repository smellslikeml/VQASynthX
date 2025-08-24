import argparse
import numpy as np
import pandas as pd
import torch
import io
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# A simple bandit agent to choose the noise multiplier
class BanditAgent:
    def __init__(self, actions, learning_rate=0.1, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_values = {action: 0.0 for action in actions}
        logger.info(f"Agent initialized with actions: {self.actions}")

    def choose_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
            logger.info(f"Agent chose random action: {action}")
        else:
            action = max(self.q_values, key=self.q_values.get)
            logger.info(
                f"Agent chose greedy action: {action} (Q-values: {self.q_values})"
            )
        return action

    def update_q_value(self, action, reward):
        old_q = self.q_values[action]
        self.q_values[action] = old_q + self.lr * (reward - old_q)
        logger.info(
            f"Updated Q({action}) with reward {reward:.4f}. New Q-value: {self.q_values[action]:.4f}"
        )


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size=128):
        self.examples = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    self.examples.append(tokenized_text[i : i + block_size])
            elif len(tokenized_text) > 0:
                self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def evaluate(model, data_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            losses.append(outputs.loss.item())
    return np.mean(losses)


def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- 2. Data ---
    # Using dummy data for a self-contained script, similar to SOURCE_DOCKERFILE
    train_csv_content = """text
"Alice was beginning to get very tired of sitting by her sister on the bank."
"So she was considering in her own mind, whether the pleasure of making a daisy-chain would be worth the trouble."
"Suddenly a White Rabbit with pink eyes ran close by her."
"The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down."
"She found herself falling down what seemed to be a very deep well."
"Down, down, down. Would the fall never come to an end!"
"'I wonder how many miles I've fallen by this time?' she said aloud."
"'I wonder if I shall fall right through the earth!'"
"""
    eval_csv_content = """text
"This is a sentence for evaluation."
"We will test the model's performance on this text."
"""
    train_df = pd.read_csv(io.StringIO(train_csv_content))
    eval_df = pd.read_csv(io.StringIO(eval_csv_content))

    # --- 3. Model & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = ModuleValidator.fix(model)
    model.to(device)

    train_dataset = TextDataset(
        tokenizer, train_df["text"].tolist(), block_size=args.block_size
    )
    eval_dataset = TextDataset(
        tokenizer, eval_df["text"].tolist(), block_size=args.block_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch)

    # --- 4. Agent, Optimizer & Privacy Engine ---
    agent = BanditAgent(actions=args.noise_options, epsilon=args.epsilon_greedy)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )

    logger.info(f"Attached Opacus PrivacyEngine. Target epsilon: {args.target_epsilon}")

    # --- 5. Training Loop ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Agent takes an action at specified intervals
            if global_step % args.rl_interval == 0:
                logger.info(f"--- Agent Decision Point at Step {global_step} ---")
                eval_loss = evaluate(model, eval_loader, device)
                reward = -eval_loss  # Reward is negative loss

                # The first action is chosen before any updates
                if global_step > 0:
                    agent.update_q_value(current_noise, reward)

                current_noise = agent.choose_action()
                optimizer.noise_multiplier = current_noise  # Dynamically set noise
                logger.info(
                    f"Set optimizer noise multiplier to {optimizer.noise_multiplier}"
                )
                logger.info(f"Current eval perplexity: {np.exp(eval_loss):.2f}")
                logger.info("--------------------------------------------------")

            optimizer.zero_grad()
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epsilon_spent = privacy_engine.get_epsilon(args.target_delta)
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "epsilon": f"{epsilon_spent:.3f}",
                    "noise": f"{optimizer.noise_multiplier:.2f}",
                }
            )
            global_step += 1

    logger.info("Training finished.")
    final_eval_loss = evaluate(model, eval_loader, device)
    logger.info(
        f"Final evaluation loss: {final_eval_loss:.4f}, perplexity: {np.exp(final_eval_loss):.2f}"
    )
    logger.info(f"Final agent Q-values: {agent.q_values}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive DP Fine-tuning Experiment")
    parser.add_argument(
        "--model", type=str, default="sshleifer/tiny-gpt2", help="Model to fine-tune."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--train_batch", type=int, default=4, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch", type=int, default=4, help="Evaluation batch size."
    )
    parser.add_argument(
        "--block_size", type=int, default=64, help="Block size for tokenization."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # DP arguments
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=8.0,
        help="Target privacy budget epsilon.",
    )
    parser.add_argument(
        "--target_delta", type=float, default=1e-5, help="Target privacy budget delta."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Clipping threshold."
    )

    # Agent arguments (inspired by RLDP)
    parser.add_argument(
        "--rl_interval", type=int, default=10, help="Steps between agent actions."
    )
    parser.add_argument(
        "--noise_options",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="List of noise multipliers for the agent to choose from.",
    )
    parser.add_argument(
        "--epsilon_greedy",
        type=float,
        default=0.2,
        help="Epsilon for agent's epsilon-greedy policy.",
    )

    args = parser.parse_args()
    main(args)
