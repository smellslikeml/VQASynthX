# -*- coding: utf-8 -*-
# This script is a self-contained experiment combining elements from mdefresne/emmental_pll.
# It integrates training logic from Main_PLL.py and analysis from Constraints_hardening.py.

import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# pytoulbar2 is a required dependency for the solver component.
import pytoulbar2


# --- Utility functions (from Scripts/utils.py) ---
def guess_device():
    """Guesses the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch_input(data):
    """Creates a one-hot identity matrix representation for the Sudoku grid."""
    bs = data.shape[0]
    grid_size_sq = data.shape[1]
    I = torch.eye(grid_size_sq)
    return I.repeat(bs, 1, 1)


class SudokuDataset(Dataset):
    """Custom PyTorch Dataset for Sudoku CSV data."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor([int(c) for c in self.data.iloc[idx, 0]], dtype=torch.long)
        y = torch.tensor([int(c) for c in self.data.iloc[idx, 1]], dtype=torch.long) - 1
        return x, y


def get_loader(data, batch_size):
    """Creates a DataLoader for the Sudoku dataset."""
    dataset = SudokuDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# --- Model definition (from Scripts/Net.py) ---
class ResNet(nn.Module):
    def __init__(self, ft_size, nblocks=1):
        super(ResNet, self).__init__()
        self.nblocks = nblocks
        self.entry_block = nn.Sequential(nn.Linear(ft_size, ft_size), nn.ReLU())
        self.block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ft_size, ft_size), nn.ReLU(), nn.Linear(ft_size, ft_size)
                )
                for _ in range(nblocks)
            ]
        )
        self.exit_block = nn.ReLU()

    def forward(self, x):
        x = self.entry_block(x)
        for i in range(self.nblocks):
            x = self.exit_block(x + self.block[i](x))
        return x


class Net(nn.Module):
    def __init__(self, grid_size, hidden_size, nblocks=1):
        super(Net, self).__init__()
        self.grid_size = grid_size
        self.nb_var = grid_size**2
        self.ft_size = 2 * self.nb_var
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(self.nb_var, self.hidden_size)
        self.resnet = ResNet(self.hidden_size, nblocks=nblocks)
        self.output_layer = nn.Linear(2 * self.hidden_size, (self.grid_size) ** 2)

    def forward(self, x, device):
        bs = x.shape[0]
        v_ft = self.resnet(self.input_layer(x))
        v_ft = v_ft.view(bs, self.nb_var, self.hidden_size)
        W = torch.zeros(
            bs, self.nb_var, self.nb_var, self.grid_size, self.grid_size
        ).to(device)

        for i in range(self.nb_var):
            for j in range(i + 1, self.nb_var):
                ft_cat = torch.cat((v_ft[:, i, :], v_ft[:, j, :]), dim=1)
                W[:, i, j, :, :] = self.output_layer(ft_cat).view(
                    bs, self.grid_size, self.grid_size
                )
        return W


# --- Solver and Training Logic (from Scripts/Sudoku.py, Scripts/PLL.py) ---
def to_CFN(W, U, grid_size, top):
    """Converts learned potentials to a Cost Function Network for the solver."""
    Problem = pytoulbar2.CFN(top)
    nb_var = grid_size**2
    for i in range(nb_var):
        Problem.AddVariable(f"x{i}", range(grid_size))
    for i in range(nb_var):
        Problem.AddFunction([i], U[i].tolist())
    for i in range(nb_var):
        for j in range(i + 1, nb_var):
            Problem.AddFunction([i, j], W[i, j].flatten().tolist())
    return Problem


def EPL(model, train_loader, optimizer, k, epoch_max, device):
    """Energy-based Probabilistic Logic Learning training loop."""
    top = 9999999
    grid_size = model.grid_size
    model.train()
    for epoch in range(epoch_max):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            grid = get_batch_input(data).to(device)
            W = model(grid, device).squeeze()

            U = torch.zeros(grid_size**2, grid_size)
            Problem = to_CFN(W.detach().cpu(), U, grid_size, top)
            Problem.Solve(show=0)
            sol = torch.LongTensor(Problem.GetSolutions()[0]).to(device)

            mask = torch.randperm(grid_size**2)[:k]
            U[mask, sol[mask]] = top
            Problem = to_CFN(W.detach().cpu(), U, grid_size, top)
            Problem.Solve(show=0)
            sol = torch.LongTensor(Problem.GetSolutions()[0]).to(device)

            loss = 0
            for i in range(grid_size**2):
                for j in range(i + 1, grid_size**2):
                    loss += W[i, j, sol[i], sol[j]]
            loss = -loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epoch_max}, Loss: {epoch_loss/len(train_loader):.4f}")
    return model


# --- Constraint Analysis (from Constraints_hardening.py) ---
def count_cst(W, top):
    """Counts the number of correct and incorrect Sudoku constraints learned in W."""
    nb_var = W.shape[1]
    grid_size = int(nb_var**0.5)
    num_square = int(grid_size**0.5)
    n_learnt_cst, n_incorrect_cst, total_possible_cst = 0, 0, 0

    for i in range(nb_var):
        line_i, col_i = i // grid_size, i % grid_size
        sq_i = num_square * (line_i // num_square) + col_i // num_square
        for j in range(i + 1, nb_var):
            line_j, col_j = j // grid_size, j % grid_size
            sq_j = num_square * (line_j // num_square) + col_j // num_square

            if (line_i == line_j) or (col_i == col_j) or (sq_i == sq_j):
                total_possible_cst += 1
                diag = torch.diagonal(W[i, j].view(grid_size, grid_size), 0)
                if torch.min(diag).item() == top:
                    n_learnt_cst += 1
                off_diag_mask = ~torch.eye(grid_size, dtype=bool)
                if (
                    torch.max(W[i, j].view(grid_size, grid_size)[off_diag_mask]).item()
                    > 0
                ):
                    n_incorrect_cst += 1
            else:
                if torch.max(W[i, j]).item() > 0:
                    n_incorrect_cst += 1
    return n_learnt_cst, n_incorrect_cst, total_possible_cst


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Sudoku Constraint Learning Experiment"
    )
    parser.add_argument(
        "--path_to_data",
        type=str,
        default="Data_raw/sudoku-hard/",
        help="Path to Sudoku data.",
    )
    parser.add_argument(
        "--train_size", type=int, default=100, help="Number of training samples."
    )
    parser.add_argument(
        "--epoch_max", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of holes for E-PLL training."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Width of hidden layers."
    )
    parser.add_argument(
        "--nblocks", type=int, default=1, help="Number of ResNet blocks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sudoku_model.pt",
        help="Path to save/load the model.",
    )
    args = parser.parse_args()

    device = torch.device(guess_device())
    grid_size = 9

    print("--- Starting Training Phase ---")
    train_csv_path = os.path.join(args.path_to_data, "train.csv")
    if not os.path.exists(train_csv_path):
        print(f"Error: Training data not found at {train_csv_path}", file=sys.stderr)
        print(
            "Please run `bash download_data.sh` from the source repo root first.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_set = pd.read_csv(train_csv_path, names=["x", "y"])
    train_loader = get_loader(train_set[: args.train_size], batch_size=1)

    model = Net(
        grid_size=grid_size, hidden_size=args.hidden_size, nblocks=args.nblocks
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model = EPL(
        model,
        train_loader,
        optimizer,
        k=args.k,
        epoch_max=args.epoch_max,
        device=device,
    )
    torch.save(model.state_dict(), args.model_path)
    print(f"Training complete. Model saved to {args.model_path}")

    print("\n--- Starting Analysis Phase ---")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    grid_input = get_batch_input(torch.ones(1, grid_size**2)).to(device)
    W = model(grid_input, device).squeeze()

    top = 9999999
    logic_W = torch.zeros_like(W)
    # Harden constraints by setting weights for forbidden assignments to a large value (top)
    # This simplifies the logic from Constraints_hardening.py for a minimal test.
    logic_W[W > 0] = top

    n_learnt_cst, n_incorrect_cst, total_possible = count_cst(
        logic_W.detach().cpu(), top
    )

    print("\n--- Analysis Results ---")
    if total_possible > 0:
        percentage_learnt = n_learnt_cst / total_possible * 100
        print(
            f"{n_learnt_cst} constraints learnt ({percentage_learnt:.2f}% of {total_possible} possible constraints)"
        )
    else:
        print(f"{n_learnt_cst} constraints learnt")
    print(f"{n_incorrect_cst} incorrect constraints found.")


if __name__ == "__main__":
    main()
