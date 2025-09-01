#!/usr/bin/env python
# This script is a self-contained experiment adapted from:
# https://github.com/bcoueraud87/geometry_regularized_autoencoders
# It demonstrates how to visualize the manifold learned by an autoencoder.

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
LATENT_DIM = 2
N_SAMPLES = 2000
N_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
OUTPUT_DIR = "./outputs/geometric_analysis"


# --- Model Definition ---
class Autoencoder(nn.Module):
    """A simple multi-layer perceptron autoencoder."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# --- Data Generation ---
def create_swiss_roll(nbr_samples: int, noise: float = 0.0):
    """Creates a Swiss roll dataset."""
    points, colors = make_swiss_roll(n_samples=nbr_samples, noise=noise)
    points = torch.from_numpy(points.astype(np.float32))
    colors = torch.from_numpy(colors.astype(np.float32))
    return points, colors


# --- Main Experiment Logic ---
def run_experiment():
    """Main function to run the training and visualization."""
    print("Starting manifold geometry visualization experiment...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    points, colors = create_swiss_roll(N_SAMPLES)
    dataset = TensorDataset(points)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = Autoencoder(input_dim=3, latent_dim=LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"Training for {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(device)
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {total_loss/len(dataloader):.6f}"
            )

    print("Training complete. Generating visualization...")

    # Visualization
    model.eval()
    with torch.no_grad():
        # Get latent space representation of original data
        latent_codes = model.encode(points.to(device)).cpu().numpy()

        # Create a regular grid in the latent space
        u_min, u_max = latent_codes[:, 0].min(), latent_codes[:, 0].max()
        v_min, v_max = latent_codes[:, 1].min(), latent_codes[:, 1].max()
        grid_u, grid_v = np.meshgrid(
            np.linspace(u_min, u_max, 20), np.linspace(v_min, v_max, 20)
        )
        grid_latent = np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1)
        grid_latent_torch = torch.from_numpy(grid_latent.astype(np.float32)).to(device)

        # Decode the grid to get the manifold in 3D space
        decoded_grid = model.decode(grid_latent_torch).cpu().numpy()
        grid_x = decoded_grid[:, 0].reshape(grid_u.shape)
        grid_y = decoded_grid[:, 1].reshape(grid_v.shape)
        grid_z = decoded_grid[:, 2].reshape(grid_u.shape)

    # Plotting
    fig = plt.figure(figsize=(12, 6))

    # 1. Latent space plot
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(
        latent_codes[:, 0], latent_codes[:, 1], c=colors, cmap="viridis"
    )
    ax1.set_title("Latent Space Embedding")
    ax1.set_xlabel("Latent Dim 1")
    ax1.set_ylabel("Latent Dim 2")
    ax1.grid(True)

    # 2. 3D manifold plot
    ax2 = fig.add_subplot(122, projection="3d")
    # Plot original data points
    ax2.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="viridis",
        s=10,
        alpha=0.5,
    )
    # Plot the learned manifold grid
    ax2.plot_surface(
        grid_x, grid_y, grid_z, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.5
    )
    ax2.set_title("Learned Manifold in Original Space")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=20, azim=-65)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "manifold_visualization.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()
