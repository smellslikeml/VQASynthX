#!/usr/bin/env python3
"""
This script implements a novelty scoring experiment inspired by the INS2ANE
(Integrated Novelty Score–Strategic Autonomous Non-Smooth Exploration) concept.
It uses the reconstruction error of a simple convolutional autoencoder to assign a
'novelty score' to images. Images that the autoencoder struggles to reconstruct
receive a higher score, indicating they are anomalous or novel compared to the
training distribution.

This approach is adapted from the demonstration script provided for the INS2ANE
project. In the context of VQASynth, this can be used as a data curation tool
to identify interesting or unusual images for more complex processing.
"""

import os
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def download_data(file_path, url):
    """Downloads data from a URL if it doesn't exist locally."""
    if not os.path.exists(file_path):
        print(f"Downloading data to {file_path}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print(f"{file_path} already exists.")


class SimpleAutoencoder(nn.Module):
    """A simple convolutional autoencoder for 32x32 images."""

    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7),  # -> 64x2x2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),  # -> 32x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> 1x32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main(args):
    """Main function to run the novelty scoring experiment."""
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Data Preparation ---
    file_path = os.path.join(args.output_dir, "Gmode_line.mat")
    url = "https://www.dropbox.com/scl/fi/w9qev2r6l18k9p3v6g00f/Gmode_line.mat?rlkey=v6l675z66m9u8b80g0qsg6m7h&dl=1"
    download_data(file_path, url)

    data = loadmat(file_path)
    images = data["data"]
    images = (images - images.min()) / (images.max() - images.min())
    images_tensor = torch.from_numpy(images).float().unsqueeze(1)
    dataset = TensorDataset(images_tensor, images_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- 2. Model Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting autoencoder training...")
    losses = []
    for epoch in range(args.epochs):
        for data_batch in dataloader:
            img, _ = data_batch
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.6f}")
        losses.append(loss.item())
    print("Training finished.")

    # --- 3. Novelty Scoring ---
    print("Scoring images for novelty...")
    model.eval()
    per_image_loss = []
    with torch.no_grad():
        for i, (img, _) in enumerate(eval_dataloader):
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            per_image_loss.append((i, loss.item()))

    # Sort images by novelty score (reconstruction loss) in descending order
    per_image_loss.sort(key=lambda x: x[1], reverse=True)

    # --- 4. Reporting Results ---
    # Plot and save training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(args.output_dir, "training_loss.png")
    plt.savefig(loss_plot_path)
    print(f"Saved training loss plot to {loss_plot_path}")

    # Visualize and save the most novel images
    print("Visualizing top novel images...")
    top_n = min(len(per_image_loss), 10)  # Show top 10 or fewer
    fig, axs = plt.subplots(2, top_n // 2, figsize=(15, 7))
    fig.suptitle(f"Top {top_n} Most Novel Images (by Reconstruction Error)")
    axs = axs.flatten()
    for i in range(top_n):
        image_index, score = per_image_loss[i]
        img_tensor = eval_dataloader.dataset[image_index][0]
        axs[i].imshow(img_tensor.squeeze(), cmap="gray")
        axs[i].set_title(f"Image #{image_index}\nScore: {score:.6f}")
        axs[i].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    novelty_plot_path = os.path.join(args.output_dir, "novelty_scores.png")
    plt.savefig(novelty_plot_path)
    print(f"Saved top novel images plot to {novelty_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Autoencoder Novelty Scoring Experiment."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/novelty_scoring",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()
    main(args)
