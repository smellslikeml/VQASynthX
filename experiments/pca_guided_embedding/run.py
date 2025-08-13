# experiments/pca_guided_embedding/run.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.decomposition import PCA
import argparse
import os

# This script adapts the core idea from the "pca_guided_ae" repository.
# Source: https://github.com/mohammedsalah98/pca_guided_ae
# Specifically, it implements the PCA-guided knowledge distillation loss from `apply_ae_kd.ipynb`.
#
# The original work applied this to time-series data from infrared thermography.
# Here, we adapt it to image data (from CIFAR10) to test its feasibility for
# learning structured visual embeddings within the VQASynth context. The goal is to
# create a compact, structured latent space that might be more suitable for
# spatial reasoning tasks than generic embeddings.


def get_cifar10_data():
    """Load CIFAR10 data and flatten images."""
    # Use a local directory to store data to avoid re-downloading in Docker
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=ToTensor()
    )
    # We only need the images for unsupervised learning
    train_images = np.array([img.numpy() for img, _ in train_dataset])
    # Flatten images: (N, C, H, W) -> (N, C*H*W)
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    return train_images_flat


def kd_cosine_loss(z_student, z_teacher):
    """
    Knowledge distillation loss based on cosine similarity.
    This function is a direct implementation from the source repo's `apply_ae_kd.ipynb`.
    """
    z_student = F.normalize(z_student, p=2, dim=1)
    z_teacher = F.normalize(z_teacher, p=2, dim=1)
    # The loss is 1 - mean(cosine_similarity), encouraging the vectors to align.
    return 1 - F.cosine_similarity(z_student, z_teacher, dim=1).mean()


class Autoencoder(nn.Module):
    """
    A simple fully-connected autoencoder.
    The architecture is adapted from `apply_ae.ipynb` and `apply_ae_kd.ipynb`.
    """

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def main(args):
    """Main training and evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare data
    print("Loading CIFAR10 data...")
    image_matrix = get_cifar10_data()
    input_dim = image_matrix.shape[1]
    print(f"Data shape: {image_matrix.shape}")

    # 2. Create the "teacher" representation using PCA
    print(f"Fitting PCA for {args.latent_dim} components (the teacher)...")
    pca = PCA(n_components=args.latent_dim)
    pca_result = pca.fit_transform(image_matrix)
    print("PCA fitting complete.")

    # 3. Prepare data loaders
    train_data_tensor = torch.tensor(image_matrix, dtype=torch.float32)
    teacher_tensor = torch.tensor(pca_result, dtype=torch.float32)

    dataset = TensorDataset(train_data_tensor, teacher_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 4. Initialize model, optimizer, and loss
    model = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    reconstruction_criterion = nn.MSELoss()

    # 5. Training loop
    print("Starting training...")
    model.train()
    for epoch in range(args.num_epochs):
        total_loss, total_recon_loss, total_distill_loss = 0, 0, 0
        for x_batch, z_teacher_batch in dataloader:
            x_batch = x_batch.to(device)
            z_teacher_batch = z_teacher_batch.to(device)

            optimizer.zero_grad()

            # Forward pass to get reconstruction and student latent vector
            x_recon, z_student = model(x_batch)

            # Calculate losses
            recon_loss = reconstruction_criterion(x_recon, x_batch)
            distill_loss = kd_cosine_loss(z_student, z_teacher_batch)

            # Combine losses
            loss = recon_loss + args.alpha * distill_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_distill_loss += distill_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_distill_loss = total_distill_loss / len(dataloader)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{args.num_epochs}], Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg Distill Loss: {avg_distill_loss:.4f}"
            )

    print("Training complete.")

    # 6. Save the trained encoder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, "pca_guided_encoder.pth")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Trained encoder saved to {encoder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PCA-Guided Autoencoder on CIFAR10"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Dimension of the latent space."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Weight for the distillation loss."
    )

    args = parser.parse_args()
    main(args)
