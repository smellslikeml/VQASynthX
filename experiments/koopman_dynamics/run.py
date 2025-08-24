import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# --- Hyperparameters ---
IMG_SIZE = 32
N_CHANNELS = 1
LATENT_DIM = 64
NUM_SEQUENCES = 500
SEQUENCE_LENGTH = 10
SHAPE_SIZE = 5
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
KOOPMAN_LOSS_WEIGHT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Synthetic Dataset ---
class MovingShapesDataset(Dataset):
    """Generates sequences of images with a shape moving linearly."""

    def __init__(self, num_sequences, seq_length, img_size, shape_size):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.img_size = img_size
        self.shape_size = shape_size
        self.sequences = self._generate_all_sequences()

    def _generate_all_sequences(self):
        print(f"Generating {self.num_sequences} synthetic sequences...")
        all_pairs = []
        for _ in tqdm(range(self.num_sequences)):
            start_x = np.random.randint(0, self.img_size - self.shape_size)
            start_y = np.random.randint(0, self.img_size - self.shape_size)

            # Ensure velocity does not immediately push shape out of bounds
            vel_x = np.random.randint(-2, 3)
            vel_y = np.random.randint(-2, 3)
            if (
                start_x + vel_x * self.seq_length >= self.img_size - self.shape_size
                or start_x + vel_x * self.seq_length < 0
            ):
                vel_x = -vel_x
            if (
                start_y + vel_y * self.seq_length >= self.img_size - self.shape_size
                or start_y + vel_y * self.seq_length < 0
            ):
                vel_y = -vel_y

            frames = []
            for t in range(self.seq_length):
                frame = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                x = start_x + vel_x * t
                y = start_y + vel_y * t
                frame[y : y + self.shape_size, x : x + self.shape_size] = 1.0
                frames.append(frame)

            for i in range(self.seq_length - 1):
                all_pairs.append((frames[i], frames[i + 1]))
        return all_pairs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_t, frame_t_plus_1 = self.sequences[idx]
        # Add channel dimension
        frame_t = torch.from_numpy(frame_t).unsqueeze(0)
        frame_t_plus_1 = torch.from_numpy(frame_t_plus_1).unsqueeze(0)
        return frame_t, frame_t_plus_1


# --- Models ---
class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder to learn latent space representation."""

    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=3, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, N_CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> 32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class LinearTransition(nn.Module):
    """Linear Koopman operator in the latent space."""

    def __init__(self, latent_dim):
        super(LinearTransition, self).__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z):
        return self.linear(z)


# --- Training Loop ---
def main():
    print(f"Using device: {DEVICE}")

    # 1. Data
    dataset = MovingShapesDataset(NUM_SEQUENCES, SEQUENCE_LENGTH, IMG_SIZE, SHAPE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Models
    autoencoder = ConvAutoencoder(LATENT_DIM).to(DEVICE)
    koopman_op = LinearTransition(LATENT_DIM).to(DEVICE)

    # 3. Optimizer and Loss
    params = list(autoencoder.parameters()) + list(koopman_op.parameters())
    optimizer = optim.Adam(params, lr=LR)
    recon_criterion = nn.MSELoss()

    # 4. Training
    for epoch in range(EPOCHS):
        total_recon_loss = 0
        total_koopman_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img_t, img_t_plus_1 in progress_bar:
            img_t = img_t.to(DEVICE)
            img_t_plus_1 = img_t_plus_1.to(DEVICE)

            optimizer.zero_grad()

            # --- Autoencoder Loss ---
            # Reconstruct the current frame to train the autoencoder
            recon_t = autoencoder(img_t)
            recon_loss = recon_criterion(recon_t, img_t)

            # --- Koopman Loss ---
            # Predict the next latent state from the current one
            with torch.no_grad():
                z_t_plus_1_target = autoencoder.encoder(img_t_plus_1)

            z_t = autoencoder.encoder(img_t)
            z_t_plus_1_pred = koopman_op(z_t)
            koopman_loss = recon_criterion(z_t_plus_1_pred, z_t_plus_1_target)

            # --- Total Loss and Backward Pass ---
            total_loss = recon_loss + KOOPMAN_LOSS_WEIGHT * koopman_loss
            total_loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_koopman_loss += koopman_loss.item()
            progress_bar.set_postfix(
                {
                    "recon_loss": f"{recon_loss.item():.6f}",
                    "koopman_loss": f"{koopman_loss.item():.6f}",
                }
            )

        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_koopman_loss = total_koopman_loss / len(dataloader)
        print(
            f"Epoch {epoch+1} Summary | Avg Recon Loss: {avg_recon_loss:.6f} | Avg Koopman Loss: {avg_koopman_loss:.6f}"
        )

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
