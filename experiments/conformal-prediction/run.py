#!/usr/bin/env python
# A self-contained script to demonstrate Conformal Prediction for 3D coordinate uncertainty.
# This experiment integrates concepts from `ElSacho/Gaussian_Conformal_Prediction` into a VQASynth context.

import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# --- Core components from SOURCE repo, included here to be self-contained ---


def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Network(nn.Module):
    """A simple multi-layer perceptron to predict the mean (center) of the distribution."""

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_dim=64,
        n_hidden_layers=2,
        dropout_rate=0.1,
    ):
        super(Network, self).__init__()
        layers = [
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        layers.append(nn.Linear(hidden_dim, output_channels))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CholeskyMatrixPredictor(nn.Module):
    """Predicts the parameters for a Cholesky decomposition of a covariance matrix."""

    def __init__(
        self, input_dim, output_rows, hidden_dim=64, n_hidden_layers=2, dropout_rate=0.1
    ):
        super(CholeskyMatrixPredictor, self).__init__()
        self.output_rows = output_rows
        # Number of elements for diagonal and lower-triangular part of the Cholesky factor
        self.n_output_params = self.output_rows + (
            self.output_rows * (self.output_rows - 1) // 2
        )

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_hidden_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        layers.append(nn.Linear(hidden_dim, self.n_output_params))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        params = self.model(x)

        # Diagonal part (log-transformed for positivity)
        log_diag_entries = params[:, : self.output_rows]
        # Lower-triangular part (off-diagonal)
        tril_entries = params[:, self.output_rows :]

        L = torch.zeros(
            batch_size,
            self.output_rows,
            self.output_rows,
            device=x.device,
            dtype=x.dtype,
        )

        # Set diagonal elements
        L.diagonal(dim1=-2, dim2=-1)[:] = torch.exp(log_diag_entries)

        # Set lower-triangular elements
        tril_indices = torch.tril_indices(
            row=self.output_rows, col=self.output_rows, offset=-1
        )
        if tril_indices.numel() > 0:
            L[:, tril_indices[0], tril_indices[1]] = tril_entries

        # Return Sigma = L @ L.T, a symmetric positive-definite matrix
        return torch.bmm(L, L.transpose(1, 2))


class GaussianPredictorLevelsets:
    """Implements Conformal Prediction using a Gaussian score."""

    def __init__(self, center_model, matrix_model, dtype=torch.float32):
        self.center_model = center_model
        self.matrix_model = matrix_model
        self.dtype = dtype
        self.nu_conformal = None

    def fit(
        self,
        trainloader,
        stoploader=None,
        num_epochs=20,
        lr_center_models=1e-3,
        lr_matrix_models=1e-3,
        verbose=1,
    ):
        optimizer = torch.optim.Adam(
            [
                {"params": self.center_model.parameters(), "lr": lr_center_models},
                {"params": self.matrix_model.parameters(), "lr": lr_matrix_models},
            ]
        )

        for epoch in range(num_epochs):
            self.center_model.train()
            self.matrix_model.train()
            total_loss = 0
            for x, y in trainloader:
                optimizer.zero_grad()
                mu = self.center_model(x)
                Sigma = self.matrix_model(x)

                # Negative log-likelihood loss for a multivariate Gaussian
                residual = y - mu
                log_det_sigma = torch.logdet(Sigma)
                mahalanobis_dist = torch.einsum(
                    "bi,bij,bj->b", residual, torch.inverse(Sigma), residual
                )
                loss = torch.mean(log_det_sigma + mahalanobis_dist)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose > 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss / len(trainloader):.4f}"
                )

    def get_scores(self, x, y):
        self.center_model.eval()
        self.matrix_model.eval()
        with torch.no_grad():
            mu = self.center_model(x)
            Sigma = self.matrix_model(x)
            residual = y - mu
            # Score is the Mahalanobis distance
            scores = torch.einsum(
                "bi,bij,bj->b", residual, torch.inverse(Sigma), residual
            )
        return scores

    def conformalize(self, calibrationloader, alpha=0.1):
        cal_scores = []
        for x_cal, y_cal in calibrationloader:
            scores = self.get_scores(x_cal, y_cal)
            cal_scores.append(scores)

        cal_scores = torch.cat(cal_scores).cpu().numpy()
        q_level = np.ceil((1 - alpha) * (len(cal_scores) + 1)) / len(cal_scores)
        self.nu_conformal = torch.tensor(
            np.quantile(cal_scores, q_level), dtype=self.dtype
        )
        return self.nu_conformal

    def get_coverage(self, x_test, y_test):
        if self.nu_conformal is None:
            raise RuntimeError("Predictor must be conformalized first.")
        test_scores = self.get_scores(x_test, y_test)
        coverage = (
            (test_scores <= self.nu_conformal.to(test_scores.device)).float().mean()
        )
        return coverage.item()

    def get_averaged_volume(self, x_test):
        """Calculates the average volume of the prediction sets (ellipsoids)."""
        self.matrix_model.eval()
        with torch.no_grad():
            Sigma_test = self.matrix_model(x_test)
            # Volume is proportional to sqrt(det(Sigma))
            volumes = torch.sqrt(torch.det(Sigma_test))
        return volumes.mean().item()


def run_vqasynth_uncertainty_demo():
    """Demonstrates applying conformal prediction to a simulated VQASynth task."""
    # 1. Setup
    print("Setting up the VQASynth uncertainty demo...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32

    # 2. Generate synthetic data for a VQASynth-like task
    print("Generating synthetic data: (image_embeddings -> 3D_coordinates)")
    embedding_dim = 128  # Dimension of a mock image embedding
    coord_dim = 3  # Dimension of the 3D coordinates (x, y, z)
    n_samples = 5000

    # Mock image embeddings
    image_embeddings = np.random.normal(0, 1, size=(n_samples, embedding_dim))
    # Create a synthetic linear relationship for the coordinates
    true_coords = image_embeddings @ np.random.randn(embedding_dim, coord_dim)
    # Add Gaussian noise to simulate prediction error
    noisy_coords = true_coords + np.random.normal(0, 0.5, size=(n_samples, coord_dim))

    X = torch.tensor(image_embeddings, dtype=dtype, device=device)
    Y = torch.tensor(noisy_coords, dtype=dtype, device=device)

    # 3. Split data into train, calibration, and test sets
    print("Splitting data...")
    n_train = int(0.8 * n_samples)
    n_calib = int(0.1 * n_samples)
    n_test = n_samples - n_train - n_calib

    (X_train, X_calib, X_test) = torch.split(X, [n_train, n_calib, n_test])
    (Y_train, Y_calib, Y_test) = torch.split(Y, [n_train, n_calib, n_test])

    train_dataset = TensorDataset(X_train, Y_train)
    calib_dataset = TensorDataset(X_calib, Y_calib)

    batch_size = 128
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calibrationloader = DataLoader(calib_dataset, batch_size=batch_size)

    # 4. Initialize models for mean (center) and covariance (matrix)
    print("Initializing models...")
    center_model = Network(input_channels=embedding_dim, output_channels=coord_dim).to(
        device
    )

    matrix_model = CholeskyMatrixPredictor(
        input_dim=embedding_dim, output_rows=coord_dim
    ).to(device)

    # 5. Initialize and train the predictor
    print("Initializing and fitting the Gaussian Predictor...")
    predictor = GaussianPredictorLevelsets(center_model, matrix_model, dtype=dtype)
    predictor.fit(trainloader, num_epochs=25, verbose=1)

    # 6. Conformalize to get a calibrated uncertainty threshold
    print("\nConformalizing the predictor...")
    alpha = 0.1
    nu_conformal = predictor.conformalize(
        calibrationloader=calibrationloader, alpha=alpha
    )
    print(
        f"Conformalization finished. Threshold nu_conformal: {nu_conformal.item():.4f}"
    )

    # 7. Evaluate on the test set
    print("\nEvaluating on the test set...")
    coverage = predictor.get_coverage(X_test, Y_test)
    avg_volume = predictor.get_averaged_volume(X_test)

    print("\n--- VQASynth Uncertainty Demo Results ---")
    print(f"Target coverage: {1 - alpha:.2f}")
    print(f"Actual coverage: {coverage:.4f}")
    print(f"Average prediction set volume: {avg_volume:.4f}")
    print("-------------------------------------------")

    if abs(coverage - (1 - alpha)) < 0.05:
        print(
            "\n✅ Demo completed successfully: Actual coverage is close to the target."
        )
    else:
        print(
            f"\n⚠️ Demo completed, but coverage ({coverage:.2f}) is not close to the target ({1-alpha:.2f})."
        )


if __name__ == "__main__":
    run_vqasynth_uncertainty_demo()
