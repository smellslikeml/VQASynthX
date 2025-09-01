import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    Simple MLP projector for mapping features to a new latent space.
    As described in CLIPin, this is an additional projection head.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class CLIPinLoss(nn.Module):
    """
    A non-contrastive loss to enforce semantic alignment between paired
    image and text embeddings, inspired by T6Yang/CLIPin.

    This implementation uses a Mean Squared Error (MSE) loss on the L2-normalized
    outputs of two separate projection heads. This encourages the representations
    of a correct image-text pair to be identical in the projected space.
    """

    def __init__(self, embedding_dim: int, projection_dim: int = 512):
        super().__init__()
        # Projectors for image and text features
        self.image_projector = Projector(
            embedding_dim, embedding_dim * 2, projection_dim
        )
        self.text_projector = Projector(
            embedding_dim, embedding_dim * 2, projection_dim
        )

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the non-contrastive alignment loss.

        Args:
            image_features: A tensor of image features from the vision encoder.
                            Shape: (batch_size, embedding_dim)
            text_features: A tensor of text features (e.g., from the [CLS] token
                           or mean-pooled last hidden state of the text encoder).
                           Shape: (batch_size, embedding_dim)

        Returns:
            The calculated CLIPin loss value.
        """
        # Project features into a new space
        image_projection = self.image_projector(image_features)
        text_projection = self.text_projector(text_features)

        # Normalize the projected features to prevent collapse and focus on cosine similarity
        image_projection = F.normalize(image_projection, p=2, dim=1)
        text_projection = F.normalize(text_projection, p=2, dim=1)

        # Calculate MSE loss between the normalized projections
        loss = F.mse_loss(image_projection, text_projection)
        return loss
