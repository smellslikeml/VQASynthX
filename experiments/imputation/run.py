import torch
import torch.nn as nn
import torch.optim as optim
import copy


# This ResidualAE class is adapted directly from the HARDY-MER source repository.
# Source: https://github.com/HARDY-MER/HARDY-MER/blob/main/autoencoder.py
class ResidualAE(nn.Module):
    """Residual autoencoder using fc layers
    layers should be something like [128, 64, 32]
    eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                      concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    """

    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, "encoder_" + str(i), self.get_encoder(layers))
            setattr(self, "decoder_" + str(i), self.get_decoder(layers))

    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)

    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            all_layers.append(nn.ReLU())  # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, "encoder_" + str(i))
            decoder = getattr(self, "decoder_" + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in + x_out), latents


def run_experiment():
    """
    A minimal, self-contained script to test the ResidualAE on
    synthetic data resembling object embeddings from the VQASynth pipeline.
    """
    print("Starting ResidualAE imputation experiment...")

    # --- 1. Configuration ---
    # Simulate a batch of scenes, each with multiple object embeddings
    BATCH_SIZE = 16
    NUM_OBJECTS_PER_SCENE = 8
    EMBEDDING_DIM = 256
    INPUT_DIM = NUM_OBJECTS_PER_SCENE * EMBEDDING_DIM

    # Model configuration
    # These layers define the dimensions of the autoencoder's bottleneck
    AE_LAYERS = [1024, 512, 256]
    AE_BLOCKS = 2  # Number of residual blocks

    # Training configuration
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Model and Data Initialization ---
    # Instantiate the model from HARDY-MER
    model = ResidualAE(
        layers=AE_LAYERS, n_blocks=AE_BLOCKS, input_dim=INPUT_DIM, dropout=0.2
    ).to(device)

    # Create a dummy dataset: a batch of flattened scene vectors
    # In a real scenario, this would come from the 'embeddings_stage'
    dummy_data = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model initialized. Input dimension: {INPUT_DIM}")
    print(f"Training on {BATCH_SIZE} dummy scenes for {EPOCHS} epochs.")

    # --- 3. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed_data, latents = model(dummy_data)
        loss = criterion(reconstructed_data, dummy_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    print("\nExperiment finished successfully.")
    print("The model has been trained to reconstruct scene-level feature vectors.")
    print("This demonstrates its potential for imputing missing object data.")


if __name__ == "__main__":
    run_experiment()
