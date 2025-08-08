import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Define Models ---
# A very simple segmentation model (e.g., a shallow U-Net or just a few conv layers)
class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1) # Output one channel for the mask
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# A simple discriminator for domain adaptation
# It tries to distinguish features from the source vs. target domain
class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32 * 64 * 64, 100), # 32 channels, 64x64 image size
            nn.ReLU(),
            nn.Linear(100, 2), # 2 classes: source vs target
            nn.LogSoftmax(dim=1)
        )

    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.model(features)

# --- 2. Mock Data ---
# Simulate a batch of data from the source (e.g., synthetic) and target (e.g., real-world) domains
BATCH_SIZE = 4
IMG_SIZE = 64

# Source domain: labeled images and masks
source_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
source_masks = torch.randint(0, 2, (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)).float()

# Target domain: unlabeled images
target_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE) # Add noise to simulate domain shift

# --- 3. Experiment Setup ---
print("Initializing experiment: Domain Adaptation for Segmentation")
segmentation_model = SimpleSegmentationModel()
discriminator_model = DomainDiscriminator()

# Optimizers
opt_seg = optim.Adam(segmentation_model.parameters(), lr=0.001)
opt_disc = optim.Adam(discriminator_model.parameters(), lr=0.001)

# Loss functions
seg_loss_fn = nn.BCEWithLogitsLoss()
domain_loss_fn = nn.NLLLoss()

# Domain labels
source_domain_labels = torch.zeros(BATCH_SIZE, dtype=torch.long)
target_domain_labels = torch.ones(BATCH_SIZE, dtype=torch.long)

# --- 4. Training Loop ---
# This simplified loop demonstrates the core concept from CEDANet:
# training the segmentor on source data while adapting its features
# to be indistinguishable from the target data.
print("Starting training loop...")
for epoch in range(1, 6):
    # --- Train Segmentation Model ---
    opt_seg.zero_grad()
    opt_disc.zero_grad()

    # A. Segmentation loss on source data
    source_features = segmentation_model.encoder(source_images)
    source_preds = segmentation_model.decoder(source_features)
    loss_seg = seg_loss_fn(source_preds, source_masks)

    # B. Adaptation loss (confuse the discriminator)
    target_features = segmentation_model.encoder(target_images)
    # We want the discriminator to predict these are from the SOURCE domain (label 0)
    disc_preds_on_target = discriminator_model(target_features)
    loss_adapt = domain_loss_fn(disc_preds_on_target, source_domain_labels)
    
    # Combine losses for the segmentation model
    # The lambda hyperparameter balances the two losses
    lambda_adapt = 0.1
    total_seg_loss = loss_seg + lambda_adapt * loss_adapt
    total_seg_loss.backward()
    opt_seg.step()

    # --- Train Discriminator Model ---
    # Detach features to prevent gradients from flowing back to the segmentor
    source_features_detached = source_features.detach()
    target_features_detached = target_features.detach()

    # A. Loss on source features
    disc_preds_on_source = discriminator_model(source_features_detached)
    loss_disc_source = domain_loss_fn(disc_preds_on_source, source_domain_labels)

    # B. Loss on target features
    disc_preds_on_target_for_disc = discriminator_model(target_features_detached)
    loss_disc_target = domain_loss_fn(disc_preds_on_target_for_disc, target_domain_labels)

    total_disc_loss = (loss_disc_source + loss_disc_target) / 2
    total_disc_loss.backward()
    opt_disc.step()

    print(
        f"Epoch {epoch}: "
        f"Seg Loss: {loss_seg.item():.4f}, "
        f"Adapt Loss: {loss_adapt.item():.4f}, "
        f"Disc Loss: {total_disc_loss.item():.4f}"
    )

print("\nTraining complete.")
# --- 5. Save Artifacts ---
model_path = "adapted_segmentation_model.pth"
torch.save(segmentation_model.state_dict(), model_path)
print(f"Adapted segmentation model saved to {model_path}")
