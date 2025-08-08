import torch
import torch.nn as nn
import torch.optim as optim
import time

# This experiment is inspired by GFNet-Dynn's early-exit architecture.
# See: https://github.com/KaHim-Lo/GFNet-Dynn/blob/main/README.md

# --- 1. Model Definition ---
# A simplified vision model with multiple exit points, similar to GFNet-Dynn.
class DynamicDepthModel(nn.Module):
    def __init__(self, num_exits=3):
        super().__init__()
        # Backbone feature extractor split into blocks
        self.backbone_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()) # Final block
        ])
        
        # Intermediate heads (exits) that predict a depth map
        # GFNet-Dynn has intermediate classifiers; here we have intermediate regressors.
        self.exit_heads = nn.ModuleList([
            nn.Conv2d(16, 1, 1), # Exit 1
            nn.Conv2d(32, 1, 1), # Exit 2
            nn.Conv2d(64, 1, 1)  # Exit 3 (after 3rd block)
        ])

        # The final, full-quality prediction head
        self.final_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        features = []
        exit_preds = []
        
        # Pass through backbone and collect features for exit heads
        h = self.backbone_blocks[0](x)
        features.append(h)
        h = self.backbone_blocks[1](h)
        features.append(h)
        h = self.backbone_blocks[2](h)
        features.append(h)

        # Get predictions from each exit
        for i, head in enumerate(self.exit_heads):
            exit_preds.append(head(features[i]))

        # Final full-path prediction
        final_pred = self.final_head(self.backbone_blocks[3](h))
        
        return exit_preds, final_pred

# --- 2. Training Logic ---
# Simplified two-phase training inspired by GFNet-Dynn's README
def train_dynamic_model(model, dataloader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Phase 1: Warmup - Train exit heads (and preceding backbone blocks)
    # In a real scenario, we'd freeze layers progressively as described in the source repo.
    # For this PoC, we train them all together.
    print("--- Phase 1: Warmup Training ---")
    model.train()
    for epoch in range(epochs):
        for i, (images, depths) in enumerate(dataloader):
            optimizer.zero_grad()
            exit_preds, final_pred = model(images)
            
            # Loss is a combination of all exit losses, similar to ce_ic_tradeoff
            loss = 0
            for pred in exit_preds:
                loss += criterion(pred, depths)
            loss.backward()
            optimizer.step()
            if i % 10 == 0: print(f"Warmup Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")

    # Phase 2: Full Model Training - Freeze exits and train the full path
    print("\n--- Phase 2: Full Model Training ---")
    for param in model.exit_heads.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs):
        for i, (images, depths) in enumerate(dataloader):
            optimizer.zero_grad()
            _, final_pred = model(images)
            loss = criterion(final_pred, depths)
            loss.backward()
            optimizer.step()
            if i % 10 == 0: print(f"Full Training Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")
    
    print("\nTraining complete.")
    return model

# --- 3. Inference Logic ---
def run_inference_with_early_exit(model, image, confidence_threshold=0.8):
    model.eval()
    with torch.no_grad():
        h = image
        start_time = time.time()

        # Process layer by layer and check for early exit
        for i in range(len(model.exit_heads)):
            h = model.backbone_blocks[i](h)
            prediction = model.exit_heads[i](h)
            
            # A mock confidence score. In reality, this could be based on prediction variance,
            # a separate learned confidence head, or other uncertainty metrics.
            confidence = 1.0 / (1.0 + torch.var(prediction)) 

            if confidence > confidence_threshold:
                end_time = time.time()
                print(f"Exiting early at Exit {i+1} with confidence {confidence:.2f}.")
                print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
                return prediction

        # If no early exit, proceed to the end
        h = model.backbone_blocks[len(model.exit_heads)](h)
        final_prediction = model.final_head(h)
        end_time = time.time()
        print("No early exit. Using final head.")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        return final_prediction

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # Create a dummy dataloader
    dummy_images = torch.randn(20, 3, 64, 64)
    dummy_depths = torch.randn(20, 1, 64, 64)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_depths)
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)

    # Instantiate and train the model
    dynamic_model = DynamicDepthModel(num_exits=3)
    trained_model = train_dynamic_model(dynamic_model, dummy_dataloader, epochs=1)

    # Demonstrate inference
    print("\n--- Inference Demonstration ---")
    test_image = torch.randn(1, 3, 64, 64)

    print("\nAttempting inference with high confidence threshold (force full path):")
    run_inference_with_early_exit(trained_model, test_image, confidence_threshold=0.99)

    print("\nAttempting inference with low confidence threshold (encourage early exit):")
    run_inference_with_early_exit(trained_model, test_image, confidence_threshold=0.7)
