import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
# Use assets present in the target repository
IMAGE_PATHS = [
    "assets/warehouse_sample_1.jpeg",
    "assets/warehouse_sample_2.jpeg",
    "assets/warehouse_sample_3.jpeg",
]

# --- Data Augmentation and Degradation ---
# Define transforms for a "good" quality image
good_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a set of transforms to create "bad" quality images
bad_transforms = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomChoice([
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        T.ColorJitter(brightness=0.5, contrast=0.5),
        T.RandomPosterize(bits=2),
    ]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Dataset Definition ---
class IQADataset(Dataset):
    """
    A synthetic dataset for Image Quality Assessment.
    For each image path, it can generate a "good" (label 1) or "bad" (label 0) sample.
    To create a balanced dataset, we generate one good and one bad version of each source image.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self._samples = []
        for path in self.image_paths:
            # Add a "good" sample and a "bad" sample for each image
            self._samples.append((path, 1)) # Good
            self._samples.append((path, 0)) # Bad

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        img_path_str, label = self._samples[idx]
        # This path logic assumes the script is run from the repo root.
        img_path = Path(img_path_str)

        if not img_path.is_file():
            raise FileNotFoundError(f"Could not find image at {img_path}. Please run this script from the root of the 'experimental-vqasynth' repository.")

        image = Image.open(img_path).convert("RGB")

        if label == 1: # Good
            tensor = good_transform(image)
        else: # Bad
            tensor = bad_transforms(image)
            
        return tensor, torch.tensor(label, dtype=torch.long)

# --- Main Experiment Logic ---
def run_experiment():
    print(f"Starting IQA classification experiment on {DEVICE}")
    print("This experiment is inspired by FetalCLIP-IQA, testing if a vision model can be")
    print("simply fine-tuned for a specialized classification task (good vs. bad image quality).\n")

    # 1. Create Dataset and DataLoader
    dataset = IQADataset(IMAGE_PATHS)
    # Since the dataset is tiny, we'll use it for both training and testing
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Created a dataset with {len(dataset)} samples ({len(IMAGE_PATHS)} good, {len(IMAGE_PATHS)} bad).")

    # 2. Initialize Model, Loss, and Optimizer
    # Inspired by FetalCLIP-IQA using various backbones like ViT for classification.
    # We use a pre-trained Vision Transformer and replace the head for our 2-class problem.
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    print("--- Training Finished ---\n")

    # 4. Evaluation
    print("--- Starting Evaluation ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final Accuracy on the dataset: {accuracy:.2f}%")
    
    if accuracy > 80:
        print("\nSuccess! Model learned to distinguish between 'good' and 'bad' quality images.")
    else:
        print("\nModel performance is low. Further tuning or a larger dataset might be needed.")

if __name__ == "__main__":
    try:
        run_experiment()
    except FileNotFoundError as e:
        print(f"Error: {e}")
