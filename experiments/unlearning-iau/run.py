import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# --- 1. Configuration & Setup ---

# Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 5 # A short training period for a baseline model
UNLEARN_RATIO = 0.1 # 10% of the training data will be 'forgotten'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- 2. Model Definition ---

class SimpleCNN(nn.Module):
    """A standard CNN for CIFAR-10 classification."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Data Preparation ---

def prepare_data():
    """Load CIFAR-10 and split into training, unlearn, remain, and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load full training and test sets
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split training data into unlearn and remain sets
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(UNLEARN_RATIO * num_train))
    unlearn_idx, remain_idx = indices[:split], indices[split:]

    unlearn_set = Subset(train_set, unlearn_idx)
    remain_set = Subset(train_set, remain_idx)

    print(f"Dataset split summary:")
    print(f"  - Total training samples: {num_train}")
    print(f"  - Unlearn samples ({UNLEARN_RATIO*100}%): {len(unlearn_set)}")
    print(f"  - Remain samples: {len(remain_set)}")
    print(f"  - Test samples: {len(test_set)}\n")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # For the IAU unlearning step, we process all data in one batch
    unlearn_loader_full = DataLoader(unlearn_set, batch_size=len(unlearn_set))
    remain_loader_full = DataLoader(remain_set, batch_size=len(remain_set))

    # DataLoaders for evaluation
    unlearn_loader_eval = DataLoader(unlearn_set, batch_size=BATCH_SIZE)
    remain_loader_eval = DataLoader(remain_set, batch_size=BATCH_SIZE)

    return train_loader, test_loader, unlearn_loader_full, remain_loader_full, unlearn_loader_eval, remain_loader_eval

# --- 4. Training and Evaluation Logic ---

def train_model(model, train_loader, epochs):
    """Train the model for a specified number of epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
    print("Training finished.\n")

def evaluate_model(model, name, test_loader, remain_loader, unlearn_loader):
    """Evaluate model accuracy on test, remain, and unlearn sets."""
    model.eval()
    print(f"--- Evaluating {name} Model ---")
    with torch.no_grad():
        for loader_name, loader in [("Test Set", test_loader), ("Remain Set", remain_loader), ("Unlearn Set", unlearn_loader)]:
            if not loader:
                continue
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"  Accuracy on {loader_name}: {accuracy:.2f}%")
    print("---------------------------\n")

# --- 5. Unlearning Step (IAU Implementation) ---

def perform_unlearning(model, remain_loader, unlearn_loader):
    """Applies the IAU unlearning algorithm.

    This function is a direct implementation of the logic in IAU_unlearn.py.
    It computes gradients on the remaining data to reinforce them, and opposing
    gradients on the data to be forgotten, then applies a single optimizer step.
    """
    print("Performing IAU unlearning step...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    optimizer.zero_grad()

    # Step 1: Reinforce knowledge of the 'remain' set (gradient descent)
    # The original paper uses the whole dataset in one go.
    for r_inputs, r_labels in remain_loader:
        r_inputs, r_labels = r_inputs.to(DEVICE), r_labels.to(DEVICE)
        r_outputs = model(r_inputs)
        loss_remain = criterion(r_outputs, r_labels)
        loss_remain.backward() # Accumulate gradients

    # Step 2: Weaken knowledge of the 'unlearn' set (gradient ascent)
    for u_inputs, u_labels in unlearn_loader:
        u_inputs, u_labels = u_inputs.to(DEVICE), u_labels.to(DEVICE)
        u_outputs = model(u_inputs)
        # Using a negative loss causes gradient ASCENT, moving parameters
        # away from the minimum for this data, effectively 'unlearning' it.
        loss_unlearn_neg = -1.0 * criterion(u_outputs, u_labels)
        loss_unlearn_neg.backward() # Accumulate gradients

    # Step 3: Apply the combined gradients
    optimizer.step()
    print("Unlearning step finished.\n")


# --- 6. Main Execution ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")
    
    # 1. Get data
    train_loader, test_loader, unlearn_loader_full, remain_loader_full, unlearn_loader_eval, remain_loader_eval = prepare_data()

    # 2. Create and train the original model
    original_model = SimpleCNN().to(DEVICE)
    train_model(original_model, train_loader, epochs=TRAIN_EPOCHS)

    # 3. Evaluate the original model as a baseline
    evaluate_model(original_model, "Original", test_loader, remain_loader_eval, unlearn_loader_eval)

    # 4. Perform the unlearning procedure
    unlearned_model = original_model # Modify in place
    perform_unlearning(unlearned_model, remain_loader_full, unlearn_loader_full)

    # 5. Evaluate the model after unlearning
    evaluate_model(unlearned_model, "Unlearned", test_loader, remain_loader_eval, unlearn_loader_eval)
