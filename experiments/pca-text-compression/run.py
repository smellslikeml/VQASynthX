import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import time
import argparse

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
NUM_PCA_COMPONENTS = 70
NUM_SAMPLES = 8000
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# --- Model Definition ---
class SimpleClassifier(nn.Module):
    """A simple feed-forward neural network for classification."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def main(fast_dev_run):
    """Main function to run the experiment."""
    num_samples = 1000 if fast_dev_run else NUM_SAMPLES
    epochs = 3 if fast_dev_run else EPOCHS
    
    print("--- Experiment: PCA for Text Embedding Compression ---")
    print(f"Inspired by: https://github.com/mnbe1973/PCA_LLM")
    print(f"Using device: {DEVICE}")
    if fast_dev_run:
        print("\n*** Running in fast development mode. Using fewer samples and epochs. ***")

    # 1. Load and prepare data
    print(f"\n[1/5] Loading 20 Newsgroups dataset ({num_samples} samples)...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data[:num_samples]
    labels = newsgroups.target[:num_samples]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Found {len(texts)} texts and {num_classes} classes.")

    # 2. Embed texts
    print(f"\n[2/5] Embedding texts with '{EMBEDDING_MODEL}'...")
    start_time = time.time()
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=128)
    original_dim = embeddings.shape[1]
    print(f"Embedding complete in {time.time() - start_time:.2f}s. Original embedding dim: {original_dim}")

    X_train_emb, X_test_emb, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )

    # 3. Apply PCA for dimensionality reduction
    print(f"\n[3/5] Applying PCA to reduce dimension to {NUM_PCA_COMPONENTS}...")
    start_time = time.time()
    pca = PCA(n_components=NUM_PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_emb)
    X_test_pca = pca.transform(X_test_emb)
    print(f"PCA complete in {time.time() - start_time:.2f}s.")
    print(f"Total explained variance by {NUM_PCA_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 4. Prepare PyTorch DataLoaders
    print("\n[4/5] Creating PyTorch DataLoaders...")
    train_dataset = TensorDataset(torch.tensor(X_train_pca, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 5. Train and evaluate the classifier
    print("\n[5/5] Training and evaluating simple classifier...")
    model = SimpleClassifier(input_dim=NUM_PCA_COMPONENTS, num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n--- Results ---")
    print(f"Final Test Accuracy on PCA-compressed embeddings: {accuracy:.2f}%")
    print(f"This demonstrates classification performance on embeddings compressed from {original_dim} to {NUM_PCA_COMPONENTS} dimensions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA Text Compression Experiment.")
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Use a small subset of data and fewer epochs for a quick test run.'
    )
    args = parser.parse_args()
    main(args.fast_dev_run)
