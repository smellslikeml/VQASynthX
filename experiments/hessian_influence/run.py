#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from backpack import backpack, extend
from backpack.extensions import DiagHessian
import numpy as np


# --- 1. Model Definition ---
# A simple logistic regression model implemented as a single-layer network.
# This is analogous to the 'Net' in the SOURCE repo's notebook but simplified.
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# --- 2. Data Preparation ---
# Create a synthetic dataset for reproducibility.
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

full_dataset = TensorDataset(X, y)
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# Define a "forget set" - a small subset of data whose influence we want to measure.
# This mirrors the concept of 'forget_set' in the source repo's RTBF_unlearning.ipynb.
forget_indices = np.random.choice(len(full_dataset), size=50, replace=False)
forget_X = X[forget_indices]
forget_y = y[forget_indices]
forget_dataset = TensorDataset(forget_X, forget_y)
forget_loader = DataLoader(
    forget_dataset, batch_size=len(forget_dataset)
)  # Process in one batch for a global Hessian

# --- 3. Model Training ---
input_dim = X.shape[1]
output_dim = len(torch.unique(y))
model = LogisticRegression(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training a simple logistic regression model...")
model.train()
for epoch in range(20):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training complete.")

# --- 4. Hessian-based Influence Analysis ---
# This is the core idea from the SOURCE paper: use second-order information
# to identify parameters sensitive to a specific subset of data.
print("\nAnalyzing parameter influence using the Hessian diagonal...")

# Extend the model and loss function with backpack
extended_model = extend(model)
extended_criterion = extend(criterion)

# Get the single batch from the forget loader
forget_inputs, forget_labels = next(iter(forget_loader))

# Calculate the loss on the "forget set"
loss = extended_criterion(extended_model(forget_inputs), forget_labels)

# Use backpack to compute the diagonal of the Hessian
with backpack(DiagHessian()):
    loss.backward()

print("Hessian diagonal computed for the 'forget set'.")

# --- 5. Identify Most Influential Parameters ---
# The parameters with the largest Hessian diagonal entries are the most sensitive
# to the "forget set" data. These would be candidates for resetting in an
# unlearning scenario.
param_influence = {}
for name, param in extended_model.named_parameters():
    if hasattr(param, "diag_h"):
        # We store the sum of the diagonal Hessian for each parameter tensor (e.g., weights, biases)
        # to get a single influence score per layer/parameter type.
        influence_score = param.diag_h.sum().item()
        param_influence[name] = {
            "influence_score": influence_score,
            "shape": param.shape,
            "num_elements": param.numel(),
        }

# Sort parameters by their influence score
sorted_params = sorted(
    param_influence.items(), key=lambda item: item[1]["influence_score"], reverse=True
)

print("\n--- Top Influential Parameters (Ranked by summed Hessian diagonal) ---")
for name, data in sorted_params:
    print(
        f"Parameter: {name:<15} | Influence Score: {data['influence_score']:.4f} | Shape: {str(data['shape']):<15}"
    )

print(
    "\nExperiment finished. This analysis identifies which parameters were most sensitive"
)
print(
    "to the 'forget set', demonstrating the core diagnostic of the FedUnlearn-PE paper."
)
