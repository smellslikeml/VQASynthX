import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import copy

# Core idea from: https://github.com/JFernando4/collas_2025_swr_paper
# Specifically, the concept of reinitializing low-magnitude weights.
# See: src/swr_functions/selective_weight_reinitialization.py


def selective_weight_reinitialization(model: nn.Module, reinit_fraction: float):
    """
    Reinitializes a fraction of the weights with the smallest magnitude in each linear layer.
    This function is a minimal implementation of the core idea from the SWR paper.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight

                # Flatten the weight tensor to find the k-th smallest element
                flat_weight = weight.view(-1)
                num_to_reinit = int(reinit_fraction * flat_weight.numel())

                if num_to_reinit == 0:
                    continue

                # Find the magnitude threshold for the weights to be reinitialized
                # We use absolute values to determine magnitude
                threshold = torch.kthvalue(flat_weight.abs(), num_to_reinit).values

                # Create a mask for weights with magnitude below the threshold
                reinit_mask = weight.abs() <= threshold

                # Get the number of weights to be reinitialized (can be slightly different from num_to_reinit due to ties)
                n_reinitialized = torch.sum(reinit_mask).item()

                # Create new random weights for reinitialization
                # For this minimal experiment, we use Kaiming uniform, a standard default.
                new_weights = torch.empty_like(weight)
                nn.init.kaiming_uniform_(new_weights, a=np.sqrt(5))

                # Apply the new weights where the mask is true
                weight.data[reinit_mask] = new_weights[reinit_mask]

                print(
                    f"  Reinitialized {n_reinitialized}/{flat_weight.numel()} weights in a layer."
                )


class SimpleMLP(nn.Module):
    """A simple MLP for the Permuted MNIST task."""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def create_permuted_mnist_task(dataset, permutation_seed):
    """Creates a permuted version of the MNIST dataset."""
    rng = np.random.RandomState(permutation_seed)
    perm = torch.from_numpy(rng.permutation(784)).long()

    permuted_data = dataset.data.clone()
    permuted_data = permuted_data.view(-1, 28 * 28)
    permuted_data = permuted_data[:, perm]
    permuted_data = permuted_data.view(-1, 28, 28)

    permuted_dataset = copy.deepcopy(dataset)
    permuted_dataset.data = permuted_data

    return permuted_dataset


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    use_swr,
    reinit_fraction,
):
    """Main training and evaluation loop."""
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Apply SWR at the end of the epoch
        if use_swr:
            print(f"Epoch {epoch+1}: Applying SWR...")
            selective_weight_reinitialization(model, reinit_fraction)

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Test Accuracy: {accuracy:.2f}%")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="SWR Experiment on Permuted MNIST")
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=3,
        help="Number of Permuted MNIST tasks to train on sequentially.",
    )
    parser.add_argument(
        "--epochs_per_task", type=int, default=2, help="Epochs to train on each task."
    )
    parser.add_argument(
        "--reinit_fraction",
        type=float,
        default=0.05,
        help="Fraction of weights to reinitialize.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load base MNIST data
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # --- Baseline Model (No SWR) ---
    print("--- Training Baseline Model (No SWR) ---")
    baseline_model = SimpleMLP()
    baseline_optimizer = optim.SGD(baseline_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    baseline_accuracies = []

    for task_id in range(args.num_tasks):
        print(f"\nTraining on Task {task_id + 1}/{args.num_tasks}")
        permuted_train_dataset = create_permuted_mnist_task(train_dataset, task_id)
        permuted_test_dataset = create_permuted_mnist_task(test_dataset, task_id)
        train_loader = DataLoader(
            permuted_train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            permuted_test_dataset, batch_size=args.batch_size, shuffle=False
        )
        acc = train_and_evaluate(
            baseline_model,
            train_loader,
            test_loader,
            baseline_optimizer,
            criterion,
            args.epochs_per_task,
            use_swr=False,
            reinit_fraction=0.0,
        )
        baseline_accuracies.append(acc)

    # --- SWR Model ---
    print("\n\n--- Training SWR Model ---")
    swr_model = SimpleMLP()
    swr_model.load_state_dict(SimpleMLP().state_dict())
    swr_optimizer = optim.SGD(swr_model.parameters(), lr=0.01, momentum=0.9)
    swr_accuracies = []

    for task_id in range(args.num_tasks):
        print(f"\nTraining on Task {task_id + 1}/{args.num_tasks}")
        permuted_train_dataset = create_permuted_mnist_task(train_dataset, task_id)
        permuted_test_dataset = create_permuted_mnist_task(test_dataset, task_id)
        train_loader = DataLoader(
            permuted_train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            permuted_test_dataset, batch_size=args.batch_size, shuffle=False
        )
        acc = train_and_evaluate(
            swr_model,
            train_loader,
            test_loader,
            swr_optimizer,
            criterion,
            args.epochs_per_task,
            use_swr=True,
            reinit_fraction=args.reinit_fraction,
        )
        swr_accuracies.append(acc)

    print("\n\n--- Experiment Results ---")
    print(
        f"Baseline Final Accuracies per Task: {[f'{acc:.2f}%' for acc in baseline_accuracies]}"
    )
    print(
        f"SWR Final Accuracies per Task:      {[f'{acc:.2f}%' for acc in swr_accuracies]}"
    )
    print(f"Average Final Accuracy (Baseline): {np.mean(baseline_accuracies):.2f}%")
    print(f"Average Final Accuracy (SWR):      {np.mean(swr_accuracies):.2f}%")


if __name__ == "__main__":
    main()
