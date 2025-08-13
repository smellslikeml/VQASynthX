import torch
import numpy as np
import random
from collections import defaultdict, Counter


# This function is adapted from the create_subset function in
# https://github.com/RAI-SCC/ResNet/blob/main/resnet/dataloader.py
# It has been modified to be a standalone utility without dependencies on perun or MPI.
def create_class_balanced_subset(original_dataset, subset_size: int, seed: int = 42):
    """
    Create a class-balanced subset of a PyTorch dataset.

    This utility is designed for creating smaller, representative data slices for
    efficient experimentation, debugging, or testing, ensuring that the class
    distribution of the original dataset is preserved.

    Args:
        original_dataset: A PyTorch-like dataset object that has a `.targets`
                          attribute containing a list or array of labels.
        subset_size (int): The desired number of samples in the subset.
        seed (int): A random seed for reproducibility.
    """
    if not hasattr(original_dataset, "targets"):
        raise ValueError("original_dataset must have a 'targets' attribute.")

    if subset_size > len(original_dataset):
        raise ValueError(
            f"subset_size ({subset_size}) cannot be larger than the "
            f"original dataset size ({len(original_dataset)})."
        )

    # Set seed for reproducible sampling
    np.random.seed(seed)
    random.seed(seed)

    # Group indices by class
    targets = np.array(original_dataset.targets)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    total_size = len(targets)
    unique_labels = sorted(class_indices.keys())

    # Calculate how many samples to take from each class
    class_share_ints = {}
    class_share_diffs = {}
    intermediate_size = 0
    for label in unique_labels:
        # Proportion of this class in the original dataset
        label_share = len(class_indices[label]) / total_size
        # Target number of samples for this class in the subset
        target_samples_float = label_share * subset_size
        class_share_ints[label] = int(target_samples_float)
        class_share_diffs[label] = target_samples_float - int(target_samples_float)
        intermediate_size += int(target_samples_float)

    # The sum of floored sample counts might not equal the target subset_size.
    # Distribute the remaining samples to classes with the largest fractional parts.
    samples_to_distribute = subset_size - intermediate_size
    top_n_classes_by_diff = sorted(
        class_share_diffs.items(), key=lambda x: x[1], reverse=True
    )[:samples_to_distribute]

    for label, _ in top_n_classes_by_diff:
        class_share_ints[label] += 1

    # Sample the indices for the subset
    subset_indices = []
    for label in unique_labels:
        num_samples = class_share_ints[label]
        # Ensure we don't request more samples than available for a class
        num_samples = min(num_samples, len(class_indices[label]))

        # Randomly sample without replacement
        chosen_indices = np.random.choice(
            class_indices[label], num_samples, replace=False
        )
        subset_indices.extend(chosen_indices)

    # Create the PyTorch Subset
    subset = torch.utils.data.Subset(original_dataset, subset_indices)

    # Add a 'targets' attribute to the subset for convenience
    subset.targets = targets[subset_indices]

    return subset


def demonstrate_subsetting():
    """
    A demonstration of the create_class_balanced_subset utility.
    """
    print("--- Demonstrating Class-Balanced Subsetting ---")

    # 1. Create a dummy dataset with a skewed class distribution
    # 1000 samples total
    # Class 0: 600 samples (60%)
    # Class 1: 300 samples (30%)
    # Class 2: 100 samples (10%)
    features = torch.randn(1000, 10)
    targets = torch.cat(
        [torch.zeros(600), torch.ones(300), torch.full((100,), 2.0)]
    ).int()

    # Shuffle the dataset
    indices = torch.randperm(1000)
    features = features[indices]
    targets = targets[indices]

    original_dataset = torch.utils.data.TensorDataset(features, targets)
    # The TensorDataset doesn't have a .targets attribute, so we add it.
    original_dataset.targets = targets.tolist()

    print(f"\nOriginal Dataset Size: {len(original_dataset)}")
    original_counts = Counter(original_dataset.targets)
    print("Original Class Distribution:")
    for label, count in sorted(original_counts.items()):
        print(f"  Class {label}: {count} samples ({count / len(original_dataset):.2%})")

    # 2. Create a 20% subset (200 samples)
    subset_size = 200
    print(f"\nCreating a subset of size {subset_size}...")
    subset_dataset = create_class_balanced_subset(original_dataset, subset_size)

    print(f"\nSubset Dataset Size: {len(subset_dataset)}")
    subset_counts = Counter(subset_dataset.targets)
    print("Subset Class Distribution:")
    for label, count in sorted(subset_counts.items()):
        print(f"  Class {label}: {count} samples ({count / len(subset_dataset):.2%})")

    # 3. Verify the distribution is preserved
    print("\n--- Verification ---")
    print(
        "The percentage of each class in the subset should be very close to the original."
    )
    original_dist = {k: v / len(original_dataset) for k, v in original_counts.items()}
    subset_dist = {k: v / len(subset_dataset) for k, v in subset_counts.items()}

    all_good = True
    for label in original_dist:
        diff = abs(original_dist[label] - subset_dist.get(label, 0))
        print(
            f"Class {label}: Original {original_dist[label]:.2%} vs Subset {subset_dist.get(label, 0):.2%} (Diff: {diff:.2%})"
        )
        if diff > 0.01:  # Allow for small rounding differences
            all_good = False

    if all_good:
        print("\n✅ Success: Class distribution is well-preserved.")
    else:
        print("\n❌ Warning: Class distribution differs significantly.")


if __name__ == "__main__":
    demonstrate_subsetting()
