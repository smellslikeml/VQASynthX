import argparse
import json
import os
from collections import Counter

import datasets
import numpy as np
import pandas as pd
from copy import deepcopy

"""
This script partitions a dataset into non-IID subsets for federated learning simulations,
adapting the Dirichlet distribution partitioning method from the FedSWA repository.
Ref: https://github.com/junkangLiu0/FedSWA/blob/main/dirichlet_data.py

The script takes a Hugging Face dataset, a target column to use for class labels,
and federation parameters (number of clients, alpha for Dirichlet distribution) to
generate data shards for each simulated client.
"""


def partition_data_dirichlet(
    dataset_name: str,
    label_column: str,
    num_clients: int,
    alpha: float,
    output_path: str,
    data_subset: str = None,
    num_samples_per_client: int = -1,
):
    """
    Partitions a dataset using a Dirichlet distribution over class labels.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        label_column (str): The column in the dataset containing class labels.
        num_clients (int): The number of clients to partition data for.
        alpha (float): The concentration parameter for the Dirichlet distribution.
                       A smaller alpha creates more heterogeneous (non-IID) partitions.
        output_path (str): Directory to save the partition files.
        data_subset (str, optional): Subset of the dataset to use (e.g., 'train'). Defaults to None.
        num_samples_per_client (int, optional): Number of samples per client.
                                                If -1, partitions the entire dataset. Defaults to -1.
    """
    print(f"Loading dataset '{dataset_name}'...")
    dataset = datasets.load_dataset(dataset_name, name=data_subset, split="train")

    # 1. Group data indices by class
    labels = np.array(dataset[label_column])
    num_classes = len(np.unique(labels))
    print(f"Found {len(labels)} samples and {num_classes} classes.")

    indices_by_class = {
        i: np.where(labels == i)[0].tolist() for i in range(num_classes)
    }

    # Keep a pristine copy for resampling if needed
    indices_by_class_copy = deepcopy(indices_by_class)

    # 2. Determine number of samples per client
    if num_samples_per_client == -1:
        if len(labels) % num_clients != 0:
            print(
                "Warning: Dataset size is not perfectly divisible by the number of clients."
            )
        num_samples_per_client = len(labels) // num_clients

    print(
        f"Partitioning data for {num_clients} clients with {num_samples_per_client} samples each."
    )

    # 3. Generate client data distributions from Dirichlet
    # Each client's distribution over classes is a sample from Dir(alpha)
    class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

    client_partitions = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        client_sample_indices = []
        for _ in range(num_samples_per_client):
            # Choose a class based on the client's distribution
            sampled_class = np.random.choice(
                num_classes, p=class_distribution[client_id]
            )

            # Pop an index from that class's list
            if len(indices_by_class[sampled_class]) > 0:
                idx = indices_by_class[sampled_class].pop()
                client_sample_indices.append(int(idx))
            else:
                # If a class runs out of samples, restock from the copy and try again
                # This ensures clients can get their desired number of samples even
                # with highly skewed distributions.
                indices_by_class[sampled_class] = deepcopy(
                    indices_by_class_copy[sampled_class]
                )
                if len(indices_by_class[sampled_class]) > 0:
                    idx = indices_by_class[sampled_class].pop()
                    client_sample_indices.append(int(idx))
                else:
                    # This should not happen if the dataset has samples for every class
                    print(
                        f"Warning: Class {sampled_class} has no samples to draw from."
                    )
                    continue

        client_partitions[client_id] = client_sample_indices

    # 4. Analyze and save partitions
    os.makedirs(output_path, exist_ok=True)
    print(f"\n--- Partition Analysis (alpha={alpha}) ---")

    all_client_stats = []
    for client_id, indices in enumerate(client_partitions):
        client_labels = labels[indices]
        class_counts = Counter(client_labels)

        # Save partition to file
        partition_file = os.path.join(output_path, f"client_{client_id}.json")
        with open(partition_file, "w") as f:
            json.dump({"indices": indices}, f)

        # Collect stats for analysis
        stats = pd.Series(class_counts).describe()
        all_client_stats.append(stats["std"])
        if client_id < 5:  # Print first 5 clients for a preview
            print(
                f"Client {client_id}: {len(indices)} samples. Label counts: {sorted(class_counts.items())}"
            )

    print("-----------------------------------------")
    print(
        f"Mean std dev of label counts across clients: {np.mean(all_client_stats):.2f}"
    )
    print(f"Partitions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition a dataset for Federated Learning using a Dirichlet distribution."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help="Name of the Hugging Face dataset to use.",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        default=None,
        help="Subset of the dataset (e.g., 'plain_text' for C4).",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the column containing class labels.",
    )
    parser.add_argument(
        "--num_clients", type=int, default=100, help="Number of clients."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha parameter for the Dirichlet distribution (controls non-IID level).",
    )
    parser.add_argument(
        "--num_samples_per_client",
        type=int,
        default=-1,
        help="Number of samples per client. -1 to distribute all data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data_partitions",
        help="Directory to save the client data partitions.",
    )

    args = parser.parse_args()
    partition_data_dirichlet(
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        num_clients=args.num_clients,
        alpha=args.alpha,
        output_path=args.output_path,
        data_subset=args.data_subset,
        num_samples_per_client=args.num_samples_per_client,
    )
