import argparse
import json
import numpy as np
from collections import Counter
from copy import deepcopy
from datasets import load_dataset


def partition_data(dataset_name, partition_column, num_clients, alpha, output_path, data_subset_size=None):
    """
    Partitions a Hugging Face dataset among clients using a Dirichlet distribution,
    inspired by the method in FedSWA's dirichlet_data.py.

    This simulates a non-IID data distribution, common in Federated Learning,
    by skewing the distribution of a specific categorical column (e.g., 'scene_category')
    across different clients.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        partition_column (str): The column name to use for partitioning.
        num_clients (int): The number of clients to partition the data for.
        alpha (float): The concentration parameter for the Dirichlet distribution.
                       A small alpha (e.g., 0.1) creates highly non-IID partitions.
                       A large alpha (e.g., 100) creates more IID-like partitions.
        output_path (str): Path to save the output JSON file with client indices.
        data_subset_size (int, optional): Use a subset of the dataset for faster processing.
    """
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split='train')

    if data_subset_size:
        dataset = dataset.select(range(data_subset_size))
        print(f"Using a subset of {data_subset_size} samples.")

    print(f"Partitioning based on column '{partition_column}'...")
    labels = dataset[partition_column]
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Found {num_classes} unique classes: {unique_labels}")

    # Convert string labels to integer IDs for partitioning
    targets = np.array([label_to_id[label] for label in labels])
    
    # Group data indices by class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    # Generate client partitions using Dirichlet distribution
    # Each client's data distribution over classes is drawn from Dir(alpha)
    class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

    # Allocate data indices to clients based on the generated distributions
    client_indices = [[] for _ in range(num_clients)]
    available_indices = deepcopy(class_indices)

    # Calculate the number of samples per client
    num_samples_per_client = len(dataset) // num_clients

    for client_id in range(num_clients):
        client_sample_count = 0
        budget = num_samples_per_client
        # Use a copy of the distribution to modify as we assign samples
        dist_copy = class_distribution[client_id].copy()

        while client_sample_count < budget:
            # Normalize the distribution of remaining desired classes
            if dist_copy.sum() > 1e-8:
                dist_copy /= dist_copy.sum()
            else: # If distribution is exhausted, sample uniformly from available
                non_empty_classes = [c for c, ind in enumerate(available_indices) if len(ind) > 0]
                if not non_empty_classes:
                    break # No more data available
                dist_copy = np.ones(num_classes)
                mask = np.ones(num_classes, dtype=bool)
                mask[non_empty_classes] = False
                dist_copy[mask] = 0.0
                dist_copy /= dist_copy.sum()

            # Choose a class based on the client's distribution
            sampled_class = np.random.choice(num_classes, p=dist_copy)
            
            if len(available_indices[sampled_class]) > 0:
                # Pop an index from the chosen class and assign to the client
                idx = available_indices[sampled_class].pop()
                client_indices[client_id].append(int(idx))
                client_sample_count += 1
            else:
                # This class is exhausted, set its probability to 0 for this client
                dist_copy[sampled_class] = 0.0

    # Save the result
    print(f"Saving client data indices to '{output_path}'...")
    with open(output_path, 'w') as f:
        json.dump(client_indices, f)
    
    print("\n--- Partitioning Stats ---")
    for i in range(num_clients):
        client_labels = [labels[idx] for idx in client_indices[i]]
        label_counts = Counter(client_labels)
        print(f"Client {i} ({len(client_indices[i])} samples): {label_counts}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partition a Hugging Face dataset for Federated Learning using a Dirichlet distribution.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset on Hugging Face Hub (e.g., remyxai/OpenSpaces_MC_R1).')
    parser.add_argument('--partition_column', type=str, required=True, help='Name of the column to partition on (e.g., scene_category).')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients to simulate.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Concentration parameter for the Dirichlet distribution.')
    parser.add_argument('--output_path', type=str, default='client_indices.json', help='Path to save the output JSON file.')
    parser.add_argument('--data_subset_size', type=int, default=None, help='Use a smaller subset of data for quick tests.')

    args = parser.parse_args()
    
    partition_data(
        dataset_name=args.dataset_name,
        partition_column=args.partition_column,
        num_clients=args.num_clients,
        alpha=args.alpha,
        output_path=args.output_path,
        data_subset_size=args.data_subset_size
    )
