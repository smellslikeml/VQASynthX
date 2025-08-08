import argparse
import numpy as np
import os
from collections import Counter
from datasets import load_dataset
import json

def partition_dataset_by_dirichlet(dataset, label_column, num_clients, alpha):
    """
    Partitions a Hugging Face dataset into non-IID subsets for clients using a Dirichlet distribution.

    This logic is adapted from the data partitioning strategy in FedSWA, which is designed
    to simulate heterogeneous data distributions in federated learning. The 'alpha' parameter
    controls the degree of non-IID-ness: a smaller alpha leads to more skewed data
    distributions among clients.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset to partition.
        label_column (str): The name of the column containing the class labels.
        num_clients (int): The number of clients to partition the data for.
        alpha (float): The concentration parameter for the Dirichlet distribution.

    Returns:
        dict: A dictionary where keys are client IDs (0 to num_clients-1) and
              values are lists of indices from the original dataset.
    """
    print(f"Partitioning data for {num_clients} clients with alpha={alpha}...")

    labels = np.array(dataset[label_column])
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    # Create a list of indices for each class
    class_indices = {i: [] for i in range(num_classes)}
    for i, label in enumerate(labels):
        class_indices[label_to_id[label]].append(i)

    # Generate client data distributions from a Dirichlet distribution
    # Each client gets a proportion vector for the classes
    client_proportions = np.random.dirichlet([alpha] * num_classes, num_clients)

    # Distribute data indices to clients based on the generated proportions
    client_data_indices = {i: [] for i in range(num_clients)}
    
    # Get total number of samples per class
    num_samples_per_class = {cls_id: len(indices) for cls_id, indices in class_indices.items()}
    
    # Allocate indices to clients
    for cls_id in range(num_classes):
        class_idx_list = class_indices[cls_id]
        np.random.shuffle(class_idx_list)
        
        # Calculate how many samples of this class each client should get
        num_samples_for_cls = num_samples_per_class[cls_id]
        proportions_for_cls = client_proportions[:, cls_id]
        samples_per_client_for_cls = (proportions_for_cls / proportions_for_cls.sum() * num_samples_for_cls).astype(int)
        
        # Correct for rounding errors
        diff = num_samples_for_cls - samples_per_client_for_cls.sum()
        for i in range(diff):
            samples_per_client_for_cls[i % num_clients] += 1
            
        # Assign indices
        start_idx = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client_for_cls[client_id]
            end_idx = start_idx + num_samples
            client_data_indices[client_id].extend(class_idx_list[start_idx:end_idx])
            start_idx = end_idx

    # Print statistics for a few clients to verify heterogeneity
    print("\n--- Partitioning Statistics ---")
    for client_id in range(min(num_clients, 5)):
        client_indices = client_data_indices[client_id]
        client_labels = [labels[i] for i in client_indices]
        label_counts = Counter(client_labels)
        print(f"Client {client_id} (Total: {len(client_indices)} samples): {label_counts.most_common(5)}")

    return client_data_indices


def main():
    parser = argparse.ArgumentParser(description="Partition a dataset non-IID using a Dirichlet distribution.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Hugging Face dataset (e.g., 'remyxai/SpaceOm').")
    parser.add_argument("--label_column", type=str, required=True, help="The column name in the dataset to use as the class label for partitioning.")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients to partition data for.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet distribution concentration parameter. Lower is more non-IID.")
    parser.add_argument("--output_dir", type=str, default="./partitioned_data", help="Directory to save the partitioned data files.")
    parser.add_argument("--dataset_subset", type=str, default="train", help="Dataset split to use (e.g., 'train').")
    
    args = parser.parse_args()

    # Load the dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_dataset(args.dataset_path, split=args.dataset_subset)
    except Exception as e:
        print(f"Could not load dataset split '{args.dataset_subset}'. Trying without split argument. Error: {e}")
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, dict):
            split_name = list(dataset.keys())[0]
            print(f"Dataset is a dictionary. Using the first split: '{split_name}'")
            dataset = dataset[split_name]


    # Partition the dataset
    client_indices = partition_dataset_by_dirichlet(
        dataset=dataset,
        label_column=args.label_column,
        num_clients=args.num_clients,
        alpha=args.alpha
    )

    # Save the partitioned data
    os.makedirs(args.output_dir, exist_ok=True)
    for client_id, indices in client_indices.items():
        client_dataset = dataset.select(indices)
        output_file = os.path.join(args.output_dir, f"client_{client_id}.jsonl")
        
        with open(output_file, 'w') as f:
            for item in client_dataset:
                f.write(json.dumps(item) + '\n')
        
    print(f"\nSuccessfully partitioned data into {args.num_clients} client files in '{args.output_dir}'.")
    
if __name__ == "__main__":
    main()
