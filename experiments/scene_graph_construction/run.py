import torch
from torch_geometric.data import Data
from itertools import combinations
import numpy as np


def create_scene_graph(object_detections):
    """
    Constructs a scene graph from a list of object detections.

    Args:
        object_detections (list): A list of dictionaries, where each dictionary
                                  represents an object and must contain 'centroid'
                                  and 'features' keys.
                                  - 'centroid': A list/tuple of (x, y, z) coordinates.
                                  - 'features': A list/numpy array of numerical features.

    Returns:
        torch_geometric.data.Data: A PyG Data object representing the scene graph.
                                   Nodes represent objects, and edges represent
                                   spatial relationships (fully connected in this case).
                                   Edge attributes store the Euclidean distance
                                   between connected objects.
    """
    if not object_detections:
        return Data()

    # 1. Node features (x)
    # Each node is an object. Features can be semantic embeddings, etc.
    node_features = [obj["features"] for obj in object_detections]
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. Node positions (pos)
    # Store 3D centroids for calculating spatial relationships.
    node_positions = [obj["centroid"] for obj in object_detections]
    pos = torch.tensor(node_positions, dtype=torch.float)

    # 3. Edges (edge_index) and Edge attributes (edge_attr)
    # Create a fully connected graph where every object is connected to every other object.
    num_nodes = len(object_detections)
    source_nodes = []
    target_nodes = []
    edge_attrs = []

    # Use itertools.combinations to create unique pairs of nodes
    for i, j in combinations(range(num_nodes), 2):
        # Add edges in both directions for an undirected graph
        source_nodes.extend([i, j])
        target_nodes.extend([j, i])

        # Calculate Euclidean distance as an edge attribute
        dist = torch.linalg.norm(pos[i] - pos[j])
        edge_attrs.extend([dist, dist])  # Same distance for both directions

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)

    # 4. Create the PyG Data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    # Validate the created graph
    graph_data.validate()

    return graph_data


def main():
    """
    Main function to demonstrate scene graph construction with mock data.
    """
    print("Demonstrating scene graph construction with PyTorch Geometric.")

    # Mock data simulating output from an object detection/localization stage.
    # In a real pipeline, this would be loaded from a file (e.g., JSON, Parquet).
    mock_detections = [
        {
            "id": "red_forklift",
            "centroid": [1.2, 0.5, 3.4],
            "features": [0.1, 0.9, 0.2],
        },
        {
            "id": "cardboard_boxes",
            "centroid": [3.5, 0.8, 3.2],
            "features": [0.8, 0.2, 0.5],
        },
        {"id": "man_in_hat", "centroid": [2.0, 1.0, 5.1], "features": [0.5, 0.5, 0.9]},
        {
            "id": "wooden_pallet",
            "centroid": [3.8, 0.2, 3.0],
            "features": [0.7, 0.3, 0.4],
        },
    ]
    print(f"\nFound {len(mock_detections)} objects in the scene.")

    # Create the scene graph
    scene_graph = create_scene_graph(mock_detections)

    print("\nSuccessfully created a PyTorch Geometric Data object:")
    print(scene_graph)

    # Print details of the graph
    print(f"\nNumber of nodes: {scene_graph.num_nodes}")
    print(f"Number of edges: {scene_graph.num_edges}")
    print(f"Node features shape: {scene_graph.x.shape}")
    print(f"Edge index shape: {scene_graph.edge_index.shape}")
    print(f"Edge attributes shape: {scene_graph.edge_attr.shape}")
    print(f"Average node degree: {scene_graph.num_edges / scene_graph.num_nodes:.2f}")
    print(f"Contains isolated nodes: {scene_graph.has_isolated_nodes()}")
    print(f"Is undirected: {scene_graph.is_undirected()}")

    # Save the graph object to a file for downstream use
    output_path = "scene_graph.pt"
    torch.save(scene_graph, output_path)
    print(f"\nScene graph saved to '{output_path}'.")

    # Example of loading it back
    loaded_graph = torch.load(output_path)
    print(f"Successfully loaded graph from '{output_path}':")
    print(loaded_graph)


if __name__ == "__main__":
    main()
