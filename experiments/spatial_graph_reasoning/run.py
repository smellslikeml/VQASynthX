import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. Define the GCN model from the PyTorch Geometric README
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_mock_scene_graph():
    """
    Creates a mock scene graph representing objects and their relationships.
    In a real VQASynth pipeline, this would be generated from image analysis.
    - Nodes: Objects in the scene.
    - Edges: Proximity or other spatial relationships.
    - Node Features: Could be visual embeddings or 3D coordinates.
    - Labels: A property to be inferred, e.g., 'is central object'.
    """
    # 5 objects in the scene, each with 16 features (e.g., embeddings)
    node_features = torch.randn((5, 16))

    # Edges define relationships. Here, object '2' is a central, highly-connected node.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 2, 0, 2, 4],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 2, 4, 2]
    ], dtype=torch.long)

    # We want to classify each node. Task: identify the 'central' object.
    # Class 0: Peripheral object
    # Class 1: Central object
    labels = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)

    graph = Data(x=node_features, edge_index=edge_index, y=labels)
    
    # Use a mask to simulate a training scenario where we have labels for all nodes
    graph.train_mask = torch.tensor([True, True, True, True, True], dtype=torch.bool)

    return graph

def run_experiment():
    """
    Main function to run the GCN reasoning experiment.
    """
    print("Initializing spatial reasoning experiment using PyTorch Geometric...")
    
    # 1. Load the data (our mock scene graph)
    scene_graph = create_mock_scene_graph()
    num_node_features = scene_graph.num_node_features
    num_classes = len(scene_graph.y.unique())

    print(f"Scene graph created with {scene_graph.num_nodes} nodes and {scene_graph.num_edges} edges.")
    print(f"Node features: {num_node_features}, Number of classes: {num_classes}")

    # 2. Initialize the GCN model
    model = GCN(in_channels=num_node_features, hidden_channels=16, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("\nStarting GCN model training...")
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(scene_graph.x, scene_graph.edge_index)
        loss = criterion(out[scene_graph.train_mask], scene_graph.y[scene_graph.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("\nTraining finished.")

    # 4. Evaluate the model (simple inference)
    model.eval()
    pred = model(scene_graph.x, scene_graph.edge_index).argmax(dim=1)
    correct = (pred[scene_graph.train_mask] == scene_graph.y[scene_graph.train_mask]).sum()
    acc = int(correct) / int(scene_graph.train_mask.sum())
    print(f'Final Accuracy: {acc:.4f}')
    print(f'Ground Truth: {scene_graph.y.tolist()}')
    print(f'Prediction:   {pred.tolist()}')

if __name__ == "__main__":
    run_experiment()
