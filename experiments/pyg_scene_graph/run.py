import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. Define a Scene as a collection of objects with 3D coordinates.
# In a real VQASynth pipeline, this data would come from the 3D reconstruction modules.
# For this minimal experiment, we use mock data.
# Each object is a node in our graph.
scene_objects = [
    {"id": "red_box", "coords": [0.1, 0.5, 1.0]},
    {"id": "blue_sphere", "coords": [0.8, 0.5, 1.2]},
    {"id": "green_cone", "coords": [0.4, 0.9, 0.8]},
    {"id": "floor", "coords": [0.5, 0.0, 1.0]},
]
num_nodes = len(scene_objects)

# 2. Convert the scene into a PyG graph representation.
# The node features `x` will be the 3D coordinates of each object.
node_features = torch.tensor([obj["coords"] for obj in scene_objects], dtype=torch.float)

# We create a fully connected graph where every object is related to every other object.
# A GNN can learn to prune or weight these connections.
# edge_index is of shape [2, num_edges]
edge_list = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            edge_list.append([i, j])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Create the PyG Data object
scene_graph = Data(x=node_features, edge_index=edge_index)

print("--- Scene Graph Data ---")
print(scene_graph)
print("------------------------\n")


# 3. Define a Graph Neural Network model using PyG's GCNConv layer.
# This model takes the node features (coords) and graph structure (edge_index)
# and computes new representations for each node based on its neighbors.
# The architecture is inspired by the basic GCN example in the PyG documentation.
class SceneGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        # First GCN layer: learns initial neighborhood representations
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Second GCN layer: learns higher-order relationships
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x shape: [num_nodes, in_channels]
        # edge_index shape: [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        # output shape: [num_nodes, out_channels]
        return x

# Instantiate the model.
# in_channels = 3 because our node features are (x, y, z) coordinates.
# out_channels = 8 for an arbitrary 8-dimensional embedding per node.
model = SceneGCN(in_channels=3, hidden_channels=16, out_channels=8)
print("--- GCN Model Architecture ---")
print(model)
print("----------------------------\n")

# 4. Perform a forward pass to process the scene graph.
# This demonstrates that the GNN can compute embeddings for each object
# based on the spatial relationships encoded in the graph.
print("--- Running Forward Pass ---")
node_embeddings = model(scene_graph.x, scene_graph.edge_index)
print("--------------------------\n")

print("Successfully computed node embeddings.")
print(f"Input node features shape: {scene_graph.x.shape}")
print(f"Output node embeddings shape: {node_embeddings.shape}")
assert node_embeddings.shape == (num_nodes, 8)
print("\nExperiment successful: PyG model processed the scene graph.")
