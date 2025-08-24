import torch
import torch.nn as nn
import random
import math
import itertools
import os
from tqdm import tqdm

# --- Vendored and simplified code from github.com/rektomar/SparsePGC ---
# This section contains the necessary classes to build the SparsePGC model,
# adapted to be self-contained in a single file for this experiment.


# Source: models/einsum/Graph.py
class Graph:
    def __init__(self, edges=None, num_variables=None):
        self.edges = edges
        self.num_variables = num_variables

    @staticmethod
    def binary_tree(num_var, num_leaves, structure):
        if structure == "balanced":
            raise NotImplementedError
        elif structure == "half":
            edges = []
            for i in range(num_var - 1):
                edges.append([i, i + 1])
            return Graph(edges=edges, num_variables=num_var)
        else:
            raise ValueError(f"Unknown graph structure {structure}")


# Source: models/einsum/ExponentialFamilyArray.py
class ExponentialFamilyArray:
    def __init__(self, exponential_family_args):
        self.args = exponential_family_args


class CategoricalArray(ExponentialFamilyArray):
    def __init__(self, exponential_family_args):
        super().__init__(exponential_family_args)
        self.K = self.args["K"]

    def get_log_potential(self, params, x, marginalized):
        if marginalized:
            return params[..., 0]
        else:
            return torch.sum(x * params, dim=-1)


# Source: models/einsum/EinsumNetwork.py
class EinsumNetwork(nn.Module):
    class Args:
        def __init__(
            self,
            num_var,
            num_dims,
            num_input_distributions,
            num_sums,
            num_classes,
            exponential_family,
            exponential_family_args,
            use_em=False,
        ):
            self.num_var = num_var
            self.num_dims = num_dims
            self.num_input_distributions = num_input_distributions
            self.num_sums = num_sums
            self.num_classes = num_classes
            self.exponential_family = exponential_family
            self.exponential_family_args = exponential_family_args
            self.use_em = use_em

    def __init__(self, graph, args):
        super().__init__()
        self.graph = graph
        self.args = args
        self.pd_entries_params = None

    def initialize(self):
        p_list = []
        for i, edges in enumerate(self.graph.edges):
            # simplified from source: only one region graph
            child_1_vars = [edges[0]]
            child_2_vars = [edges[1]]
            parent_vars = child_1_vars + child_2_vars

            if i == 0:
                # Leaf layer
                p = torch.randn(
                    self.args.num_input_distributions,
                    self.args.num_sums,
                    self.args.exponential_family_args["K"],
                )
            else:
                # Sum-product layer
                p = torch.randn(
                    self.args.num_sums, self.args.num_sums, self.args.num_sums
                )

            p_list.append(nn.Parameter(p))

        # Root layer
        p_list.append(
            nn.Parameter(torch.randn(self.args.num_sums, self.args.num_classes))
        )
        self.pd_entries_params = nn.ParameterList(p_list)

    def forward(self, x, marginalized=False):
        leaf_log_potentials = self.args.exponential_family(
            self.args.exponential_family_args
        ).get_log_potential(self.pd_entries_params[0], x, marginalized)

        edge_log_potentials = [leaf_log_potentials]
        for i in range(1, len(self.graph.edges)):
            p = torch.log_softmax(self.pd_entries_params[i], dim=-1)
            prev_potential = edge_log_potentials[-1]
            new_potential = torch.logsumexp(prev_potential.unsqueeze(-1) + p, dim=-2)
            edge_log_potentials.append(new_potential)

        root_p = torch.log_softmax(self.pd_entries_params[-1], dim=0)
        ll = torch.logsumexp(edge_log_potentials[-1] + root_p, dim=-1)
        return ll


# Source: models/backend.py
class BTreeSPN(EinsumNetwork):
    def __init__(self, nd, nk, nc, nl, ns, ni):
        args = EinsumNetwork.Args(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=nc,
            exponential_family=CategoricalArray,
            exponential_family_args={"K": nk},
            use_em=False,
        )
        graph = Graph.binary_tree(nd, nl, "half")
        super().__init__(graph, args)
        self.initialize()


# Source: models/sparse_pgc.py (heavily simplified)
class SparsePGC(nn.Module):
    def __init__(self, model_hpars):
        super().__init__()
        self.num_nodes = model_hpars["max_nodes"]
        self.node_vocab_size = model_hpars["node_vocab_size"]

        self.node_spn = BTreeSPN(
            self.num_nodes, self.node_vocab_size, **model_hpars["node_spn_hpars"]
        )
        self.edge_spn = BTreeSPN(
            self.num_nodes * (self.num_nodes - 1) // 2,
            2,
            **model_hpars["edge_spn_hpars"],
        )

    def forward(self, x, adj):
        # x: (batch, num_nodes)
        # adj: (batch, num_nodes, num_nodes)
        batch_size = x.shape[0]
        x_onehot = nn.functional.one_hot(x, self.node_vocab_size).float()

        # upper triangular of adj matrix
        adj_flat = adj[
            :,
            torch.triu_indices(self.num_nodes, self.num_nodes, 1).unbind(1)[0],
            torch.triu_indices(self.num_nodes, self.num_nodes, 1).unbind(1)[1],
        ]
        adj_flat_onehot = nn.functional.one_hot(adj_flat.long(), 2).float()

        node_ll = self.node_spn(x_onehot.unsqueeze(2))
        edge_ll = self.edge_spn(adj_flat_onehot.unsqueeze(2))

        return node_ll + edge_ll

    def sample(self, num_samples):
        with torch.no_grad():
            node_samples_onehot = self.node_spn.sample(num_samples).squeeze(-2)
            edge_samples_onehot = self.edge_spn.sample(num_samples).squeeze(-2)

            node_labels = torch.argmax(node_samples_onehot, dim=-1)
            edge_labels_flat = torch.argmax(edge_samples_onehot, dim=-1)

            adj_matrices = []
            for i in range(num_samples):
                adj = torch.zeros(
                    self.num_nodes, self.num_nodes, device=edge_labels_flat.device
                )
                indices = torch.triu_indices(self.num_nodes, self.num_nodes, 1)
                adj[indices[0], indices[1]] = edge_labels_flat[i]
                adj = adj + adj.T
                adj_matrices.append(adj)

            return node_labels, torch.stack(adj_matrices)


# --- Experiment Setup ---


def get_synthetic_data(num_samples, max_nodes, vocab):
    """Generates a synthetic dataset of simple scene graphs."""
    graphs = []
    templates = [
        # cat on mat
        {"nodes": [vocab["cat"], vocab["mat"]], "adj": [[0, 1], [1, 0]]},
        # box next to chair
        {"nodes": [vocab["box"], vocab["chair"]], "adj": [[0, 1], [1, 0]]},
        # cat on chair, which is next to box
        {
            "nodes": [vocab["cat"], vocab["chair"], vocab["box"]],
            "adj": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        },
        # single object
        {"nodes": [vocab["chair"]], "adj": [[0]]},
    ]

    for _ in range(num_samples):
        template = random.choice(templates)

        nodes = template["nodes"]
        adj = template["adj"]

        # Pad nodes
        padded_nodes = nodes + [vocab["pad"]] * (max_nodes - len(nodes))
        # Pad adjacency matrix
        padded_adj = torch.zeros(max_nodes, max_nodes)
        padded_adj[: len(nodes), : len(nodes)] = torch.tensor(adj, dtype=torch.float)

        graphs.append(
            {"nodes": torch.tensor(padded_nodes, dtype=torch.long), "adj": padded_adj}
        )

    return graphs


if __name__ == "__main__":
    # Hyperparameters
    MAX_NODES = 5
    NODE_VOCAB = {"pad": 0, "cat": 1, "mat": 2, "box": 3, "chair": 4}
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Model hyperparameters, inspired by SOURCE repo's configs
    model_hpars = {
        "max_nodes": MAX_NODES,
        "node_vocab_size": len(NODE_VOCAB),
        "node_spn_hpars": {"nc": 1, "nl": 3, "ns": 16, "ni": 16},
        "edge_spn_hpars": {"nc": 1, "nl": 3, "ns": 16, "ni": 16},
    }

    # Data
    dataset = get_synthetic_data(512, MAX_NODES, NODE_VOCAB)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Model & Optimizer
    model = SparsePGC(model_hpars).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            nodes = batch["nodes"].to(DEVICE)
            adj = batch["adj"].to(DEVICE)

            optimizer.zero_grad()
            ll = model(nodes, adj)
            loss = -ll.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average NLL: {avg_loss:.4f}")

    # Sampling
    print("\nTraining finished. Generating samples...")
    model.eval()
    with torch.no_grad():
        # Monkey-patch sample method for BTreeSPN (simplified for PoC)
        def sample_btree(self, num_samples):
            # This is a simplified, non-ancestral sampling for demonstration.
            # A real implementation would use ancestral sampling.
            device = next(self.parameters()).device
            pd_params_softmax = [
                torch.softmax(p, dim=-1) for p in self.pd_entries_params
            ]
            final_dist = (
                pd_params_softmax[0].mean(dim=0).mean(dim=0)
            )  # crude approximation
            indices = torch.multinomial(
                final_dist, self.args.num_var * num_samples, replacement=True
            )
            samples = nn.functional.one_hot(
                indices, num_classes=self.args.exponential_family_args["K"]
            ).float()
            return samples.reshape(
                num_samples,
                self.args.num_var,
                1,
                self.args.exponential_family_args["K"],
            )

        BTreeSPN.sample = sample_btree

        sampled_nodes, sampled_adjs = model.sample(num_samples=10)

        print("\n--- Generated Scene Graphs ---")
        for i in range(10):
            print(f"\nSample {i+1}:")
            node_names = [
                list(NODE_VOCAB.keys())[j] for j in sampled_nodes[i].cpu().numpy()
            ]
            print(f"  Nodes: {node_names}")
            print(f"  Adjacency Matrix:\n{sampled_adjs[i].cpu().numpy().astype(int)}")
