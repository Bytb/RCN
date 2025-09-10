import torch
import networkx as nx
from rnbrw.weights import compute_weights
import os
import pickle

def load_polbooks_graph(use_onehot=True, device='cuda'):
    """
    Loads the PolBooks graph from .gpickle with RNBRW edge weights and node features.

    Args:
        use_onehot (bool): Whether to use one-hot node features. If False, uses degree.
        device (str): 'cpu' or 'cuda'

    Returns:
        x (Tensor): Node features [N, D]
        edge_index (Tensor): Edge list [2, E]
        y (Tensor): Community labels [N]
        rnbrw_weights (Tensor): RNBRW edge weights [E]
        G (networkx.Graph): Original graph
    """
    # Load the graph
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Graphs", "polbooks_graph.gpickle")
    with open(file_path, "rb") as f:
        G = pickle.load(f)
        G = G.to_undirected()

    # Sort nodes by ID for consistency
    nodes = sorted(G.nodes())
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    G = nx.relabel_nodes(G, node_mapping)

    # Extract labels
    label_map = {'l': 0, 'c': 1, 'n': 2}
    labels = [label_map.get(G.nodes[n].get("label", "n"), 2) for n in range(len(G.nodes))]
    y = torch.tensor(labels, dtype=torch.long, device=device)

    # Features
    if use_onehot:
        x = torch.eye(len(G.nodes), device=device)
    else:
        degrees = torch.tensor([val for _, val in G.degree()], dtype=torch.float32, device=device).unsqueeze(1)
        x = degrees

    # Edge index (undirected, doubled)
    edges = list(G.edges())
    undirected_edges = edges + [(v, u) for u, v in edges]
    edge_index = torch.tensor(undirected_edges, dtype=torch.long, device=device).T

    # RNBRW weights
    G_rnbrw = compute_weights(G.copy(), nsim=len(edges), n_jobs=8)
    edge2weight = {
        (min(u, v), max(u, v)): d['ret_n']
        for u, v, d in G_rnbrw.edges(data=True)
    }

    weights_expanded = []
    src, dst = edge_index.cpu().numpy()
    for u, v in zip(src, dst):
        a, b = min(u, v), max(u, v)
        weights_expanded.append(edge2weight.get((a, b), 1.0))
    rnbrw_weights = torch.tensor(weights_expanded, dtype=torch.float32, device=device)

    return x, edge_index, y, rnbrw_weights, G
