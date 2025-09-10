import os
import torch
import pickle
import numpy as np
from torch_geometric.datasets import Planetoid

def load_cora_graph(use_onehot=False, device="cpu"):
    # === Load gpickle ===
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Graphs", "cora_graph.gpickle")
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    # === Load true labels from PyG ===
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    raw_labels = dataset[0].y.numpy()  # shape: [2708]

    # === Inject true labels into G ===
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["label"] = int(raw_labels[i])

    # === Check that all edges have ret_n ===
    for u, v in G.edges():
        if 'ret_n' not in G[u][v]:
            raise ValueError(f"Missing 'ret_n' on edge ({u}, {v})")

    # === Create mapping and reverse-mapping ===
    node_list = list(G.nodes())
    mapping = {node: idx for idx, node in enumerate(node_list)}

    # === Build edge_index and edge_weight ===
    edge_index = []
    edge_weight = []

    for u, v in G.edges():
        u_idx = mapping[u]
        v_idx = mapping[v]
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])  # undirected
        w = G[u][v]['ret_n']
        edge_weight.extend([w, w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # === Node features ===
    if use_onehot:
        num_nodes = len(node_list)
        x = torch.eye(num_nodes)
    else:
        deg = np.array([G.degree(n) for n in node_list], dtype=np.float32).reshape(-1, 1)
        x = torch.tensor(deg, dtype=torch.float)

    # === Labels ===
    y = torch.tensor([G.nodes[n]["label"] for n in node_list], dtype=torch.long)

    return x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device), G

