import torch
import pickle
import numpy as np
import os

def load_facebook_graph(use_onehot=False, device="cpu"):
    # === Load gpickle ===
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Graphs", "facebook_graph.gpickle")
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    # === Check that all edges have ret_n ===
    for u, v in G.edges():
        if 'ret_n' not in G[u][v]:
            raise ValueError(f"Missing 'ret_n' on edge ({u}, {v})")

    # === Node list + mapping
    node_list = list(G.nodes())
    mapping = {node: idx for idx, node in enumerate(node_list)}

    # === Build edge_index and edge_weight
    edge_index = []
    edge_weight = []

    for u, v in G.edges():
        u_idx = mapping[u]
        v_idx = mapping[v]
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])
        w = G[u][v]['ret_n']
        edge_weight.extend([w, w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # === Node features
    if use_onehot:
        x = torch.eye(len(node_list))
    else:
        degrees = np.array([G.degree(n) for n in node_list], dtype=np.float32).reshape(-1, 1)
        x = torch.tensor(degrees, dtype=torch.float)

    # === Labels (Ego typically unlabeled) → fallback to zeros if no 'label'
    y_vals = []
    has_label_attr = all(('label' in G.nodes[n]) for n in node_list)
    if has_label_attr:
        for node in node_list:
            y_vals.append(int(G.nodes[node]['label']))
    else:
        y_vals = [0] * len(node_list)  # placeholder labels for API compatibility
    y = torch.tensor(y_vals, dtype=torch.long)

    return x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device), G
