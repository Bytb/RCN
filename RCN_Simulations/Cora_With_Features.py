import os
import torch
import pickle
import numpy as np
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss
from HelperFunctions import cluster_with_kmeans

import torch.nn.functional as F
import random
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

SEEDS = [888, 0, 2024, 42, 328476]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_list = [7, 12]  # <-- change here if needed

dataset = "Cora"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(embeddings, true_labels, k):
    preds = cluster_with_kmeans(embeddings, k)

    y_np = true_labels.detach().cpu().numpy()
    Z = embeddings.detach().cpu().numpy()

    ari = adjusted_rand_score(y_np, preds)
    nmi = normalized_mutual_info_score(y_np, preds)
    sil = silhouette_score(Z, preds)

    return ari, nmi, sil


# ============================================================
# Graph + Feature Loader
# ============================================================

def load_cora_graph():

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "Data", "Graphs", "cora_graph.gpickle")
    file_path = os.path.abspath(file_path)

    with open(file_path, "rb") as f:
        G = pickle.load(f)

    dataset_pyg = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset_pyg[0]

    features = data.x.numpy()
    labels = data.y.numpy()

    for node in G.nodes():
        G.nodes[node]["label"] = int(labels[node])
        G.nodes[node]["features"] = features[node].tolist()

    node_list = list(G.nodes())
    mapping = {node: idx for idx, node in enumerate(node_list)}

    edge_index = []
    edge_weight = []

    for u, v in G.edges():

        u_idx = mapping[u]
        v_idx = mapping[v]

        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])

        w = G[u][v]["ret_n"]
        edge_weight.extend([w, w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    feature_list = [G.nodes[n]["features"] for n in node_list]

    x_features = torch.tensor(feature_list, dtype=torch.float)
    x_onehot = torch.eye(len(node_list))

    y = torch.tensor([G.nodes[n]["label"] for n in node_list], dtype=torch.long)

    return x_features.to(device), x_onehot.to(device), edge_index.to(device), edge_weight.to(device), y.to(device)


# ============================================================
# RCN Training
# ============================================================

def train_rcn(x, edge_index, edge_weight, y, k):

    model = CAGATAblationModel(
        in_dim=x.size(1),
        hidden_dim=8,
        out_dim=k,
        heads=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(200):

        model.train()
        optimizer.zero_grad()

        _, _, embeddings = model(x, edge_index, edge_weight)

        loss = combined_community_loss(
            embeddings=embeddings,
            edge_index=edge_index,
            edge_weight=edge_weight,
            lambda_mod=0.6,
            lambda_lap=0.001,
            lambda_contrast=0.0001,
            lambda_orth=0.000001,
            contrast_tau=0.5,
            contrast_variant="node",
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        _, _, embeddings = model(x, edge_index, edge_weight)

    return evaluate(embeddings, y, k)


# ============================================================
# MAIN ABLATION
# ============================================================

if __name__ == "__main__":

    os.makedirs("logs/Cora_Ablation", exist_ok=True)

    x_features, x_onehot, edge_index, rnbrw_weight, y = load_cora_graph()

    ones_weight = torch.ones(edge_index.shape[1], device=device)

    jobs = [
        ("onehot_unweighted", x_onehot, ones_weight),
        ("onehot_weighted", x_onehot, rnbrw_weight),
        ("features_unweighted", x_features, ones_weight),
        ("features_weighted", x_features, rnbrw_weight),
    ]

    total_runs = len(k_list) * len(jobs) * len(SEEDS)

    raw_results = []

    with tqdm(total=total_runs, desc="RCN Ablation") as pbar:

        for k in k_list:

            for job_name, x, w in jobs:

                for seed in SEEDS:

                    set_seed(seed)

                    ari, nmi, sil = train_rcn(x, edge_index, w, y, k)

                    raw_results.append({
                        "k": k,
                        "job": job_name,
                        "seed": seed,
                        "ARI": ari,
                        "NMI": nmi,
                        "Silhouette": sil
                    })

                    pbar.update(1)

    raw_df = pd.DataFrame(raw_results)

    raw_df.to_csv(
        "logs/Cora_Ablation/cora_rcn_ablation_raw.csv",
        index=False
    )

    summary_df = raw_df.groupby(["k", "job"]).agg(
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        NMI_mean=("NMI", "mean"),
        NMI_std=("NMI", "std"),
        Sil_mean=("Silhouette", "mean"),
        Sil_std=("Silhouette", "std"),
    ).reset_index()

    summary_df.to_csv(
        "logs/Cora_Ablation/cora_rcn_ablation_summary.csv",
        index=False
    )

    print("\nAblation complete.")
    print(summary_df)