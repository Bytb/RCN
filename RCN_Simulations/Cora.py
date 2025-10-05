import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from networkx.algorithms.community import louvain_communities
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from Models.LossFunctions import combined_community_loss, modularity_loss_nornbrw
from HelperFunctions import cluster_with_kmeans, generate_final_results
from Data.Cora import load_cora_graph

from Models.GAT_Attention import GAT
from Models.GCN import GCN
from Models.GraphSAGE import GraphSAGE
from Models.RCN import CAGATAblationModel
from Models.BaselineModels import Encoder, GRACEModel, get_augmented_views, DAEGCModel, SDCNModel, BGRLModel, DMoNModel
from torch_geometric.nn import DeepGraphInfomax, GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
from torch_geometric.utils import to_dense_adj

SEEDS = [888, 0, 2024, 42, 328476]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = "Cora(7)"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(embeddings, true_labels, k):
    preds = cluster_with_kmeans(embeddings, k)  # keep as-is if your helper accepts torch
    y_np = true_labels.detach().cpu().numpy() if torch.is_tensor(true_labels) else np.asarray(true_labels)
    Z = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    ari = adjusted_rand_score(y_np, preds)
    nmi = normalized_mutual_info_score(y_np, preds)
    sil = silhouette_score(Z, preds)
    return ari, nmi, sil, preds

def run_gnn_model(ModelClass, model_name, x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/{model_name}"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)
        model = ModelClass(x.size(1), num_clusters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        loss_log, sil_log = [], []

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            _, _, embeddings = model(Data(x=x, edge_index=edge_index))
            Q = torch.nn.functional.softmax(embeddings, dim=1)
            loss = modularity_loss_nornbrw(Q, edge_index)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _, _, embeddings_eval = model(Data(x=x, edge_index=edge_index))
                ari, nmi, sil, _ = evaluate(embeddings_eval, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)
        # Loss vs Epoch
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (RCN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        # Silhouette vs Epoch
        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset,
            "Model": model_name,
            "Loss Function": "Modularity",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_rcn(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/RCN"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)
        model = CAGATAblationModel(in_dim=x.size(1), hidden_dim=8, out_dim=num_clusters, heads=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        with torch.no_grad():
            _, _, emb_init = model(x, edge_index, edge_weight)

        loss_log, sil_log = [], []

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            _, _, embeddings = model(x, edge_index, edge_weight)
            q_soft = torch.nn.functional.softmax(embeddings, dim=1)
            loss = combined_community_loss(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weight=edge_weight,
                lambda_mod=0.6,
                lambda_lap=0.001,  # Laplacian/L2 (in-loss)
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
                ari, nmi, sil, preds = evaluate(embeddings, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)
        # Loss vs Epoch
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (RCN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        # Silhouette vs Epoch
        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (RCN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()
        results.append({
            "Dataset": dataset,
            "Model": "RCN",
            "Loss Function": "CustomRNBRW",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_louvain(G, y, weighted=False):
    results = []
    y_true = y.cpu().numpy()
    weight_key = 'weight' if weighted else None
    partition = louvain_communities(G, weight=weight_key, resolution=1)
    cluster_map = {}
    for cid, nodes in enumerate(partition):
        for n in nodes:
            cluster_map[n] = cid
    labels = [cluster_map[i] for i in range(len(cluster_map))]
    embeddings = np.eye(len(labels))

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = silhouette_score(embeddings, labels)

    results.append({
        "Dataset": dataset,
        "Model": "Louvain (Weighted)" if weighted else "Louvain (Unweighted)",
        "Loss Function": "None",
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Silhouette Score": round(sil, 4),
    })
    return results

class GCNEncoderDGI(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def run_dgi(x, edge_index, y, num_clusters):
    results = []
    logdir = f"logs/{dataset}/DGI"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)
        encoder = GCNEncoderDGI(x.size(1), 64)
        summary = lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0))
        model = DeepGraphInfomax(64, encoder, summary, corruption=corruption).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        loss_log, sil_log = [], []
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            pos_z, neg_z, summary_vec = model(x, edge_index)
            loss = model.loss(pos_z, neg_z, summary_vec)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                z, _, _ = model(x, edge_index)
                ari, nmi, sil, _ = evaluate(z.cpu(), y.cpu(), num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)
        # Loss vs Epoch
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (DGI, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        # Silhouette vs Epoch
        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (DGI, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset,
            "Model": "DGI",
            "Loss Function": "--NA--",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_grace(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/GRACE"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)

        # Model and optimizer
        encoder = Encoder(in_channels=x.size(1), out_channels=128, activation=torch.nn.PReLU(), base_model=GCNConv, k=2)
        model = GRACEModel(encoder=encoder, num_hidden=128, num_proj_hidden=128, tau=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_log, sil_log = [], []

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()

            # Generate two augmented views
            x1, x2, e1, e2 = get_augmented_views(x, edge_index, drop_edge_rate=0.2, drop_feat_rate=0.3)

            # Forward pass
            z1 = model(x1.to(device), e1.to(device))
            z2 = model(x2.to(device), e2.to(device))
            loss = model.loss(z1, z2)

            loss.backward()
            optimizer.step()

            # Evaluation on original graph
            model.eval()
            with torch.no_grad():
                z = model(x.to(device), edge_index.to(device))
                ari, nmi, sil, _ = evaluate(z, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)

        # Loss vs Epoch
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (GRACE, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        # Silhouette vs Epoch
        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (GRACE, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset,
            "Model": "GRACE",
            "Loss Function": "Contrastive",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_daegc(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/DAEGC"
    os.makedirs(logdir, exist_ok=True)

    # Convert edge_index to dense adjacency matrix [N x N]
    A_true = to_dense_adj(edge_index)[0]  # shape: [N, N]

    for seed in SEEDS:
        set_seed(seed)

        model = DAEGCModel(
            in_channels=x.size(1),
            out_channels=128,
            num_clusters=num_clusters,
            dropout=0.0
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_log, sil_log = [], []

        # ---- KMeans init ----
        with torch.no_grad():
            model.eval()
            z_init, _, _ = model(x.to(device), edge_index.to(device))
            kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(z_init.cpu().numpy())
            model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        # ---- Training ----
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()

            z, A_pred, q = model(x.to(device), edge_index.to(device))
            p = model.target_distribution(q.detach())

            loss_recon = model.reconstruction_loss(A_pred, A_true.to(device))
            loss_cluster = model.clustering_loss(q, p)
            loss = loss_recon + loss_cluster

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                z_eval, _, _ = model(x.to(device), edge_index.to(device))
                ari, nmi, sil, _ = evaluate(z_eval, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        # ---- Save Logs ----
        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)

        # ---- Plots ----
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (DAEGC, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (DAEGC, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        # ---- Store Results ----
        results.append({
            "Dataset": dataset,
            "Model": "DAEGC",
            "Loss Function": "Recon+KL",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_sdcn(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/SDCN"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)

        # Define the model
        model = SDCNModel(
            in_dim=x.size(1),
            hidden_dims=[500, 500, 2000, 128],
            num_clusters=num_clusters,
            alpha=0.5
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_log, sil_log = [], []

        # ---- KMeans init ----
        with torch.no_grad():
            model.eval()
            ae_embed, _, _, _ = model(x.to(device), edge_index.to(device))
            kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(ae_embed.cpu().numpy())
            model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        # ---- Training ----
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()

            ae_embed, x_hat, z, q = model(x.to(device), edge_index.to(device))
            p = model.target_distribution(q.detach())

            loss_recon = model.reconstruction_loss(x_hat, x.to(device))
            loss_cluster = model.clustering_loss(q, p)
            loss = loss_cluster + 0.1 * loss_recon  # α=0.1 weighting

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _, _, z_eval, _ = model(x.to(device), edge_index.to(device))
                ari, nmi, sil, _ = evaluate(z_eval, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        # ---- Save Logs ----
        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)

        # ---- Plots ----
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (SDCN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (SDCN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset,
            "Model": "SDCN",
            "Loss Function": "Recon+KL",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_bgrl(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/BGRL"
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)

        model = BGRLModel(in_dim=x.size(1), hidden_dim=128, momentum=0.99).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_log, sil_log = [], []

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()

            # Generate two augmented graph views
            x1, x2, e1, e2 = get_augmented_views(x, edge_index, drop_edge_rate=0.2, drop_feat_rate=0.3)

            # Forward pass
            p1, p2, z1_target, z2_target = model(x1.to(device), e1.to(device), x2.to(device), e2.to(device))

            # Loss and backprop
            loss = model.loss_fn(p1, p2, z1_target, z2_target)
            loss.backward()
            optimizer.step()

            # Momentum update of target encoder
            model.momentum_update()

            # Evaluation on current embeddings
            model.eval()
            with torch.no_grad():
                z = model.embed(x.to(device), edge_index.to(device))
                ari, nmi, sil, _ = evaluate(z, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        # ---- Save Logs ----
        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)

        # ---- Plots ----
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (BGRL, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (BGRL, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        # ---- Store Results ----
        results.append({
            "Dataset": dataset,
            "Model": "BGRL",
            "Loss Function": "Bootstrap Cosine",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results

def run_dmon(x, edge_index, y, edge_weight, num_clusters):
    results = []
    logdir = f"logs/{dataset}/DMoN"
    os.makedirs(logdir, exist_ok=True)

    # Build sparse adjacency matrix
    num_nodes = x.size(0)
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=x.device),
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=x.device
    ).to_dense()  # convert to dense for loss (only used in loss, not model)

    for seed in SEEDS:
        set_seed(seed)
        model = DMoNModel(
            in_dim=x.size(1),
            hidden_dim=64,
            num_clusters=num_clusters,
            dropout=0.3,
            lambda_orth=0.001
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        loss_log, sil_log = [], []

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            cluster_assignments = model(x.to(device), edge_index.to(device))
            loss = model.dmon_loss(cluster_assignments, adj.to(device))
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                cluster_assignments = model(x.to(device), edge_index.to(device))
                # Use the softmax outputs as embedding for clustering metrics
                ari, nmi, sil, _ = evaluate(cluster_assignments, y, num_clusters)
                loss_log.append(loss.item())
                sil_log.append(sil)

        # Save logs
        pd.DataFrame({"Loss": loss_log, "Silhouette": sil_log}).to_csv(f"{logdir}/seed_{seed}.csv", index=False)

        # Plots
        plt.figure()
        plt.plot(loss_log, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs (DMoN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/loss_seed{seed}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(sil_log, label="Silhouette Score")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs Epochs (DMoN, seed {seed})")
        plt.legend()
        plt.savefig(f"{logdir}/silhouette_seed{seed}.png", dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset,
            "Model": "DMoN",
            "Loss Function": "Modularity + Orthogonality",
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette Score": round(sil, 4),
        })

    return results


if __name__ == "__main__":

    # === Load Dataset ===
    x, edge_index, y, edge_weight, G = load_cora_graph(use_onehot=True, device=device)
    x, edge_index, y, edge_weight = x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device)
    num_clusters = 7

    all_results = []

    # === GNNs ===
    tasks = [
        ("GCN", lambda: run_gnn_model(GCN, "GCN", x, edge_index, y, edge_weight, num_clusters)),
        ("GraphSAGE", lambda: run_gnn_model(GraphSAGE, "GraphSAGE", x, edge_index, y, edge_weight, num_clusters)),
        ("GAT", lambda: run_gnn_model(GAT, "GAT", x, edge_index, y, edge_weight, num_clusters)),
        ("DGI", lambda: run_dgi(x, edge_index, y, num_clusters)),
        ("GRACE", lambda: run_grace(x, edge_index, y, edge_weight, num_clusters)),
        ("DAEGC", lambda: run_daegc(x, edge_index, y, edge_weight, num_clusters)),
        ("SDCN", lambda: run_sdcn(x, edge_index, y, edge_weight, num_clusters)),
        ("BGRL", lambda: run_bgrl(x, edge_index, y, edge_weight, num_clusters)),
        ("DMoN", lambda: run_dmon(x, edge_index, y, edge_weight, num_clusters)),
        ("RCN", lambda: run_rcn(x, edge_index, y, edge_weight, num_clusters)),
    ]

    pbar = tqdm(tasks, total=len(tasks), desc="Models")
    for name, task in pbar:
        pbar.set_postfix_str(name)
        all_results += task()

    # === Louvain ===
    all_results += run_louvain(G.copy(), y, weighted=False)
    nx.set_edge_attributes(G, { (int(u), int(v)): float(edge_weight[i].item()) for i, (u, v) in enumerate(edge_index.T.cpu()) }, "weight")
    all_results += run_louvain(G.copy(), y, weighted=True)

    logdir = f"logs/{dataset}/FinalResults"
    os.makedirs(logdir, exist_ok=True)

    pd.DataFrame(all_results).to_csv(f"{logdir}/{dataset}_Results.csv", index=False)

    # ===========================
    # Call at end of __main__
    # ===========================
    # After you save the master CSV:
    #   logdir = f"logs/{dataset}/FinalResults"
    #   pd.DataFrame(all_results).to_csv(f"{logdir}/{dataset}_Results.csv", index=False)
    # then call:
    generate_final_results(dataset, base_dir="../../../OneDrive - University of South Florida/RCN_Neurips/RCN_Simulations/logs", final_subdir="FinalResults")

    print("✅ Finished all simulations. Results saved.")
