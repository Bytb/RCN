import os
os.environ["OMP_NUM_THREADS"] = "1"

from Models.BaselineModels import DMoNModel
from Models.RCN import CAGATAblationModel
from HelperFunctions import get_topk_neighbor_mask, plot_embedding_2d, plot_elbow_and_silhouette, elbow_method
from sklearn.cluster import KMeans
from Models.LossFunctions import combined_community_loss
import matplotlib.pyplot as plt
import networkx as nx
#from Simulations.RCN_Simulations.ToyGraphs import *
from Data.Cora import load_cora_graph
from Data.PPI import load_ppi_graph
import random
import numpy as np
import torch

device = 'cpu'
seed = 328476
def set_seed(seed=120):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)  # or any fixed seed you want

# --- Step 1: Create a toy graph ---
def run_dmon(x, edge_index, edge_weight, max_k):
    x, edge_index, edge_weight = x.to(device), edge_index.to(device), edge_weight.to(device)
    # Build sparse adjacency matrix
    num_nodes = x.size(0)
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=x.device),
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=x.device
    ).to_dense()  # convert to dense for loss (only used in loss, not model)

    model = DMoNModel(
        in_dim=x.size(1),
        hidden_dim=64,
        num_clusters=32,
        dropout=0.3,
        lambda_orth=0.001
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        cluster_assignments = model(x.to(device), edge_index.to(device))
        loss = model.dmon_loss(cluster_assignments, adj.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if epoch == 200:
                embeddings = model(x.to(device), edge_index.to(device))
                # Use the softmax outputs as embedding for clustering metrics
                elbow_method(embeddings, max_k=max_k, model_name="DMoN")
    return embeddings

def run_rcn(x, edge_index, edge_weight, max_k):
    x, edge_index, edge_weight = x.to(device), edge_index.to(device), edge_weight.to(device)
    model = CAGATAblationModel(in_dim=x.size(1), hidden_dim=8, out_dim=32, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    with torch.no_grad():
        _, _, emb_init = model(x, edge_index, edge_weight)
        topk_mask = get_topk_neighbor_mask(emb_init, edge_index, edge_weight, use_rnbrw=True)

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
                        lambda_contrast=0.0001,
                        lambda_lap=0.001,
                        lambda_orth=0.1,
                        contrast_variant="node"
                    )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if epoch == 200:
                _, _, embeddings = model(x, edge_index, edge_weight)
                elbow_method(embeddings, max_k=max_k, model_name='RCN')
    return embeddings

def plot_predicted_communities(G, pos, embeddings, model_name, k_best):
    if hasattr(embeddings, 'detach'):
        embeddings = embeddings.detach().cpu().numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=k_best, n_init=10, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    # Assign color per cluster
    cmap = plt.cm.get_cmap('tab10', k_best)
    node_colors = [cmap(label) for label in labels]

    # Plot
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, edge_color='gray', width=2, font_weight='bold')
    plt.title(f"{model_name} Predicted Communities")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{model_name}_Predicted_Communities.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # --- Toy Graphs --- #
    # ring_of_cliques: break modularity via resolution limit - breaks DMoN
    # single_large_clique: trigger over segmentation -
    # ring_lattice: degeneracy in modularity solutions
    # unequal_cliques: unequal-size bias
    # star_with_leaf_links: degree heterogeity bias
    #G, pos, x, edge_index, edge_weight = ring_of_cliques()
    x, edge_index, y, edge_weight, G = load_cora_graph(use_onehot="True")
    k = 100
    # ---Elbow Methods---#
    # --- Step 2: Create basic node embeddings ---
    # dmon_embeddings = run_dmon(x, edge_index, edge_weight, k)
    rcn_embeddings = run_rcn(x, edge_index, edge_weight, k)
    # 1) Check for NaNs / infs
    Z = rcn_embeddings.detach().cpu().numpy()
    assert np.isfinite(Z).all(), "Found NaN/inf in embeddings"

    # plot_predicted_communities(G, pos, dmon_embeddings, "DMoN", k_best=6)
    # plot_predicted_communities(G, pos, rcn_embeddings, "RCN", k_best=8)

    # # 1) Just like before, but choose a method:
    # plot_embedding_2d(dmon_embeddings, title="DMoN Embeddings", method="pca")
    # plot_embedding_2d(dmon_embeddings, title="DMoN Embeddings", method="tsne", tsne_perplexity=25)
    #
    # # 4) Reproduce elbow, and also check silhouette to verify k:
    # ks, inertias, sils = plot_elbow_and_silhouette(dmon_embeddings, k_range=range(2, 16))

    #--- RCN ---#
    plot_embedding_2d(rcn_embeddings, title="RCN_Embeddings", method="pca")
    plot_embedding_2d(rcn_embeddings, title="RCN_Embeddings", method="tsne", tsne_perplexity=25)

    # 4) Reproduce elbow, and also check silhouette to verify k:
    ks, inertias, sils = plot_elbow_and_silhouette(rcn_embeddings, k_range=range(2, k))

    print("✅ Finished all simulations. Results saved.")