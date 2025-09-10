import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rnbrw.weights import compute_weights

# ========== 1) Resolution limit: Ring of cliques ==========
def ring_of_cliques(num_cliques=8, clique_size=5):
    # --- Build ring of cliques connected by single edges ---
    G = nx.Graph()
    cliques = []
    node_offset = 0
    for c in range(num_cliques):
        nodes = list(range(node_offset, node_offset + clique_size))
        G.add_nodes_from(nodes)
        # fully connect the clique
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                G.add_edge(nodes[i], nodes[j])
        cliques.append(nodes)
        node_offset += clique_size

    # connect cliques in a ring by single edges
    for c in range(num_cliques):
        a = cliques[c][-1]
        b = cliques[(c + 1) % num_cliques][0]
        G.add_edge(a, b)

    # --- RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = []
    for u, v in edges_und:
        rnb_list.append(G_rnb[u][v].get('ret_n', 0.0))
    rnb_list = np.array(rnb_list) * 2.0
    for (u, v), w in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = w

    # --- One-hot features ---
    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes)

    # --- Edge index & weights ---
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Layout: each clique on a small circle; cliques on a big circle ---
    pos = {}
    R = 8.0
    for c, nodes in enumerate(cliques):
        angle = 2 * np.pi * c / num_cliques
        cx, cy = R * np.cos(angle), R * np.sin(angle)
        small = nx.circular_layout(G.subgraph(nodes))
        for n in nodes:
            sx, sy = small[n]
            pos[n] = (cx + 1.2 * sx, cy + 1.2 * sy)

    # --- Draw ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, edge_color='gray', width=2, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Resolution Limit: Ring of Cliques")
    plt.axis("off"); plt.tight_layout(); plt.show()

    return G, pos, x, edge_index, edge_weight


# ========== 2) Over-segmentation: Single large clique ==========
def single_large_clique(n=30):
    # --- Single dense community (clique) ---
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j)

    # --- RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = np.array([G_rnb[u][v].get('ret_n', 0.0) for u, v in edges_und]) * 2.0
    for (u, v), w in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = w

    # --- One-hot features ---
    x = torch.eye(G.number_of_nodes())

    # --- Edge index & weights ---
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Layout ---
    pos = nx.circular_layout(G)

    # --- Draw ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(9, 9))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=550, edge_color='gray', width=2, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
    plt.title("Over-Segmentation: Single Large Clique")
    plt.axis("off"); plt.tight_layout(); plt.show()

    return G, pos, x, edge_index, edge_weight


# ========== 3) Degeneracy: Ring lattice (many near-equal partitions) ==========
def ring_lattice(n=40, k=4):
    """
    n nodes on a ring; each node connected to k/2 neighbors on each side (k even).
    """
    assert k % 2 == 0 and k < n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for s in range(1, k // 2 + 1):
            G.add_edge(i, (i + s) % n)
            G.add_edge(i, (i - s) % n)

    # --- RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = np.array([G_rnb[u][v].get('ret_n', 0.0) for u, v in edges_und]) * 2.0
    for (u, v), w in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = w

    # --- One-hot features ---
    x = torch.eye(G.number_of_nodes())

    # --- Edge index & weights ---
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Layout ---
    pos = nx.circular_layout(G)

    # --- Draw ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='wheat', node_size=520, edge_color='gray', width=2, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)
    plt.title("Degeneracy: Ring Lattice")
    plt.axis("off"); plt.tight_layout(); plt.show()

    return G, pos, x, edge_index, edge_weight


# ========== 4) Unequal-size bias: Big clique + small clique with bridge ==========
def unequal_cliques(big=20, small=5):
    G = nx.Graph()
    big_nodes = list(range(big))
    small_nodes = list(range(big, big + small))
    G.add_nodes_from(big_nodes + small_nodes)

    # fully connect each clique
    for i in range(big):
        for j in range(i + 1, big):
            G.add_edge(big_nodes[i], big_nodes[j])
    for i in range(small):
        for j in range(i + 1, small):
            G.add_edge(small_nodes[i], small_nodes[j])

    # bridge between cliques
    G.add_edge(big_nodes[0], small_nodes[0])

    # --- RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = np.array([G_rnb[u][v].get('ret_n', 0.0) for u, v in edges_und]) * 2.0
    for (u, v), w in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = w

    # --- One-hot features ---
    x = torch.eye(G.number_of_nodes())

    # --- Edge index & weights ---
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Layout: two offset circles ---
    pos = {}
    big_circle = nx.circular_layout(G.subgraph(big_nodes))
    small_circle = nx.circular_layout(G.subgraph(small_nodes))
    for n in big_nodes:
        x0, y0 = big_circle[n]
        pos[n] = (x0 - 4.0, y0)
    for n in small_nodes:
        x0, y0 = small_circle[n]
        pos[n] = (x0 + 4.0, y0 + 0.5)

    # --- Draw ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, edge_color='gray', width=2, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Unequal-Size Bias: Big Clique + Small Clique")
    plt.axis("off"); plt.tight_layout(); plt.show()

    return G, pos, x, edge_index, edge_weight


# ========== 5) Degree heterogeneity bias: Star with a few weak cross-leaf links ==========
def star_with_leaf_links(num_leaves=20, extra_links=5, seed=42):
    rng = np.random.default_rng(seed)

    # --- Star ---
    G = nx.Graph()
    center = 0
    leaves = list(range(1, num_leaves + 1))
    G.add_node(center); G.add_nodes_from(leaves)
    for u in leaves:
        G.add_edge(center, u)

    # --- Add a handful of cross-leaf edges (weak structure) ---
    # (no weights here; RNBRW will still reflect cycle participation differences)
    for _ in range(extra_links):
        a, b = rng.choice(leaves, size=2, replace=False)
        if not G.has_edge(a, b):
            G.add_edge(a, b)

    # --- RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = np.array([G_rnb[u][v].get('ret_n', 0.0) for u, v in edges_und]) * 2.0
    for (u, v), w in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = w

    # --- One-hot features ---
    x = torch.eye(G.number_of_nodes())

    # --- Edge index & weights ---
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Layout: star in circle; center in middle ---
    pos = {}
    theta = np.linspace(0, 2*np.pi, len(leaves), endpoint=False)
    R = 3.0
    for i, leaf in enumerate(leaves):
        pos[leaf] = (R*np.cos(theta[i]), R*np.sin(theta[i]))
    pos[center] = (0.0, 0.0)

    # --- Draw ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(9, 8))
    nx.draw(G, pos, with_labels=True, node_color='pink', node_size=600, edge_color='gray', width=2, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
    plt.title("Degree Heterogeneity Bias: Star + Cross-Leaf Links")
    plt.axis("off"); plt.tight_layout(); plt.show()

    return G, pos, x, edge_index, edge_weight

def star_cluster():
    # --- Initialize Graph ---
    G = nx.Graph()

    # --- Create Star Cluster ---
    star_center = 0
    star_leaves = list(range(1, 9))  # 8 leaves
    G.add_node(star_center)
    G.add_nodes_from(star_leaves)
    for leaf in star_leaves:
        G.add_edge(star_center, leaf)

    # --- Create Dense Cluster (Fully Connected Subgraph) ---
    dense_nodes = list(range(9, 14))  # 5 nodes
    G.add_nodes_from(dense_nodes)
    for i in range(len(dense_nodes)):
        for j in range(i + 1, len(dense_nodes)):
            G.add_edge(dense_nodes[i], dense_nodes[j])

    # --- Connect Star Center to One Node in Dense Cluster ---
    G.add_edge(star_center, dense_nodes[0])  # Connect star center to dense node 9

    # Compute RNBRW weights (ret_n)
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    rnb_list = []
    edges_und = list(G_rnb.edges())

    for u, v in edges_und:
        try:
            rnb_list.append(G_rnb[u][v]['ret_n'])
        except KeyError:
            rnb_list.append(0.0)
    rnb_list = np.array(rnb_list)
    rnb_list = rnb_list * 2

    # Attach to data
    num_nodes = G.number_of_nodes()
    rnbrw_tensor = torch.zeros((num_nodes, num_nodes))
    for (u, v), weight in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = weight

    # --- Convert to PyG Data ---
    # Use one-hot features for each node
    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes)

    # Convert to edge_index
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()

    # RNBRW edge weights
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Custom layout ---
    pos = {}

    # Star layout: circular
    theta = np.linspace(0, 2 * np.pi, len(star_leaves), endpoint=False)
    radius = 2.5
    for i, leaf in enumerate(star_leaves):
        pos[leaf] = (radius * np.cos(theta[i]), radius * np.sin(theta[i]))
    pos[star_center] = (0, 0)

    # Dense cluster: circular, shifted
    # Shift and assign positions for dense cluster
    dense_circle = nx.circular_layout(G.subgraph(dense_nodes))
    for node in dense_nodes:
        x_pos, y_pos = dense_circle[node]
        pos[node] = (x_pos + 5, y_pos - 1.5)

    # Swap node 9 and 11 to improve visual spacing
    pos[9], pos[11] = pos[11], pos[9]

    # --- Draw graph ---
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', width=2,
            font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Toy Graph with RNBRW Weights (Improved Layout)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, pos, x, edge_index, edge_weight


def triangles():
    # --- Build the custom 15-node graph ---
    edge_dict = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3, 4],
        3: [2, 4],
        4: [3, 2, 5],
        5: [4, 6, 7],
        6: [5, 7],
        7: [5, 6, 8, 9],
        8: [7, 9],
        9: [7, 8, 10],
        10: [9, 11, 12],
        11: [10, 12],
        12: [11, 10, 13, 14],
        13: [12, 14],
        14: [12, 13]
    }

    G = nx.Graph()
    for node, neighbors in edge_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # --- Compute RNBRW weights ---
    G_rnb = compute_weights(G.copy(), nsim=100, n_jobs=1)
    edges_und = list(G_rnb.edges())
    rnb_list = []
    for u, v in edges_und:
        try:
            rnb_list.append(G_rnb[u][v]['ret_n'])
        except KeyError:
            rnb_list.append(0.0)
    rnb_list = np.array(rnb_list)

    # Attach RNBRW weights to edges
    num_nodes = G.number_of_nodes()
    rnbrw_tensor = torch.zeros((num_nodes, num_nodes))
    for (u, v), weight in zip(edges_und, rnb_list):
        G[u][v]['rnbrw_weight'] = weight
        rnbrw_tensor[u, v] = weight
        rnbrw_tensor[v, u] = weight  # Undirected

    # --- Convert to PyG format ---
    x = torch.eye(num_nodes)
    edge_index = torch.tensor(edges_und, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(rnb_list, dtype=torch.float32)

    # --- Draw the graph with RNBRW weights ---
    pos = nx.spring_layout(G, seed=42)  # Optional: fixed layout for reproducibility
    edge_labels = {(u, v): f"{d['rnbrw_weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', width=2,
            font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Custom Graph with RNBRW Weights")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, pos, x, edge_index, edge_weight