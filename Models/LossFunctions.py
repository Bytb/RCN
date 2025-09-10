import torch
import torch.nn.functional as F
import time
import sys

def modularity_loss(Q, edge_index, edge_weight):
    """
    Modularity loss (negative modularity for minimization).
    Q: N x C matrix after softmax (soft community assignments)
    edge_index: 2 x E edge index tensor
    edge_weight: E RNBRW weights
    """
    N, C = Q.shape
    device = Q.device

    # 🔁 Normalize edge weights first
    edge_weight = edge_weight + 1e-8  # prevent zero weights
    #edge_weight = edge_weight / edge_weight.sum() * edge_weight.numel()

    A = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N), device=device).to_dense()
    degrees = A.sum(dim=1, keepdim=True)
    m = edge_weight.sum()

    expected = degrees @ degrees.T / (2 * m)
    B = A - expected

    trace_val = torch.trace(Q.T @ B @ Q)
    mod = trace_val / (2 * m)

    return -mod  # negative for minimization


def laplacian_loss(H, edge_index, edge_weight):
    """
    Laplacian smoothness loss.
    H: node embeddings (N x D)
    """
    src, dst = edge_index
    diffs = H[src] - H[dst]
    loss = (edge_weight * (diffs ** 2).sum(dim=1)).sum()
    return loss

def contrastive_loss_node_weighted(H, edge_index, edge_weight, tau=0.5):
    N = H.size(0)
    device = H.device

    # Normalize embeddings
    Hn = F.normalize(H, p=2, dim=1)
    sim_matrix = F.cosine_similarity(Hn.unsqueeze(1), Hn.unsqueeze(0), dim=-1) / tau
    exp_sim = torch.exp(sim_matrix)

    # RNBRW degree per node
    rnbrw_degree = torch.zeros(N, device=device)
    src, dst = edge_index
    rnbrw_degree.scatter_add_(0, src, edge_weight)
    rnbrw_degree.scatter_add_(0, dst, edge_weight)

    # Build positive mask: edge (i,j) exists with RNBRW > 0
    # pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    # for i, j, w in zip(src, dst, edge_weight):
    #     if w > 0:
    #         pos_mask[i, j] = True
    #         pos_mask[j, i] = True  # undirected
    mask = edge_weight > 0
    i = src[mask]
    j = dst[mask]

    pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    pos_mask[i, j] = True
    pos_mask[j, i] = True  # undirected

    # Negative = all except self
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = ~eye

    numerator = (exp_sim * pos_mask).sum(dim=1)
    denominator = (exp_sim * neg_mask).sum(dim=1)

    valid = (numerator > 0) & (denominator > 0)
    loss_vec = torch.zeros(N, device=device)
    loss_vec[valid] = -torch.log(numerator[valid] / denominator[valid])

    # Weight loss by RNBRW degree
    weighted_loss = loss_vec * rnbrw_degree
    return weighted_loss.sum() / (rnbrw_degree.sum() + 1e-6)

def contrastive_loss_edge_scaled(H, edge_index, edge_weight, tau=0.5):
    N = H.size(0)
    device = H.device

    Hn = F.normalize(H, p=2, dim=1)
    sim_matrix = F.cosine_similarity(Hn.unsqueeze(1), Hn.unsqueeze(0), dim=-1) / tau

    # Build edge weight matrix W (symmetric)
    W = torch.zeros((N, N), device=device)
    for i, j, w in zip(edge_index[0], edge_index[1], edge_weight):
        W[i, j] = w
        W[j, i] = w  # undirected

    scaled_sim = torch.exp(sim_matrix * W)

    pos_mask = W > 0
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = (~eye) & (~pos_mask)

    numerator = (scaled_sim * pos_mask).sum(dim=1)
    denominator = (scaled_sim * neg_mask).sum(dim=1)

    valid = (numerator > 0) & (denominator > 0)
    loss_vec = torch.zeros(N, device=device)
    loss_vec[valid] = -torch.log(numerator[valid] / denominator[valid])

    return loss_vec[valid].mean() if valid.any() else torch.tensor(0.0, device=device)

def contrastive_loss(H, edge_index, edge_weight, tau=0.5,
                                   epsilon=1e-6, leaf_push=True, alpha=1.0):
    N, device = H.size(0), H.device
    Hn = F.normalize(H, p=2, dim=1)
    sim = F.cosine_similarity(Hn.unsqueeze(1), Hn.unsqueeze(0), dim=-1) / tau
    exp_sim = torch.exp(sim)

    # RNBRW degree and leaf mask
    rnbrw_degree = torch.zeros(N, device=device)
    src, dst = edge_index
    rnbrw_degree.scatter_add_(0, src, edge_weight)
    rnbrw_degree.scatter_add_(0, dst, edge_weight)
    leaf_mask = (rnbrw_degree == 0)
    eye = torch.eye(N, dtype=torch.bool, device=device)

    # Positives: RNBRW>0 edges (undirected)
    pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    for i, j, w in zip(src, dst, edge_weight):
        if w > 0:
            pos_mask[i, j] = True
            pos_mask[j, i] = True
    pos_mask.fill_diagonal_(False)

    # Negatives: everything except self and positives (strict negatives)
    neg_mask = (~eye) & (~pos_mask)

    # ---- Leaf handling: repel leaves from other leaves ----
    if leaf_push:
        leaf_idx = torch.nonzero(leaf_mask, as_tuple=False).flatten()
        for i in leaf_idx.tolist():
            # no positives for leaves
            pos_mask[i, :] = False
            # all other leaves are negatives
            others = leaf_idx[leaf_idx != i]
            neg_mask[i, others] = True

    numerator = (exp_sim * pos_mask).sum(dim=1)
    denominator = (exp_sim * neg_mask).sum(dim=1)

    # Valid rows: must have a denominator; add epsilon in numerator to avoid -inf
    valid = (denominator > 0)

    loss_vec = torch.zeros(N, device=device)
    loss_vec[valid] = -torch.log((numerator[valid] + epsilon) / denominator[valid])

    # Weight: normal RNBRW degree + alpha for leaves so they contribute
    weights = rnbrw_degree + alpha * leaf_mask.float()
    return (loss_vec * weights).sum() / (weights.sum() + 1e-6)

### NON - RNBRW versions
def modularity_loss_nornbrw(Q, edge_index):
    """
    Modularity loss (unweighted graph).
    Q: N x C matrix after softmax (soft community assignments)
    edge_index: 2 x E edge index tensor
    """
    N, C = Q.shape
    device = Q.device

    edge_weight = torch.ones(edge_index.size(1), device=device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N), device=device).to_dense()
    degrees = A.sum(dim=1, keepdim=True)
    m = edge_weight.sum()

    expected = degrees @ degrees.T / (2 * m)
    B = A - expected

    trace_val = torch.trace(Q.T @ B @ Q)
    mod = trace_val / (2 * m)

    return -mod  # negative for minimization

def laplacian_loss_nornbrw(H, edge_index):
    """
    Laplacian loss without edge weights (assumes weight=1).
    """
    src, dst = edge_index
    diffs = H[src] - H[dst]
    loss = (diffs ** 2).sum(dim=1).sum()
    return loss

def contrastive_loss_nornbrw(H, topk_mask, tau=0.5):
    """
    Contrastive loss using a precomputed top-k mask.

    H: [N, D] node embeddings
    topk_mask: [N, N] boolean mask for positive neighbors
    tau: temperature for scaling
    """
    N, D = H.size()
    device = H.device

    sim_matrix = F.cosine_similarity(H.unsqueeze(1), H.unsqueeze(0), dim=-1) / tau
    exp_sim = torch.exp(sim_matrix)

    # Ensure diagonal is not included in numerator/denominator
    pos_mask = topk_mask.clone()
    pos_mask.fill_diagonal_(False)
    neg_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    numerator = (exp_sim * pos_mask).sum(dim=1)
    denominator = (exp_sim * neg_mask).sum(dim=1)

    valid = (numerator > 0) & (denominator > 0)
    loss_vec = torch.zeros(N, device=device)
    loss_vec[valid] = -torch.log(numerator[valid] / denominator[valid])

    return loss_vec[valid].mean() if valid.any() else torch.tensor(0.0, device=device)

def combined_community_loss_nornbrw(
    embeddings,
    edge_index,
    topk_mask,
    lambda_mod=0.0,
    lambda_lap=0.0,
    lambda_contrast=0.0,
    contrast_tau=0.5
):
    Q = F.softmax(embeddings, dim=1)
    mod_loss = modularity_loss_nornbrw(Q, edge_index) if lambda_mod > 0 else torch.tensor(0.0, device=embeddings.device)
    lap_loss = laplacian_loss_nornbrw(embeddings, edge_index) if lambda_lap > 0 else torch.tensor(0.0, device=embeddings.device)
    contrast = contrastive_loss_nornbrw(embeddings, topk_mask, tau=contrast_tau) if lambda_contrast > 0 else torch.tensor(0.0, device=embeddings.device)

    return lambda_mod * mod_loss + lambda_lap * lap_loss + lambda_contrast * contrast

def combined_dmon_rcn_loss_wL2(z, edge_index, edge_weight, q_soft, topk_mask, lambda_mod=1.0, lambda_contrast=0.001, lambda_orth=0.1, lambda_lap = 0.01, temperature=0.5, rnbrw_edge_weight=None):
    """
    z: node embeddings [N, d]
    edge_index: graph structure
    edge_weight: edge weights (e.g., degree or RNBRW)
    q_soft: soft assignments [N, K]
    rnbrw_edge_weight: optional RNBRW weights for modularity and contrastive loss
    """
    # --- Modularity Loss ---
    mod_loss = modularity_loss(q_soft, edge_index, edge_weight)

    # --- Contrastive Loss ---
    contrast_loss = contrastive_loss(z, topk_mask=topk_mask, edge_index=edge_index, edge_weight=edge_weight, tau=temperature)

    # --- DMoN Orthogonality Loss ---
    S = q_soft / (q_soft.sum(dim=0, keepdim=True) + 1e-9)  # [N, K]
    I = torch.eye(S.size(1), device=S.device)
    orth_loss = torch.norm(S.T @ S - I)

    #L2
    lap_loss = laplacian_loss(z, edge_index, edge_weight) if lambda_lap > 0 else torch.tensor(0.0, device=z.device)

    # --- Total Loss ---
    total_loss = (
        lambda_mod * mod_loss +
        lambda_contrast * contrast_loss +
        lambda_orth * orth_loss +
        lambda_lap * lap_loss
    )

    return total_loss

def combined_community_loss(
    embeddings,
    edge_index,
    edge_weight,
    lambda_mod=0.0,
    lambda_lap=0.0,
    lambda_contrast=0.0,
    lambda_orth=0.0,
    contrast_tau=0.5,
    contrast_variant='node'  # 'node', 'edge', or 'both'
):
    # --- Ortho --- #
    # --- DMoN Orthogonality Loss --- #
    Q = F.softmax(embeddings, dim=1)
    S = Q / (Q.sum(dim=0, keepdim=True) + 1e-9)  # [N, K]
    I = torch.eye(S.size(1), device=S.device)
    orth_loss = torch.norm(S.T @ S - I)

    mod_loss = modularity_loss(Q, edge_index, edge_weight) if lambda_mod > 0 else torch.tensor(0.0, device=embeddings.device)
    lap_loss = laplacian_loss(embeddings, edge_index, edge_weight) if lambda_lap > 0 else torch.tensor(0.0, device=embeddings.device)

    # Choose contrastive variant
    if lambda_contrast > 0:
        if contrast_variant == 'node':
            contrast = contrastive_loss_node_weighted(embeddings, edge_index, edge_weight, tau=contrast_tau)
        elif contrast_variant == 'edge':
            contrast = contrastive_loss_edge_scaled(embeddings, edge_index, edge_weight, tau=contrast_tau)
        else:
            raise ValueError(f"Unknown contrast_variant: {contrast_variant}")
    else:
        contrast = torch.tensor(0.0, device=embeddings.device)

    return lambda_mod * mod_loss + lambda_lap * lap_loss + lambda_contrast * contrast + lambda_orth * orth_loss

def combined_community_loss_I(
    embeddings,
    edge_index,
    edge_weight,
    topk_mask,
    lambda_mod=0.0,
    lambda_lap=0.0,
    lambda_contrast=0.0,
    lambda_orth=0.0,
    contrast_tau=0.5
):
    """
    Combined community loss:
    lambda_mod * modularity_loss +
    lambda_lap * laplacian_loss +
    lambda_contrast * contrastive_loss
    """
    # --- Ortho --- #
    # --- DMoN Orthogonality Loss --- #
    Q = F.softmax(embeddings, dim=1)
    S = Q / (Q.sum(dim=0, keepdim=True) + 1e-9)  # [N, K]
    I = torch.eye(S.size(1), device=S.device)
    orth_loss = torch.norm(S.T @ S - I)

    t0 = time.perf_counter()
    mod_loss = modularity_loss(Q, edge_index, edge_weight) if lambda_mod > 0 else torch.tensor(0.0, device=embeddings.device)
    t1 = time.perf_counter()
    lap_loss = laplacian_loss(embeddings, edge_index, edge_weight) if lambda_lap > 0 else torch.tensor(0.0, device=embeddings.device)
    t2 = time.perf_counter()
    contrast_loss = contrastive_loss(embeddings, edge_index=edge_index, edge_weight=edge_weight, tau=contrast_tau)
    t3 = time.perf_counter()

    #print(f"  [Loss Timing] Mod: {t1-t0:.3f}s | Lap: {t2-t1:.3f}s | Contrast: {t3-t2:.3f}s")
    sys.stdout.flush()
    total = lambda_mod * mod_loss + lambda_lap * lap_loss + lambda_contrast * contrast_loss + lambda_orth * orth_loss
    return total