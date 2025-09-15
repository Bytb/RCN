import torch
import torch.nn.functional as F
import time
import sys
torch.autograd.set_detect_anomaly(True)

def contrastive_loss_node_weighted_blockwise(
    H, edge_index, edge_weight, tau=0.5, row_block_size=2048, col_block_size=16384
):
    device, dtype = H.device, H.dtype
    N = H.size(0)

    # Normalize safely (avoid NaNs if a row is all-zeros)
    Hn = F.normalize(H, p=2, dim=1, eps=1e-12)

    # RNBRW degree weights
    src, dst = edge_index
    ew = edge_weight.to(dtype)
    deg = torch.zeros(N, device=device, dtype=dtype)
    deg.scatter_add_(0, src, ew)
    deg.scatter_add_(0, dst, ew)

    # Positive neighbors (RNBRW>0), undirected neighbor lists
    pos_mask = ew > 0
    ps, pd = src[pos_mask], dst[pos_mask]
    nbrs = [[] for _ in range(N)]
    for u, v in zip(ps.tolist(), pd.tolist()):
        nbrs[u].append(v); nbrs[v].append(u)
    nbrs = [torch.tensor(n, device=device, dtype=torch.long) if n else None for n in nbrs]

    # --- helpers ---
    def lse_stream_init(size):
        m = torch.full((size,), float("-inf"), device=device, dtype=dtype)
        s = torch.zeros((size,), device=device, dtype=dtype)
        return m, s

    def lse_stream_update(m, s, x):
        """
        Streaming combine old (m,s) with a new log-sum vector x.
        Handles -inf rows safely; no grad path for all -inf rows in x.
        """
        # x can contain -inf; treat them as no contribution
        new_m = torch.maximum(m, x)
        both_neg_inf = (~torch.isfinite(m)) & (~torch.isfinite(x))

        s_scaled = s * torch.exp(torch.where(torch.isfinite(m), m - new_m, torch.zeros_like(m)))
        x_scaled = torch.exp(torch.where(torch.isfinite(x), x - new_m, torch.full_like(x, float("-inf"))))
        new_s = s_scaled + x_scaled

        new_m = torch.where(both_neg_inf, m, new_m)
        new_s = torch.where(both_neg_inf, torch.zeros_like(new_s), new_s)
        return new_m, new_s

    def lse_stream_finish(m, s):
        out = m + torch.log(torch.clamp(s, min=1e-38))
        return torch.where(s > 0, out, torch.full_like(out, float("-inf")))

    def rowwise_logsumexp_safe(logits_block):
        """
        Logsumexp over dim=1, but **skip rows with all -inf** so backward doesn’t NaN.
        Returns a vector [Br] where rows with all -inf are set to -inf without grad path.
        """
        Br = logits_block.size(0)
        # row max and mask of finite rows (at least one finite element)
        row_max, _ = torch.max(logits_block, dim=1)
        finite_rows = torch.isfinite(row_max)

        out = torch.full((Br,), float("-inf"), device=device, dtype=dtype)
        if finite_rows.any():
            lb_safe = logits_block[finite_rows]
            # standard logsumexp on safe rows
            out_safe = torch.logsumexp(lb_safe, dim=1)
            idx = finite_rows.nonzero(as_tuple=True)[0]
            out = out.index_copy(0, idx, out_safe)
        return out

    loss_accum  = torch.zeros((), device=device, dtype=dtype)
    total_weight= torch.zeros((), device=device, dtype=dtype)

    # --- main ---
    for r_start in range(0, N, row_block_size):
        r_end = min(r_start + row_block_size, N)
        rows = torch.arange(r_start, r_end, device=device, dtype=torch.long)
        H_rows = Hn[rows]
        Br = rows.numel()

        den_m, den_s = lse_stream_init(Br)
        pos_m, pos_s = lse_stream_init(Br)

        for c_start in range(0, N, col_block_size):
            c_end = min(c_start + col_block_size, N)
            cols = torch.arange(c_start, c_end, device=device, dtype=torch.long)
            H_cols = Hn[cols]

            # logits for this block
            logits_block = (H_rows @ H_cols.T) / tau  # [Br, Bc]

            # mask out self (no in-place)
            self_in_block = (rows >= c_start) & (rows < c_end)
            if self_in_block.any():
                row_idx = self_in_block.nonzero(as_tuple=True)[0]
                col_pos = (rows[self_in_block] - c_start).to(torch.long)
                mask = torch.zeros_like(logits_block, dtype=torch.bool)
                mask[row_idx, col_pos] = True
                logits_block = logits_block.masked_fill(mask, float('-inf'))

            # denominator logsumexp per row, NaN-safe
            den_block = rowwise_logsumexp_safe(logits_block)  # [Br]
            den_m, den_s = lse_stream_update(den_m, den_s, den_block)

            # numerator: per-row logsumexp over neighbors inside this slice
            pos_vals = torch.full((Br,), float("-inf"), device=device, dtype=dtype)
            rows_list = rows.tolist()
            for r_idx, i in enumerate(rows_list):
                ni = nbrs[i]
                if ni is None:
                    continue
                m = (ni >= c_start) & (ni < c_end)
                if m.any():
                    pos_cols = (ni[m] - c_start).to(torch.long)
                    # If somehow all selected cols are -inf, rowwise_logsumexp_safe will handle
                    pos_vals[r_idx] = rowwise_logsumexp_safe(logits_block[r_idx:r_idx+1, pos_cols])[0]

            pos_m, pos_s = lse_stream_update(pos_m, pos_s, pos_vals)

        # finalize
        lse_den = lse_stream_finish(den_m, den_s)
        lse_pos = lse_stream_finish(pos_m, pos_s)

        has_pos = torch.tensor(
            [nbrs[i] is not None and nbrs[i].numel() > 0 for i in rows.tolist()],
            device=device, dtype=torch.bool
        )
        valid = has_pos & torch.isfinite(lse_pos) & torch.isfinite(lse_den)

        row_loss = torch.zeros_like(lse_den)
        if valid.any():
            idx = valid.nonzero(as_tuple=True)[0]
            row_loss_valid = -(lse_pos[idx] - lse_den[idx])
            row_loss = row_loss.index_copy(0, idx, row_loss_valid)

        w = deg[rows]
        loss_accum  = loss_accum  + (row_loss * w).sum()
        total_weight= total_weight + w.sum()
    assert torch.isfinite(Hn).all(), "Hn has NaNs/Infs (check normalization)"
    # Optional logs during debug:
    print("Any finite in lse_den?", torch.isfinite(lse_den).any().item())
    print("Any finite in lse_pos?", torch.isfinite(lse_pos).any().item())
    return loss_accum / (total_weight + 1e-6)



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
            contrast = contrastive_loss_node_weighted_blockwise(embeddings, edge_index, edge_weight, tau=contrast_tau)
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