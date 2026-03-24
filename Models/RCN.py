import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class RCNLayer(nn.Module):
    """
    Cycle-Aware Graph Attention Layer with flexible bias integration for ablation studies.
    """

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, eps=1e-8,
                 attention_strategy='pre_softmax_bias', temperature=1.0, add_self_loops=False):
        super(RCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.eps = eps
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.strategy = attention_strategy
        self.temperature = temperature

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(size=(heads, out_dim)))
        self.a_dst = nn.Parameter(torch.zeros(size=(heads, out_dim)))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        if self.strategy == 'learned_gate':
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, rnbrw_weights):
        N = x.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            orig_E = rnbrw_weights.size(0)
            num_new = edge_index.size(1) - orig_E
            if num_new > 0:
                loop_weights = torch.ones(num_new, device=x.device)
                rnbrw_weights = torch.cat([rnbrw_weights, loop_weights], dim=0)

        rnbrw_weights = rnbrw_weights.detach().clone()  # avoid in-place side effects

        E = edge_index.size(1)
        h = self.W(x).view(N, self.heads, self.out_dim)

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        score = (h_src * self.a_src).sum(dim=-1) + (h_dst * self.a_dst).sum(dim=-1)

        if self.strategy == 'no_bias':
            pass

        elif self.strategy == 'pre_softmax_bias':
            bias_score = torch.log(rnbrw_weights.unsqueeze(-1) + self.eps)
            score = score + bias_score

        elif self.strategy == 'learned_gate':
            raw_bias = torch.log(rnbrw_weights.unsqueeze(-1) + self.eps)
            score = score + self.gate * raw_bias

        score = F.leaky_relu(score, negative_slope=0.2)

        alpha = torch.zeros(E, self.heads, device=x.device)
        for h_idx in range(self.heads):
            scaled_score = score[:, h_idx] / self.temperature
            alpha[:, h_idx] = self._edge_softmax(dst, scaled_score)

        if self.strategy == 'post_softmax_mult':
            normed_weights = rnbrw_weights / (rnbrw_weights.max() + self.eps)
            alpha = alpha * normed_weights.unsqueeze(-1)

        elif self.strategy == 'multi_head_specialized':
            normed_weights = rnbrw_weights / (rnbrw_weights.max() + self.eps)
            for h_idx in range(self.heads):
                if h_idx < self.heads // 2:
                    alpha[:, h_idx] = alpha[:, h_idx] * normed_weights

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        for h_idx in range(self.heads):
            out[:, h_idx, :].index_add_(0, dst, alpha[:, h_idx].unsqueeze(-1) * h_src[:, h_idx])

        out = out.view(N, self.heads * self.out_dim)

        # Optional debug logging
        if not self.training and self.strategy == 'post_softmax_mult':
            print(f"[Debug] Strategy={self.strategy}, Temp={self.temperature}, Heads={self.heads}")
            print("  RNBRW → min:", rnbrw_weights.min().item(),
                  "max:", rnbrw_weights.max().item(),
                  "mean:", rnbrw_weights.mean().item())
            print("  alpha[0:3]:", alpha[:3])

        return out, alpha

    def _edge_softmax(self, dst, scores):
        max_score = torch.zeros_like(scores).scatter_reduce_(0, dst, scores, reduce='amax', include_self=False)
        exp_scores = torch.exp(scores - max_score[dst])
        denom = torch.zeros_like(scores).scatter_add_(0, dst, exp_scores)
        return exp_scores / (denom[dst] + self.eps)

class RCNModel(nn.Module):
    """
    Wrapper for two-layer RCN with ablation support for attention strategies and temperature.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, heads=1, dropout=0.0,
                 attention_strategy='pre_softmax_bias', attention_temperature=1.0, add_self_loops=False):
        super().__init__()
        self.layer1 = RCNLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            attention_strategy=attention_strategy,
            temperature=attention_temperature,
            add_self_loops=add_self_loops
        )

        self.layer2 = RCNLayer(
            in_dim=hidden_dim * heads,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            attention_strategy=attention_strategy,
            temperature=attention_temperature,
            add_self_loops=add_self_loops
        )

        self.out_proj = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index, rnbrw_weights):
        h1, attn1 = self.layer1(x, edge_index, rnbrw_weights)
        h2, attn2 = self.layer2(h1, edge_index, rnbrw_weights)
        logits = F.log_softmax(self.out_proj(h2), dim=1)
        return logits, (attn1, attn2), h2
