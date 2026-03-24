import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dropout_edge
import copy
import torch

class DMoNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_clusters, dropout=0.0, lambda_orth=0.001):
        super(DMoNModel, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_clusters)  # Soft cluster assignments
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.lambda_orth = lambda_orth

    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        cluster_assignments = F.softmax(self.gcn2(h, edge_index), dim=-1)
        return cluster_assignments

    def dmon_loss(self, cluster_assignments, adj):
        # Modularity loss
        N = cluster_assignments.shape[0]
        degrees = adj.sum(dim=1)
        m = degrees.sum() / 2

        B = adj - torch.outer(degrees, degrees) / (2 * m)
        mod = torch.trace(cluster_assignments.T @ B @ cluster_assignments) / (2 * m)

        # Orthogonality regularization
        S = cluster_assignments / (cluster_assignments.sum(dim=0, keepdim=True) + 1e-9)
        I = torch.eye(self.num_clusters).to(S.device)
        orth_loss = torch.norm(S.T @ S - I)

        return -mod + self.lambda_orth * orth_loss


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 2 * hidden_dim)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BGRLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, momentum=0.99):
        super().__init__()
        self.online_encoder = GCNEncoder(in_dim, hidden_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = MLP(hidden_dim, hidden_dim)
        self.momentum = momentum

    def forward(self, x1, edge_index1, x2, edge_index2):
        # Online encoder
        z1_online = self.online_encoder(x1, edge_index1)
        z2_online = self.online_encoder(x2, edge_index2)

        # Predictor on online embeddings
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # Target encoder (no gradients)
        with torch.no_grad():
            z1_target = self.target_encoder(x1, edge_index1).detach()
            z2_target = self.target_encoder(x2, edge_index2).detach()

        return p1, p2, z1_target, z2_target

    def momentum_update(self):
        # Exponential moving average update of target encoder
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1.0 - self.momentum) * online_param.data

    def loss_fn(self, p1, p2, z1, z2):
        # Cosine similarity loss
        loss = (1 - F.cosine_similarity(p1, z2.detach(), dim=-1)).mean()
        loss += (1 - F.cosine_similarity(p2, z1.detach(), dim=-1)).mean()
        return loss / 2.0

    def embed(self, x, edge_index):
        return self.online_encoder(x, edge_index)

class SDCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_clusters, alpha=1.0):
        """
        hidden_dims: list like [500, 500, 2000, 128]
        in_dim: input feature size
        num_clusters: number of clusters
        alpha: fusion weight (how much to weigh AE vs GCN)
        """
        super().__init__()
        self.alpha = alpha
        self.num_clusters = num_clusters

        # ----- Autoencoder -----
        ae_dims = [in_dim] + hidden_dims
        self.encoder_layers = nn.ModuleList([
            nn.Linear(ae_dims[i], ae_dims[i + 1]) for i in range(len(ae_dims) - 1)
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Linear(ae_dims[i + 1], ae_dims[i]) for i in reversed(range(len(ae_dims) - 1))
        ])

        # ----- GCN -----
        self.gcn1 = GCNConv(ae_dims[-1], ae_dims[-1])  # Input from AE's second-to-last layer
        self.gcn2 = GCNConv(ae_dims[-1], ae_dims[-1])  # Output: embedding

        # ----- Cluster centers -----
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, ae_dims[-1]))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def encode(self, x):
        ae_outputs = []
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
            ae_outputs.append(x)
        return ae_outputs

    def decode(self, z):
        for layer in self.decoder_layers:
            z = F.relu(layer(z))
        return z

    def forward(self, x, edge_index):
        ae_embed = self.encode(x)[-1]  # [N, 128]
        x_hat = self.decode(ae_embed)

        gcn_input = self.alpha * ae_embed + (1 - self.alpha) * self.gcn1(ae_embed, edge_index)
        z = self.gcn2(gcn_input, edge_index)  # final embedding

        # Soft assignment Q
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)

        return ae_embed, x_hat, z, q

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def clustering_loss(self, q, p):
        return F.kl_div(q.log(), p, reduction='batchmean')

    def reconstruction_loss(self, x_hat, x_true):
        return F.mse_loss(x_hat, x_true)

class DAEGCModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters, dropout=0.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout = dropout

        # GAT Encoder
        self.gat1 = GATConv(in_channels, 2 * out_channels, heads=1, dropout=dropout)
        self.gat2 = GATConv(2 * out_channels, out_channels, heads=1, dropout=dropout)

        # Cluster centers (updated externally)
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, out_channels))
        torch.nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        z = self.gat2(x, edge_index)  # Final embeddings [N x d]
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))  # Inner product decoder

        # Compute soft cluster assignments Q using Student’s t-distribution
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)

        return z, A_pred, q

    def target_distribution(self, q):
        # Compute P from Q as in DEC/DAEGC
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def clustering_loss(self, q, p):
        # KL divergence between soft assignment q and target p
        return F.kl_div(q.log(), p, reduction='batchmean')

    def reconstruction_loss(self, A_pred, A_true):
        # A_true is dense 0-1 matrix (adjacency)
        return F.binary_cross_entropy(A_pred, A_true)


# --------- Shared GCN Encoder ---------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# --------- GRACE ---------
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class GRACEModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(GRACEModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
def get_augmented_views(x, edge_index, drop_edge_rate=0.2, drop_feat_rate=0.3):
    edge_index_1, _ = dropout_edge(edge_index, p=drop_edge_rate)
    edge_index_2, _ = dropout_edge(edge_index, p=drop_edge_rate)

    x_1 = drop_feature(x, drop_feat_rate)
    x_2 = drop_feature(x, drop_feat_rate)

    return x_1, x_2, edge_index_1, edge_index_2
