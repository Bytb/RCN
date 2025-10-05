import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, self_loops=True):
        super(GCN, self).__init__()
        self.hid = 8

        self.conv1 = GCNConv(in_dim, self.hid, add_self_loops=self_loops)
        self.conv2 = GCNConv(self.hid, out_dim, add_self_loops=self_loops)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), None, x
