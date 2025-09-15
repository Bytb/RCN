import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, self_loops=True):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1


        self.conv1 = GATConv(in_dim, self.hid, heads=self.in_head, dropout=0.6, add_self_loops=self_loops)
        self.conv2 = GATConv(self.hid*self.in_head, out_dim, concat=False,
                             heads=self.out_head, dropout=0.6, add_self_loops=self_loops)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)

        return F.log_softmax(x, dim=1), attn_weights, x
