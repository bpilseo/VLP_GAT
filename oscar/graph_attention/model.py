import torch
import torch.nn as nn
import torch.nn.functional as F
from ..graph_attention.layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout, alpha, nheads, batch_num, node_num,):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.batch_num = batch_num
        self.node_num = node_num

        device = torch.device('cuda')

        self.adj = nn.Parameter(torch.ones(self.batch_num, self.node_num, self.node_num)).to(device)

        self.attentions = [GraphAttentionLayer(in_feat_dim, nhid, dropout=dropout, alpha=alpha, batch_num=batch_num, node_num=node_num, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, out_feat_dim, dropout=dropout, alpha=alpha, batch_num=batch_num, node_num=node_num, concat=False)

    # def forward(self, x, adj):
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)