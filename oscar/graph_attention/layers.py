# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GraphAttentionLayer(nn.Module):
#     """
#     https://github.com/Diego999/pyGAT/blob/master/layers.py
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         device = torch.device('cuda')

#         # self.adj = nn.Parameter(torch.ones(10)).to(device)

#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(device)
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1))).to(device)
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         """
#         :param h: (batch_size, number_nodes, in_features)
#         :param adj: (batch_size, number_nodes, number_nodes)
#         :return: (batch_size, number_nodes, out_features)
#         """

#         # batchwise matrix multiplication
#         Wh = torch.matmul(h, self.W)  # (batch_size, number_nodes, in_features) * (in_features, out_features) -> (batch_size, number_nodes, out_features)
#         e = self.prepare_batch(Wh)  # (batch_size, number_nodes, number_nodes)

#         # (batch_size, number_nodes, number_nodes)
#         zero_vec = -9e15 * torch.ones_like(e)

#         # (batch_size, number_nodes, number_nodes)
#         # attention = torch.where(adj > 0, e, zero_vec)
#         attention = e

#         # (batch_size, number_nodes, number_nodes)
#         attention = F.softmax(attention, dim=-1)

#         # (batch_size, number_nodes, number_nodes)
#         attention = F.dropout(attention, self.dropout, training=self.training)

#         # batched matrix multiplication (batch_size, number_nodes, out_features)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def prepare_batch(self, Wh):
#         """
#         with batch training
#         :param Wh: (batch_size, number_nodes, out_features)
#         :return:
#         """
#         # Wh.shape (B, N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (B, N, 1)
#         # e.shape (B, N, N)

#         B, N, E = Wh.shape  # (B, N, N)

#         # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

#         # broadcast add (B, N, 1) + (B, 1, N)
#         e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, batch_num, node_num, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.batch_num = batch_num
        self.node_num = node_num

        device = torch.device('cuda')

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: (batch_size, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_size, number_nodes, out_features)
        """

        # batchwise matrix multiplication
        Wh = torch.matmul(h, self.W)  # (batch_size, number_nodes, in_features) * (in_features, out_features) -> (batch_size, number_nodes, out_features)
        e = self.prepare_batch(Wh)  # (batch_size, number_nodes, number_nodes)

        # (batch_size, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # (batch_size, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_size, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_size, number_nodes, number_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_size, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_size, number_nodes, out_features)
        :return:
        """
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)

        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        # broadcast add (B, N, 1) + (B, 1, N)
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'