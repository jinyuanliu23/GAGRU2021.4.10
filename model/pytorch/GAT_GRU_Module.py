#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[25]:


from gat import GATSubNet


# In[ ]:


# class AttGat(nn.Module):
#     def __init__(self, in_feats, n_feats, use_bias):

#         super(AttGat, self).__init__()
#         self.k_linear = nn.Linear(in_feats, n_feats, bias=use_bias)
#         self.q_linear = nn.Linear(in_feats, n_feats, bias=use_bias)
#         self.v_linear = nn.Linear(in_feats, n_feats, bias=use_bias)

#         self.act = nn.ReLU() ## TODO

#     def _mat_trans(self, A):
#         '''
#         if something goes wrong, refer to 
#         https://github.com/LeronQ/GCN_predict-Pytorch/blob/main/gcnnet.py#L41
#         '''
#         ## TODO: Souis: Maybe add torch.no_grad() here.
#         n_nodes = A.size(0)
#         ## this turns non-zero elements to 1.
#         A_ones = A.bool().float()
#         matrix_i = torch.eye(n_nodes, dtype=torch.float, device=A.device)
#         A = A_ones + matrix_i

#         degree_matrix = torch.sum(A, dim=1, keepdim=False)
#         degree_matrix = degree_matrix.pow(-1)
#         degree_matrix[degree_matrix == float("inf")] = 0. 

#         degree_matrix = torch.diag(degree_matrix)

#         return torch.mm(degree_matrix, A)

#     def forward(self, X, A):
#         A = self._mat_trans(A)

#         print(X.shape)
#         k = self.k_linear(X)
#         print('K', k.shape, 'X', X.shape, "A", A.shape)
#         k = self.act(torch.matmul(A, k))

#         q = self.q_linear(X)
#         q = self.act(torch.matmul(A, q))
        
#         v = self.v_linear(X)
#         v = self.act(torch.matmul(A, v))

#         r = torch.matmul(q, k.T) / torch.sqrt(X.shape[0])
#         r = r * v
#         return r

# class MtlGat(nn.Module):
#     def __init__(self, n_heads, dropout, in_feats, n_feats, use_bias):

#         super(MtlGat, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.heads = nn.ModuleList([AttGat(in_feats, n_feats, use_bias) for _ in range(n_heads)])

#     def forward(self, X, A):
#         X = self.dropout(X)
#         X = torch.cat([att(X, A) for att in self.heads], dim=1)
#         return F.log_softmax(X, dim=1)


# In[52]:


class GAT_GRU(nn.Module):

    def __init__(self, in_c, hid_c, n_heads):

        super(GAT_GRU, self).__init__()

        self.mtl1 = GATSubNet(in_c * 2, hid_c, in_c * 2, n_heads)
        self.mtl2 = GATSubNet(in_c * 2, hid_c, in_c, n_heads)

    def forward(self, X, h, A):
        '''
        input:
            X: Batch x N_Nodes x InputDim
            h: Batch x N_Nodes x InputDim
            A: N_Nodes x N_Nodes
        '''
        if h is None:
            ## init with 0
            h = torch.zeros_like(X)

        ## batchsize, n_nodes*inputdim
        b, ni, ci = X.shape
        gtkt = torch.cat([X, h], dim=-1)
        gtkt = self.mtl1(gtkt, A)
        gt, kt = torch.split(
            gtkt,
            split_size_or_sections=[ci, ci],
            dim=-1)

        gt = F.sigmoid(gt) ## rt
        kt = F.sigmoid(kt) ## ut

        it = torch.cat([X, gt * h], dim=-1)
        it = self.mtl2(it, A)
        it = F.tanh(it)

        out = kt * it + (1 - kt) * h

        return out

## test case:
gatgru = GAT_GRU(**{
    'in_c': 2,
    'hid_c': 4,
    'n_heads': 4
})

## nfeat, nhid, nclass, dropout, alpha, nheads)
## in_features, out_features, dropout, alpha, concat=True


# In[53]:


input_tensor = torch.rand(3, 10, 2)
adj_mat = torch.rand(10, 10)
out = gatgru(input_tensor, None, adj_mat)


# In[ ]:




