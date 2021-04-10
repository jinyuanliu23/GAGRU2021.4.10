import numpy as np
import torch
# from MultiGAT import GAT
from lib import utils
from gatl import GAT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


# !/usr/bin/env python
# coding: utf-8

# In[1]:


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

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
#
#
# class GAT_GRU(nn.Module):
#
#     def __init__(self, in_c, hid_c, A,n_heads):
#         super(GAT_GRU, self).__init__()
#
#         self.mtl1 = GATSubNet(in_c * 2, hid_c, in_c * 2, n_heads)
#         self.mtl2 = GATSubNet(in_c * 2, hid_c, in_c, n_heads)
#         self.A = A
#
#     def forward(self, X, h):
#         '''
#         input:
#             X: Batch x N_Nodes x InputDim
#             h: Batch x N_Nodes x InputDim
#             A: N_Nodes x N_Nodes
#         '''
#         if h is None:
#             ## init with 0
#             h = torch.zeros_like(X)
#
#         ## batchsize, n_nodes*inputdim
#         b, ni, ci = X.shape
#         gtkt = torch.cat([X, h], dim=-1)
#         gtkt = self.mtl1(gtkt, self.A)
#         gt, kt = torch.split(
#             gtkt,
#             split_size_or_sections=[ci, ci],
#             dim=-1)
#
#         gt = F.sigmoid(gt)  ## rt
#         kt = F.sigmoid(kt)  ## ut
#
#         it = torch.cat([X, gt * h], dim=-1)
#         it = self.mtl2(it, self.A)
#         it = F.tanh(it)
#
#         out = kt * it + (1 - kt) * h
#
#         return out


## test case:
# gatgru = GAT_GRU(**{
#     'in_c': 2,
#     'hid_c': 4,
#     'n_heads': 4
# })

## nfeat, nhid, nclass, dropout, alpha, nheads)
## in_features, out_features, dropout, alpha, concat=True


# In[53]:


# input_tensor = torch.rand(3, 10, 2)
# adj_mat = torch.rand(10, 10)
# out = gatgru(input_tensor, None, adj_mat)

# In[ ]:


class GAGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx,  num_nodes, nonlinearity='tanh', use_ga_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self.multi_head_nums = 3
        self._supports = []
        self._use_ga_for_ru = use_ga_for_ru
        self.adj_mx = adj_mx

        supports = []
        # if filter_type == "laplacian":
        #     supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        # elif filter_type == "random_walk":
        #     supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        # elif filter_type == "dual_random_walk":
        #     supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        #     supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        # else:
        #     supports.append(utils.calculate_scaled_laplacian(adj_mx))
        # for support in supports:
        #     self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc')
        self._gat_params = LayerParams(self, 'gat')
        self.model = GATSubNet(66, 66, 66, self.multi_head_nums)


    # def _build_sparse_matrix(L):
    #     L = L.tocoo()
    #     indices = np.column_stack((L.row, L.col))
    #     indices = indices.astype(float)  # numpy强制类型转换
    #
    #     # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
    #     indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
    #     L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
    #
    #     return L

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_ga_for_ru:
            fn = self._GAT
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size,bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        # loss = value.sum()
        # loss.backward()
        # assert False
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes , self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes , self._num_units))

        c = self._GAT(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        print('u-shape',u.shape)
        print('hx.shape',hx.shape)
        print('c.shape',c.shape)
        new_state = u * hx + (1.0 - u) * c

        return new_state


    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value



    def _GAT(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state

        x = self.model(x , self.adj_mx)
        # x = torch.tensor(x)
        weights = self._gat_params.get_weights((input_size, output_size))
        x = torch.sigmoid(torch.matmul(x, weights))  # (batch_size * self._num_nodes, output_size)
        biases = self._gat_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        # return torch.reshape(x, [batch_size, self._num_nodes * output_size])
        return  x