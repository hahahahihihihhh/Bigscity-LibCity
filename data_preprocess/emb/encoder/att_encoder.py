import sys
sys.path.append('/media/tmuc2907/Data/hahi/MCK_GWN')

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.data_preprocess.utils.utils import initialize_seed

initialize_seed(43)
device = torch.device('cuda')

class AttEncoder(nn.Module):
    def __init__(self, key_dim, val_dim):
        super(AttEncoder, self).__init__()
        self.a = nn.Linear(key_dim * 2, 1)
        nn.init.xavier_uniform_(self.a.weight)
        self.W = nn.Parameter(torch.zeros(key_dim + val_dim, key_dim))
        nn.init.xavier_uniform_(self.W)
        # self.W = nn.Parameter(torch.randn(key_dim + val_dim, key_dim))  # 标准正态分布
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  # For attention scores

    def forward(self, attribute_triples, ent_feats, att_feats, val_feats):
        ## fixme: consider not use norisy attribute if all the attribute are noisy
        ## fixme: consider share the attribute importance to all nodes
        N = ent_feats.shape[0]
        E = attribute_triples.shape[0]
        device = ent_feats.device
        h, att, val = attribute_triples.transpose(0, 1)  # shape=[E]
        attention_score = self.a(torch.cat((ent_feats[h], att_feats[att]), dim=-1))

        attention_score = attention_score.squeeze(-1)  # shape = [E,]
        attention_score = torch.exp(self.leaky_relu(attention_score))

        edges = torch.stack((h, torch.arange(E, device=device)), dim=0)

        incidence_matrix = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1], device=device),
                                                    size=(N, E))  # shape = [N, E]
        row_sum = torch.sparse.mm(incidence_matrix, attention_score.reshape(-1, 1)).squeeze(-1)  # shape = [N,]

        attention_p = attention_score / row_sum[h]  # shape = [E]
        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [E, dim1 + dim2]

        att_vals = att_vals @ self.W  # shape = [E, dim]
        # att_vals = self.W(att_vals)
        att_vals = att_vals * attention_p.reshape(-1, 1)  # shape = [E, dim]
        to_feats = torch.sparse.mm(incidence_matrix, att_vals)  # shape = [N, dim]
        to_feats = to_feats + ent_feats
        to_feats = F.elu(to_feats)
        return to_feats
