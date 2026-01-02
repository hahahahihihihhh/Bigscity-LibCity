import sys
sys.path.append('/media/tmuc2907/Data/hahi/MCK_GWN')

import math
import torch
import torch.nn.functional as F

from torch import nn
from libcity.data_preprocess.utils.utils import initialize_seed

initialize_seed(43)
device = torch.device('cuda')

class PathAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(PathAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, feat, padding_mask):
        """
        Args:
            feat: (L, N, E) where L = S is the target sequence length, N is the batch size, E is the embedding dimension
            padding_mask: (N, S)` where N is the batch size, S is the source sequence length.
        """
        attn_output, attn_output_weights = self.attention(feat, feat, feat, key_padding_mask=padding_mask)
        # feat = feat + attn_output
        return attn_output

class LinearBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(c_in, c_out, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class MHScoreLayer(nn.Module):
    def __init__(self, max_path_len, input_dim, node_num, pool_layer=nn.MaxPool1d):
        super(MHScoreLayer, self).__init__()
        self.max_path_len = max_path_len
        self.input_dim = input_dim
        self.node_num = node_num
        self.hop_max_pooling = pool_layer(kernel_size=self.max_path_len)
        self.fcb = nn.Sequential(
            LinearBlock(self.input_dim, self.input_dim // 2),
            LinearBlock(self.input_dim // 2, self.input_dim // 2)
        )
        self.fc = nn.Linear(self.input_dim // 2, 1)

    def forward(self, feat, padding_mask, fusion_type='maxpooling'):
        """

        Args:
            feat: [L, B, D]
            padding_mask: [B, L]
            fusion_type:
        """
        feat = feat.permute(1, 2, 0)  # (max_path_len, L, ke_dim) -> (L, ke_dim, max_path_len)
        # (L, max_path) -> (L, max_path, 1) -> (L, max_path, ke_dim) -> (L, ke_dim, max_path)
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, self.input_dim).permute(0, 2, 1)
        # valued_mask = 1 - padding_mask
        # weight_mask = feat.sum(1).unsequence(1).repeat(1, self.input_dim, 1)  # B, D, L
        feat = feat.masked_fill(padding_mask, float('-inf'))        # 掩码

        if fusion_type == 'maxpooling':
            feat = self.hop_max_pooling(feat).squeeze(-1)   # (L, ke_dim, max_path) -> (L, ke_dim, 1) -> (L, ke_dim)
        # elif fusion_type == 'attention':
        #     # 将 attention 的输出，加权，
        #     feat = feat * weight_mask

        feat = self.fcb(feat)       # (L, ke_dim) -> (L, ke_dim // 2)
        score = self.fc(feat)       # (L, ke_dim // 2) -> (L, 1)
        return score

def gen_origin_path_feats(ent_emb, rel_emb, dct_triples, num_nodes, max_hop, rel_weight):
    n_ent = ent_emb.shape[0]
    dct = {}
    adj_lst = [[] for _ in range(n_ent)]
    dis = [[math.inf for _ in range(num_nodes)] for _ in range(num_nodes)]
    st = set()
    for _s, _t in dct_triples.keys():
        st.add((_s, _t))
        assert (_s != _t)
        adj_lst[_s].append(_t)
    assert (len(st) == len(dct_triples.keys()))
    # 无自环重边
    if (max_hop >= 0):
        for _x in range(num_nodes):
            path_emb = ent_emb[_x]
            dct.setdefault((_x, _x), []).append(path_emb)
            dis[_x][_x] = 0
    if (max_hop >= 1):
        for _x in range(num_nodes):
            for _y in adj_lst[_x]:
                if (_y < num_nodes and (dis[_x][_y] == math.inf or dis[_x][_y] == 1)):
                    path_emb = torch.concat((ent_emb[_x], rel_weight[(_x, _y)] * rel_emb[dct_triples[(_x, _y)]], ent_emb[_y]), dim=0)
                    dct.setdefault((_x, _y), []).append(path_emb)
                    dis[_x][_y] = 1
    if (max_hop >= 2):
        for _x in range(num_nodes):
            for _y in adj_lst[_x]:
                for _z in adj_lst[_y]:
                    if (_z < num_nodes and (dis[_x][_z] == math.inf or dis[_x][_z] == 2)):
                        path_emb = torch.concat((ent_emb[_x], rel_weight[(_x, _y)] * rel_emb[dct_triples[(_x, _y)]], ent_emb[_y], rel_weight[(_y, _z)] * rel_emb[dct_triples[(_y, _z)]], ent_emb[_z]), dim=0)
                        dct.setdefault((_x, _z), []).append(path_emb)
                        dis[_x][_z] = 2
    if (max_hop >= 3):
        for _x in range(num_nodes):
            for _y in adj_lst[_x]:
                for _z in adj_lst[_y]:
                    for _a in adj_lst[_z]:
                        if (_a < num_nodes and (dis[_x][_a] == math.inf or dis[_x][_a] == 3)):
                            path_emb = torch.concat((ent_emb[_x], rel_weight[(_x, _y)] * rel_emb[dct_triples[(_x, _y)]], ent_emb[_y], rel_weight[(_y, _z)] * rel_emb[dct_triples[(_y, _z)]], ent_emb[_z], rel_weight[(_z, _a)] * rel_emb[dct_triples[(_z, _a)]], ent_emb[_a]), dim=0)
                            dct.setdefault((_x, _a), []).append(path_emb)
                            dis[_x][_a] = 3
    dct = dict(sorted(dct.items()))
    for _k, _path_emb_list in dct.items():
        _path_emb = torch.stack(_path_emb_list).transpose(0, 1).unsqueeze(0)
        _path_emb = F.max_pool1d(_path_emb, kernel_size=len(_path_emb_list)).flatten()
        dct[_k] = _path_emb
        # print(_k, len(dct[_k]))
    return dct

def process_feats(origin_path_feats, ke_dim, num_nodes):
    """
    returns:
        mh_feat: (L, NxN, D)
        mh_value_mask: (N, N) -> 1 表示有意义的位置
        mh_padding_mask: (NxN, L) -> 1 表示被 padding 的位置
        max_path_len: int
    """
    max_path_len = 0
    for _k, _v in origin_path_feats.items():
        max_path_len = max(max_path_len, len(_v) // ke_dim)
    padding_feat, padding_mask, i1d_to_ij2d, value_mask = [], [], {}, torch.zeros((num_nodes, num_nodes))
    cur_i1d = 0
    for _h, _t in origin_path_feats.keys():
        zero_feat = torch.zeros((1, max_path_len, ke_dim))  # (1, max_path_len, ke_dim)
        one_mask = torch.ones(1, max_path_len)  # (1, max_path_len)
        value_feat = origin_path_feats[(_h, _t)].view(-1, ke_dim)  # (path_len, ke_dim)
        zero_feat[:, :value_feat.shape[0]] = value_feat  # (1, max_path_len, ke_dim) 填充路径，其余位置用0填充
        one_mask[:, :value_feat.shape[0]] = False  # (1, max_path_len) 填充路径，其余位置用1填充
        padding_feat.append(zero_feat)  # L * (1, max_path_len, ke_dim)
        padding_mask.append(one_mask)  # L * (1, max_path_len)
        value_mask[_h][_t] = True  # 将有路径的结点对置为 1 (num_nodes, num_nodes)
        i1d_to_ij2d[cur_i1d] = [_h, _t]  # 字典存储有路径的结点对
        cur_i1d += 1
    mh_feat = torch.cat(padding_feat, dim=0).permute(1, 0, 2).to(device)     # (L, max_path_len, ke_dim) -> (max_path_len, L, ke_dim)
    mh_padding_mask = torch.cat(padding_mask, dim=0).to(device).bool()             # (L, max_path_len)
    mh_value_mask = value_mask.to(device).bool()                                   # (num_nodes, num_nodes)
    return max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d
