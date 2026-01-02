import os
import json
import pandas as pd
import torch

from encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, load_triples


dataset = "SZ_TAXI"
initialize_seed(43)
device = torch.device('cuda')
config_file = 'config.json'
with open(config_file) as config:
    config = json.load(config)

sparsity = config['sparsity']
num_nodes = config["num_nodes"]
ke_dim, max_hop = config['ke_dim'], config['max_hop']
attn_num_layers, attn_num_heads = config["attn_num_layers"], config["attn_num_heads"]
key_dim, val_dim = ke_dim, ke_dim

dataset_config = config[dataset]
S_kg_prefix = dataset_config['S_kg']['kg_prefix']
S_kg_emb_prefix = dataset_config['S_kg']['emb_prefix'].format(ke_dim)
mh_score_dir = dataset_config['mh_score_dir']


def pair_encode(ent_emb, dct_triples):
    dct = {}
    for _k in dct_triples.keys():
        dct[_k] = torch.concat((ent_emb[_k[0]], ent_emb[_k[1]]))
    for _ in range(num_nodes):
        dct[(_, _)] = torch.concat((ent_emb[_], ent_emb[_]))
    return dct


def inter_encode(origin_path_feats):
    # after path feature fusion
    max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d = process_feats(origin_path_feats, ke_dim, num_nodes)  # 注意，value_mask与padding_mask的意义相反
    for att_layer in range(attn_num_layers):
        mh_feat_encoder = PathAttention(embedding_dim=ke_dim, num_heads=attn_num_heads).to(device)
        att_out = mh_feat_encoder(mh_feat, mh_padding_mask)
        mh_feat = mh_feat + att_out
    mh_score_layer = MHScoreLayer(max_path_len, ke_dim, num_nodes).to(device)
    score = mh_score_layer(mh_feat, mh_padding_mask)  # (L, 1)
    mh_score = torch.zeros((num_nodes, num_nodes))
    for i1d, (i2d, j2d) in i1d_to_ij2d.items():
        mh_score[i2d][j2d] = score[i1d]

    return mh_score


def get_dct_triples(rel_triples):
    dct_triples = {}
    for _e1, _r, _e2 in rel_triples:
        dct_triples[(int(_e1), int(_e2))] = int(_r)
    return dct_triples


if __name__ == '__main__':
    # 无多源关联感知模块
    ent_emb_path = os.path.join(S_kg_emb_prefix, 'entity2vec0.txt')
    ent_rel_ent2id_path = os.path.join(S_kg_prefix, 'ent_rel_ent2id.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    rel_triples = load_triples(ent_rel_ent2id_path).to(device)

    dct_triples = get_dct_triples(rel_triples)
    origin_path_feats = pair_encode(ent_emb, dct_triples)
    mh_score = inter_encode(origin_path_feats)
    (pd.DataFrame(mh_score.detach().cpu().numpy())
     .to_csv(os.path.join(mh_score_dir, '{}d.csv'.format(ke_dim)), header = False, index = False))