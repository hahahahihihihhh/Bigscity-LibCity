import pickle
import sys

import os
import json
import pandas as pd
import torch

from encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer, gen_origin_path_feats
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, load_triples

dataset = "TDRIVE20150406"
initialize_seed(43)
device = torch.device('cuda')
config_file = 'config.json'
with open(config_file) as config:
    config = json.load(config)
dataset_config = config[dataset]

attn_num_layers, attn_num_heads = config["attn_num_layers"], config["attn_num_heads"]
num_nodes = dataset_config["num_nodes"]
ke_dim = dataset_config['ke_dim']
sparsity = dataset_config['sparsity']
max_hop = dataset_config["max_hop"]
key_dim, val_dim = ke_dim, ke_dim

dataset_config = config[dataset]
wSTF_kg_prefix = dataset_config['STF_kg']['kg_prefix'].format(sparsity)
wSTF_kg_emb_prefix = dataset_config['STF_kg']['emb_prefix'].format(ke_dim, sparsity)
mck_score_dir = dataset_config['mck_score_dir']


def rel_encode(ent_emb, rel_emb, dct_triples, rel_weight):
    return gen_origin_path_feats(ent_emb, rel_emb, dct_triples, num_nodes, max_hop, rel_weight)


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
    # w/o POI-cor
    ent_emb_path = os.path.join(wSTF_kg_emb_prefix, 'entity2vec0.txt')
    rel_emb_path = os.path.join(wSTF_kg_emb_prefix, 'relation2vec0.txt')
    ent_rel_ent2id_path = os.path.join(wSTF_kg_prefix, 'ent_rel_ent2id.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    rel_emb = read_emb(rel_emb_path).to(device)

    rel_triples = load_triples(ent_rel_ent2id_path).to(device)
    dct_triples = get_dct_triples(rel_triples)

    rel_weight_path = os.path.join(wSTF_kg_prefix, 'rel_weight.pickle')
    with open(rel_weight_path, 'rb') as f:
        rel_weight = pickle.load(f)

    origin_path_feats = rel_encode(ent_emb, rel_emb, dct_triples, rel_weight)
    mh_score = inter_encode(origin_path_feats)

    pd.DataFrame(mh_score.detach().cpu().numpy()).to_csv(os.path.join(mh_score_dir, 'wSTF_{}d_{}hop_s{}.csv'.format(ke_dim, max_hop, sparsity)), header = False, index = False)