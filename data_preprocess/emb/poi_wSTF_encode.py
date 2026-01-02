import pickle
import sys
sys.path.append('/media/tmuc2907/Data/hahi/MCK_GWN')

import os
import json
import pandas as pd
import torch

from encoder.att_encoder import AttEncoder
from encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer, gen_origin_path_feats
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
poi_wSTF_kg_prefix = dataset_config['poi_wSTF_kg']['kg_prefix'].format(sparsity)
poi_wSTF_kg_emb_prefix = dataset_config['poi_wSTF_kg']['emb_prefix'].format(ke_dim, sparsity)
mh_score_dir = dataset_config['mh_score_dir']


def att_encode(ent_emb, att_emb, val_emb, att_triples):

    att_encoder = AttEncoder(key_dim, val_dim).to(device)
    att_ent_emb = att_encoder(att_triples, ent_emb, att_emb, val_emb)

    return att_ent_emb

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
    # MCK_GWN
    ent_emb_path = os.path.join(poi_wSTF_kg_emb_prefix, 'entity2vec0.txt')
    att_emb_path = os.path.join(poi_wSTF_kg_emb_prefix, 'attr2vec0.txt')
    val_emb_path = os.path.join(poi_wSTF_kg_emb_prefix, 'val2vec0.txt')
    rel_emb_path = os.path.join(poi_wSTF_kg_emb_prefix, 'relation2vec0.txt')
    ent_rel_ent2id_path = os.path.join(poi_wSTF_kg_prefix, 'ent_rel_ent2id.txt')
    ent_att_val2id_path = os.path.join(poi_wSTF_kg_prefix, 'ent_att_val2id.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    att_emb = read_emb(att_emb_path).to(device)
    val_emb = read_emb(val_emb_path).to(device)
    rel_emb = read_emb(rel_emb_path).to(device)
    rel_triples = load_triples(ent_rel_ent2id_path).to(device)
    att_triples = load_triples(ent_att_val2id_path).to(device)

    dct_triples = get_dct_triples(rel_triples)

    rel_weight_path = os.path.join(poi_wSTF_kg_prefix, 'rel_weight.pickle')
    with open(rel_weight_path, 'rb') as f:
        rel_weight = pickle.load(f)

    ent_emb = att_encode(ent_emb, att_emb, val_emb, att_triples)

    origin_path_feats = rel_encode(ent_emb, rel_emb, dct_triples, rel_weight)

    mh_score = inter_encode(origin_path_feats)

    pd.DataFrame(mh_score.detach().cpu().numpy()).to_csv(os.path.join(mh_score_dir, 'poi_wSTF_{}d_{}hop_s{}.csv'.format(ke_dim, max_hop, sparsity)), header = False, index = False)