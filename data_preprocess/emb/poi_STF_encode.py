import pickle

import os
import json
import sys
sys.path.append('/media/tmuc2907/Data/hahi/DK-GNN/Bigscity-LibCity')

import pandas as pd
import torch
from encoder.att_encoder import AttEncoder
from encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer, gen_origin_path_feats
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, load2dict, \
    load2triples, triples2id, ensure_dir, get_dct_triples


dataset = "NYCTAXI20140103"
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

poi_STF_kg_prefix = dataset_config['poi_STF_kg']['kg_prefix'].format(sparsity)
poi_STF_kg_emb_prefix = dataset_config['poi_STF_kg']['kg_emb_prefix'].format(sparsity, ke_dim)
rel_weight_path = dataset_config['poi_STF_kg']['rel_weight_path'].format(sparsity)
mck_score_dir = dataset_config['mck_score_dir']


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


if __name__ == '__main__':
    # MCK_GWN
    ent_emb_path = os.path.join(poi_STF_kg_emb_prefix, 'entity2vec0.txt')
    att_emb_path = os.path.join(poi_STF_kg_emb_prefix, 'attr2vec0.txt')
    val_emb_path = os.path.join(poi_STF_kg_emb_prefix, 'val2vec0.txt')
    rel_emb_path = os.path.join(poi_STF_kg_emb_prefix, 'relation2vec0.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    att_emb = read_emb(att_emb_path).to(device)
    val_emb = read_emb(val_emb_path).to(device)
    rel_emb = read_emb(rel_emb_path).to(device)

    ent2id_path = os.path.join(poi_STF_kg_prefix, 'entity2id.txt')
    ent2id = load2dict(ent2id_path)
    rel2id_path = os.path.join(poi_STF_kg_prefix, 'relation2id.txt')
    rel2id = load2dict(rel2id_path)
    att2id_path = os.path.join(poi_STF_kg_prefix, 'attribute2id.txt')
    att2id = load2dict(att2id_path)
    val2id_path = os.path.join(poi_STF_kg_prefix, 'val2id.txt')
    val2id = load2dict(val2id_path)

    rel_triples_path = os.path.join(poi_STF_kg_prefix, 'train-rel.txt')
    rel_triples = load2triples(rel_triples_path)
    rel_triples2id = triples2id(ent2id, rel2id, ent2id, rel_triples).to(device)
    rel_dct_triples = get_dct_triples(rel_triples2id)

    attr_triples_path = os.path.join(poi_STF_kg_prefix, 'train-attr.txt')
    attr_triples = load2triples(attr_triples_path)
    attr_triples2id = triples2id(ent2id, att2id, val2id, attr_triples).to(device)

    with open(rel_weight_path, 'rb') as f:
        rel_weight = pickle.load(f)
    ent_emb = att_encode(ent_emb, att_emb, val_emb, attr_triples2id)
    origin_path_feats = rel_encode(ent_emb, rel_emb, rel_dct_triples, rel_weight)
    mck_score = inter_encode(origin_path_feats)

    ensure_dir(mck_score_dir)
    pd.DataFrame(mck_score.detach().cpu().numpy()).to_csv(os.path.join(mck_score_dir, 'poi_STF_d{}_hop{}_s{}.csv'.format(ke_dim, max_hop, sparsity)), header = False, index = False)