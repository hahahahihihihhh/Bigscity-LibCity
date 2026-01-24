import os
import json
import pandas as pd
import torch
from pystow.cli import ensure

from libcity.data_preprocess.emb.encoder.att_encoder import AttEncoder
from libcity.data_preprocess.emb.encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer, gen_origin_path_feats
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, ensure_dir

dataset, model = "NYCTAXI20140103", "KMHNet"
initialize_seed(43)
device = torch.device('cuda')
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
kg_assist_dir = settings['kg_assist_dir']
dataset_cfg = settings[dataset]
num_node = dataset_cfg["num_node"]
edges = dataset_cfg["edges"]
cfg = settings[dataset][model]
ke_dim, max_hop = cfg['ke_dim'], cfg['max_hop']
attn_num_layers, attn_num_heads = cfg["attn_num_layers"], cfg["attn_num_heads"]
key_dim, val_dim = ke_dim, ke_dim
kg_emb_prefix = cfg["kg_emb_prefix_template"].format(ke_dim)
mh_score_dir = f"{kg_assist_dir}/{dataset}/{model}"


def att_encode(ent_emb, att_emb, val_emb, att_triples):

    att_encoder = AttEncoder(key_dim, val_dim).to(device)
    att_ent_emb = att_encoder(att_triples, ent_emb, att_emb, val_emb)

    return att_ent_emb


def rel_encode(ent_emb, rel_emb, dct_triples, rel_weight):
    return gen_origin_path_feats(ent_emb, rel_emb, dct_triples, num_node, max_hop, rel_weight)


def inter_encode(origin_path_feats):
    # after path feature fusion
    max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d = process_feats(origin_path_feats, ke_dim, num_node)  # 注意，value_mask与padding_mask的意义相反
    for att_layer in range(attn_num_layers):
        mh_feat_encoder = PathAttention(embedding_dim=ke_dim, num_heads=attn_num_heads).to(device)
        att_out = mh_feat_encoder(mh_feat, mh_padding_mask)
        mh_feat = mh_feat + att_out
    mh_score_layer = MHScoreLayer(max_path_len, ke_dim, num_node).to(device)
    score = mh_score_layer(mh_feat, mh_padding_mask)  # (L, 1)
    mh_score = torch.zeros((num_node, num_node))
    for i1d, (i2d, j2d) in i1d_to_ij2d.items():
        mh_score[i2d][j2d] = score[i1d]

    return mh_score


def get_dct_triples(edges):
    dct_triples = {}
    for _e1, _e2 in edges:
        dct_triples[(int(_e1), int(_e2))] = 0
    return dct_triples


if __name__ == '__main__':
    # KMHNet
    ent_emb_path = os.path.join(kg_emb_prefix, 'entity2vec0.txt')
    att_emb_path = os.path.join(kg_emb_prefix, 'attr2vec0.txt')
    val_emb_path = os.path.join(kg_emb_prefix, 'val2vec0.txt')
    rel_emb_path = os.path.join(kg_emb_prefix, 'relation2vec0.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    att_emb = read_emb(att_emb_path).to(device)
    val_emb = read_emb(val_emb_path).to(device)
    rel_emb = read_emb(rel_emb_path).to(device)

    dct_triples = get_dct_triples(edges)

    rel_weight = {k: 1 for k in dct_triples.keys()}
    origin_path_feats = rel_encode(ent_emb, rel_emb, dct_triples, rel_weight)

    mh_score = inter_encode(origin_path_feats)
    print(mh_score)
    ensure_dir(mh_score_dir)
    pd.DataFrame(mh_score.detach().cpu().numpy()).to_csv(os.path.join(mh_score_dir, 'd{}_hop{}.csv'.format(ke_dim, max_hop)), header = False, index = False)