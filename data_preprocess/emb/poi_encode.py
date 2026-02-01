import os
import json
import pandas as pd
import torch

from encoder.att_encoder import AttEncoder
from encoder.rel_encoder import process_feats, PathAttention, MHScoreLayer
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, load2dict, load2triples, \
    triples2id, ensure_dir


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
key_dim, val_dim = ke_dim, ke_dim

poi_kg_prefix = dataset_config['poi_kg']['kg_prefix']
poi_kg_emb_prefix = dataset_config['poi_kg']['kg_emb_prefix'].format(ke_dim)
mck_score_dir = dataset_config['mck_score_dir']


def att_encode(ent_emb, att_emb, val_emb, att_triples):
    att_encoder = AttEncoder(key_dim, val_dim).to(device)
    att_ent_emb = att_encoder(att_triples, ent_emb, att_emb, val_emb)
    return att_ent_emb


def pair_encode(ent_emb, dct_triples = None):
    dct = {}
    if dct_triples is None:
        for _u in range(num_nodes):
            for _v in range(num_nodes):
                dct[(_u, _v)] = torch.concat((ent_emb[_u], ent_emb[_v]))
    else:
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


if __name__ == '__main__':
    # 无交通结点间语义关联知识
    ent_emb_path = os.path.join(poi_kg_emb_prefix, 'entity2vec0.txt')
    att_emb_path = os.path.join(poi_kg_emb_prefix, 'attr2vec0.txt')
    val_emb_path = os.path.join(poi_kg_emb_prefix, 'val2vec0.txt')

    ent_emb = read_emb(ent_emb_path).to(device)
    att_emb = read_emb(att_emb_path).to(device)
    val_emb = read_emb(val_emb_path).to(device)

    ent2id_path = os.path.join(poi_kg_prefix, 'entity2id.txt')
    ent2id = load2dict(ent2id_path)
    att2id_path = os.path.join(poi_kg_prefix, 'attribute2id.txt')
    att2id = load2dict(att2id_path)
    val2id_path = os.path.join(poi_kg_prefix, 'val2id.txt')
    val2id = load2dict(val2id_path)

    attr_triples_path = os.path.join(poi_kg_prefix, 'train-attr.txt')
    attr_triples = load2triples(attr_triples_path)
    attr_triples2id = triples2id(ent2id, att2id, val2id, attr_triples).to(device)

    ent_emb = att_encode(ent_emb, att_emb, val_emb, attr_triples2id)
    origin_path_feats = pair_encode(ent_emb)
    mck_score = inter_encode(origin_path_feats)

    ensure_dir(mck_score_dir)
    pd.DataFrame(mck_score.detach().cpu().numpy()).to_csv(os.path.join(mck_score_dir, 'poi_{}d.csv'.format(ke_dim)), header = False, index = False)