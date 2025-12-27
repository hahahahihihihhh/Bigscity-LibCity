#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from libcity.data_preprocess.utils.utils import ensure_dir, initialize_seed, toTxt

dataset = "SZ_TAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

poi_zh2en = cfg["poi_zh2en"]
poi2id = cfg["poi2id"]
road2id = cfg["road2id"]
params = cfg["params"]
sparsity = params["functional_graph"]["sparsity"]
num_node = params["num_node"]

initialize_seed(43)
poi_types = list(poi2id.keys())
id2road = {v: k for k, v in road2id.items()}


def functional_graph(kg_tuples):
    poi_mat = np.zeros((num_node, len(poi2id)))
    for tuple in kg_tuples:
        if tuple[1] in poi_types:
            road_id = road2id[str(tuple[0])]
            poi_id = poi2id[tuple[1]]
            poi_mat[road_id][poi_id] = tuple[2]
    poi_nodes = []
    for _ in range(num_node):
        if poi_mat[_].any():
            poi_nodes.append(_)
    lst = []
    for _x in poi_nodes:
        for _y in poi_nodes:
            pearson_corr, _ = pearsonr(poi_mat[_x], poi_mat[_y])
            pearson_corr2, _ = pearsonr(poi_mat[_y], poi_mat[_x])
            assert (pearson_corr == pearson_corr2)
            lst.append(((_x, _y), pearson_corr))
    lst.sort(key=lambda x: x[1], reverse=True)
    max_val, min_val = lst[0][1], max(0, lst[-1][1])
    threshold = int(num_node * num_node * sparsity)
    w_adj = np.zeros([num_node, num_node])
    w = np.zeros([num_node, num_node])
    for _ in range(min(len(lst), threshold)):
        x, y = lst[_][0]
        w_adj[x][y] = 1
        w[x][y] = (lst[_][1] - min_val) / (max_val - min_val)
        # w_adj[x][y] = lst[_][1]
    for i in range(num_node):
        for j in range(num_node):
            if w_adj[i][j] == 1:
                w_adj[j][i] = w_adj[i][j]
                w[j][i] = w[i][j]
            if i == j:
                w[i][j] = 1
    print("threshold: ", lst[min(len(lst), threshold) - 1][1])
    print("Sparsity of adj: ", len(w_adj.nonzero()[0]) / (num_node * num_node))
    fg_prefix = os.path.join(dataset, "functional_graph")
    ensure_dir(fg_prefix)
    fg_path = os.path.join(fg_prefix, "fg_s{}.csv".format(sparsity))
    pd.DataFrame(w_adj).to_csv(fg_path, index=False, header=None)
    print("The adjacent matrix of functional graph is generated!")
    weight_fg_path = os.path.join(fg_prefix, 'w_fg_s{}.csv'.format(sparsity))
    pd.DataFrame(w).to_csv(weight_fg_path, index=False, header=None)
    print("The weighted matrix of temporal graph is generated!")
    lst = []
    for i in range(num_node):
        for j in range(num_node):
            if i != j and w_adj[i][j] == 1:
                lst.append((id2road[i], 'adj_func', id2road[j]))
    tri_fg_path = os.path.join(fg_prefix, 'tri_fg_s{}.txt'.format(sparsity))
    toTxt(lst, tri_fg_path)
    print("Triples of functional graph is generated!")
    return


if __name__ == '__main__':
    kg = pd.read_csv(os.path.join(dataset, "kg.csv"))
    kg_triples = kg.values
    kg_tuples = [tuple(triple) for triple in kg_triples]
    adj_fg = functional_graph(kg_tuples)
