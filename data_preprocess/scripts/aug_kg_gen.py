# NYCTAXI20140103   sparsity: 740 / 39800 = 0.0186
# sparsity ->  [0.01, 0.02, 0.03]
import json
import pickle
import shutil

import pandas as pd
import numpy as np
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, ensure_dir
from libcity.data_preprocess.utils.graph_gen import struct_graph_gen, function_graph_gen, pattern_graph_gen


dataset, model = "TDRIVE20150406", "DMKG_GNN"
initialize_seed(43)
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
dataset_cfg = settings[dataset]
dataset_prefix = dataset_cfg["dataset_prefix"]
kg_prefix = dataset_cfg["kg_prefix"]
num_node = dataset_cfg["num_node"]
active_nodes = dataset_cfg["active_nodes"]
edges = dataset_cfg["edges"]
poi2id = dataset_cfg["poi2id"]
cfg = settings[dataset][model]
sparsity = cfg["sparsity"]
period = cfg["pattern_graph"]["period"]


def g_merge(sem_g, g, type):
    for i in range(num_node):
        for j in range(num_node):
            if i == j:  continue
            if g[i][j] > sem_g[i][j][1]:
                sem_g[i][j] = (type, g[i][j])


def main():
    poi_kg_df = pd.read_csv(f"{dataset_prefix}/poi_kg.csv")
    dataset_df_inflow = np.expand_dims(pd.read_csv(f"{dataset_prefix}/{dataset}_inflow.csv").values, axis=2)
    dtw_inflow_path = f"{dataset_prefix}/dtw_{dataset}_inflow.npy"
    dataset_df_outflow = np.expand_dims(pd.read_csv(f"{dataset_prefix}/{dataset}_outflow.csv").values, axis=2)
    dtw_outflow_path = f"{dataset_prefix}/dtw_{dataset}_outflow.npy"
    poi_kg_list = [
        (int(g), int(poi2id[t]), int(n))
        for g, n, t in poi_kg_df[["grid_id", "num", "poi_type"]].itertuples(index=False, name=None)
    ]
    struct_g = struct_graph_gen(edges, sparsity, num_node)
    pattern_g = pattern_graph_gen(dataset_df_inflow, dtw_inflow_path, dataset_df_outflow, dtw_outflow_path, period, num_node, sparsity, active_nodes)
    function_g = function_graph_gen(poi2id, poi_kg_list, sparsity, num_node)
    sem_g = [[(None, 0) for _ in range(num_node)] for _ in range(num_node)]
    g_merge(sem_g, struct_g, type="struct_adj")
    g_merge(sem_g, pattern_g, type="pattern_adj")
    g_merge(sem_g, function_g, type="function_adj")
    w_dict = {}
    sem_kg = []
    for i in range(num_node):
        for j in range(num_node):
            if sem_g[i][j][0]:
                w_dict[(i, j)] = sem_g[i][j][1]
                sem_kg.append([str(i), sem_g[i][j][0], str(j)])
    print("sparsity: {}".format(sparsity))
    print(f"semantic graph sparsity: {len(sem_kg) / (num_node * (num_node - 1))}")
    # make aug_kg
    sem_kg_df = pd.DataFrame(sem_kg, columns=["origin", "rel", "destination"])
    aug_kg_dir = f"{kg_prefix}/aug_kg_s{sparsity}"
    ensure_dir(aug_kg_dir)
    sem_kg_df.to_csv(f"{aug_kg_dir}/sem_kg_s{sparsity}.csv", index=False)
    with open(f"{aug_kg_dir}/rel_weight.pickle", "wb") as f:
        pickle.dump(w_dict, f)
    shutil.copy2(f"{dataset_prefix}/poi_kg.csv", f"{aug_kg_dir}/poi_kg.csv")


if __name__ == '__main__':
    main()