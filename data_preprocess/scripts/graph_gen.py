# NYCTAXI20140103   sparsity: 740 / 39800 = 0.0186
# sparsity ->  [0.01, 0.02, 0.03]
import os
import json
import pickle
import shutil

import pandas as pd
import numpy as np
from libcity.data_preprocess.utils.utils import read_emb, initialize_seed, ensure_dir
from libcity.data_preprocess.utils.graph_gen import struct_graph_gen, function_graph_gen, pattern_graph_gen


dataset, model = "NYCTAXI20140103", "DMKG_GNN"
initialize_seed(43)
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
dataset_cfg = settings[dataset]
kg_assist_dir = settings["kg_assist_dir"]
dataset_prefix = dataset_cfg["dataset_prefix"]
kg_prefix = dataset_cfg["kg_prefix"]
num_node = dataset_cfg["num_node"]
active_nodes = dataset_cfg["active_nodes"]
edges = dataset_cfg["edges"]
poi2id = dataset_cfg["poi2id"]
cfg = settings[dataset][model]
struct_sparsity = cfg["struct_graph"]["sparsity"]
pattern_sparsity = cfg["pattern_graph"]["sparsity"]
function_sparsity = cfg["functional_graph"]["sparsity"]
period = cfg["pattern_graph"]["period"]
kg_assist_graph_dir = f"{kg_assist_dir}/{dataset}/{model}/"


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
    struct_g = struct_graph_gen(edges, struct_sparsity, num_node)
    pattern_g = pattern_graph_gen(dataset_df_inflow, dtw_inflow_path, dataset_df_outflow, dtw_outflow_path, period, num_node, pattern_sparsity, active_nodes)
    function_g = function_graph_gen(poi2id, poi_kg_list, function_sparsity, num_node)
    ensure_dir(kg_assist_graph_dir)
    pd.DataFrame(struct_g).to_csv(f"{kg_assist_graph_dir}/sg_s{struct_sparsity}.csv", index=False, header=False)
    pd.DataFrame(pattern_g).to_csv(f"{kg_assist_graph_dir}/pg_s{pattern_sparsity}.csv", index=False, header=False)
    pd.DataFrame(function_g).to_csv(f"{kg_assist_graph_dir}/fg_s{function_sparsity}.csv", index=False, header=False)


if __name__ == '__main__':
    main()