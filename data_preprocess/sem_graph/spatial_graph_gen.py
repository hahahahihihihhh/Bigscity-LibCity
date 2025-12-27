# import json
# import os
# import numpy as np
# import pandas as pd
# from libcity.data_preprocess.utils.utils import load_dict, toTxt, initialize_seed
#
#
# dataset = "SZ_TAXI"
# with open("setting.json", "r", encoding="utf-8") as f:
#     settings = json.load(f)
# cfg = settings[dataset]
#
# params = cfg["params"]
# road2id = cfg["road2id"]
#
# dataset_path = params["dataset_path"]
# num_node = params["num_node"]
# dataset_rel_path = params["dataset_rel_path"]
# link_cap = params["spatial_graph"]["link_cap"]
# decay_ratio = params["spatial_graph"]["decay_ratio"]
# sparsity = params["temporal_graph"]["sparsity"]
# INF = 0x3f3f3f3f
# initialize_seed(43)
#
#
# def getEdge(dataset_rel_df):
#     edge = [[0 for _ in range(num_node)] for _ in range(num_node)]
#     for _row in dataset_rel_df.values:
#         ori, dest, link = _row[2], _row[3], _row[4]
#         if link:
#             edge[road2id[str(ori)]][road2id[str(dest)]] = 1
#     return edge
#
#
# def floyed(edge):
#     dis = edge.copy()
#     for _i in range(num_node):
#         for _j in range(num_node):
#             if _i != _j and dis[_i][_j] == 0:
#                 dis[_i][_j] = INF
#     for _i in range(num_node):
#         for _j in range(num_node):
#             for _k in range(num_node):
#                 if dis[_i][_k] + dis[_k][_j] < dis[_i][_j]:
#                     dis[_i][_j] = dis[_i][_k] + dis[_k][_j]
#     # for _i in range(args.num_node):
#     #     for _j in range(args.num_node):
#     #         if (dis[_i][_j] != INF):
#     #             print(_i, _j, dis[_i][_j])
#     return dis
#
#
# def spatial_graph(dis, id2ent):
#     lst = []
#     w = np.zeros([num_node, num_node])
#     for _i in range(num_node):
#         for _j in range(num_node):
#             if _i == _j:
#                 w[_i, _j] = 1
#             if _i != _j and dis[_i][_j] != 0 and dis[_i][_j] <= link_cap:
#                 lst.append((id2ent[_i], "adj_spat", id2ent[_j]))
#                 w[_i, _j] = 1 * (decay_ratio ** (dis[_i][_j] - 1))
#
#     w_sg_path = os.path.join(sg_prefix, 'w_sg_{}.csv'.format(dataset))
#     pd.DataFrame(w).to_csv(w_sg_path, index=False, header=None)
#     print("The weight matrix of spatial graph is generated!")
#
#     tri_sg_path = os.path.join(sg_prefix, 'tri_sg_{}.txt'.format(dataset))
#     toTxt(lst, tri_sg_path)
#     print("Triples of spatial graph is generated!")
#
#
# if __name__ == '__main__':
#     dataset_rel_df = pd.read_csv(dataset_rel_path)
#     edge = getEdge(dataset_rel_df)
#     dis = floyed(edge)
#     spatial_graph(dis, id2ent)
#     # print(dis)
#
#
