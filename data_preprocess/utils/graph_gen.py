import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from libcity.data_preprocess.utils.utils import normalize


def _equal(x, y):
    eps = 1e-6
    return math.fabs(x - y) < eps


def struct_graph_gen(edges, sparsity, num_node):
    # floyed 计算任意两点间路径
    dist = [[0 for _ in range(num_node)] for _ in range(num_node)]
    for u, v in edges:
        dist[u][v] = 1
    for _i in range(num_node):
        for _j in range(num_node):
            if _i != _j and not dist[_i][_j]:
                dist[_i][_j] = math.inf
    for _i in range(num_node):
        for _j in range(num_node):
            for _k in range(num_node):
                dist[_i][_j] = min(dist[_i][_j], dist[_i][_k] + dist[_k][_j])
    # 边权重计算
    weighted_edges = []
    for _i in range(num_node):
        for _j in range(num_node):
            if _i != _j:
                weighted_edges.append((_i, _j, 0.5 ** (dist[_i][_j] - 1)))
    weighted_edges.sort(key=lambda x: x[2], reverse=True)
    end = min(len(weighted_edges), int(num_node * (num_node - 1) * sparsity))
    while end < len(weighted_edges) and _equal(weighted_edges[end][2], weighted_edges[end - 1][2]):
        end += 1
    shifted_edges = weighted_edges[:end]
    max_value, min_value = shifted_edges[0][2], shifted_edges[-1][2]
    struct_mat = [[0 for _ in range(num_node)] for _ in range(num_node)]
    for edge in shifted_edges:
        if max_value == min_value:
            struct_mat[edge[0]][edge[1]] = 1
        else:
            struct_mat[edge[0]][edge[1]] = (edge[2] - min_value) / (max_value - min_value)
    print(f"struct graph sparsity: {len(shifted_edges) / len(weighted_edges)}")
    return struct_mat


def function_graph_gen(poi2id, poi_kg, sparsity, num_node):
    poi_mat = np.zeros((num_node, len(poi2id)))
    for poi_tuple in poi_kg:
        poi_mat[poi_tuple[0]][poi_tuple[1]] = poi_tuple[2]
    poi_nodes = []
    for _ in range(num_node):
        if poi_mat[_].any():
            poi_nodes.append(_)
    weighted_edges = []
    for _x in poi_nodes:
        for _y in poi_nodes:
            if _x == _y:    continue
            pearson_corr, _ = pearsonr(poi_mat[_x], poi_mat[_y])
            pearson_corr2, _ = pearsonr(poi_mat[_y], poi_mat[_x])
            assert (pearson_corr == pearson_corr2)
            weighted_edges.append((_x, _y, pearson_corr))
    weighted_edges.sort(key=lambda x: x[2], reverse=True)
    end = min(len(weighted_edges), int(num_node * (num_node - 1) * sparsity))
    while end < len(weighted_edges) and _equal(weighted_edges[end][2], weighted_edges[end - 1][2]):
        end += 1
    shifted_edges = weighted_edges[:end]
    max_value, min_value = shifted_edges[0][2], shifted_edges[-1][2]
    function_mat = [[0 for _ in range(num_node)] for _ in range(num_node)]
    for edge in shifted_edges:
        if max_value == min_value:
            function_mat[edge[0]][edge[1]] = 1
        else:
            function_mat[edge[0]][edge[1]] = (edge[2] - min_value) / (max_value - min_value)
    print(f"function graph sparsity: {len(shifted_edges) / len(weighted_edges)}")
    return function_mat


def compute_dtw(a, b, order=1, Ts=12, normal=True):
    if normal:
        a = normalize(a)
        b = normalize(b)
    T0 = a.shape[1]
    # assert T0 == Ts
    d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])
    d = np.linalg.norm(d, axis=0, ord=order)
    D = np.zeros([T0, T0])
    for i in range(T0):
        for j in range(max(0, i - Ts), min(T0, i + Ts + 1)):
            if i == 0 and j == 0:
                D[i, j] = d[i, j] ** order
                continue
            if i == 0:
                D[i, j] = d[i, j] ** order + D[i, j - 1]
                continue
            if j == 0:
                D[i, j] = d[i, j] ** order + D[i - 1, j]
                continue
            if j == i - Ts:
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j])
                continue
            if j == i + Ts:
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i, j - 1])
                continue
            D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1] ** (1.0 / order)


def get_dtw(dataset_df, dtw_path, period, num_node):
    if os.path.exists(dtw_path):
        print(f"{dtw_path} already exists, skip.")
        return
    num_samples = dataset_df.shape[0]
    num_train = int(num_samples * 0.7)
    num_dtw = int(num_train / period) * period
    data = dataset_df[:num_dtw, :, :1].reshape([-1, period, num_node])  # (day, time_slot, ndim)
    d = np.zeros([num_node, num_node])
    for i in range(num_node):
        t1 = time.time()
        for j in range(i + 1, num_node):
            d[i, j] = compute_dtw(data[:, :, i], data[:, :, j])
        t2 = time.time()
        print("Line", i, "finished in", t2 - t1, "seconds.")
    dtw = d + d.T
    np.save(dtw_path, dtw)
    print("The calculation of time series is done!")
    return dtw


def pattern_graph_gen(dataset_df_inflow, dtw_inflow_path, dataset_df_outflow, dtw_outflow_path, period, num_node, sparsity, active_nodes):
    get_dtw(dataset_df_inflow, dtw_inflow_path, period, num_node)
    get_dtw(dataset_df_outflow, dtw_outflow_path, period, num_node)
    adj_inflow = np.load(dtw_inflow_path)
    adj_outflow = np.load(dtw_outflow_path)
    adj_inflow = adj_inflow + adj_inflow.T
    adj_outflow = adj_outflow + adj_outflow.T
    adj = (adj_inflow + adj_outflow) / 2.0
    w = np.zeros([num_node, num_node])
    top = int(num_node * sparsity)
    min_val, max_val = adj.min(), adj.max()
    for i in range(num_node):
        for j in range(num_node):
            if i not in active_nodes or j not in active_nodes:
                adj[i, j] = math.inf   # 仅考虑活跃结点
    shifted_edges = []
    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            if adj[i, a[j]] == math.inf:
                continue
            w[i, a[j]] = 1 - (adj[i, a[j]] - min_val) / (max_val - min_val)
            if w[i, a[j]]:
                shifted_edges.append([i, a[j], w[i, a[j]]])
    pattern_mat = [[0 for _ in range(num_node)] for _ in range(num_node)]
    for edge in shifted_edges:
        pattern_mat[edge[0]][edge[1]] = edge[2]
    print(f"pattern graph sparsity: {len(shifted_edges) / (num_node * (num_node - 1))}")
    return pattern_mat
