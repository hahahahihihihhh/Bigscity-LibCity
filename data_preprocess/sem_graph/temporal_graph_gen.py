#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os.path
import numpy as np
import pandas as pd
import time
from libcity.data_preprocess.utils.utils import ensure_dir, initialize_seed, normalize, toTxt

dataset = "SZ_TAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

params = cfg["params"]
road2id = cfg["road2id"]

assist_prefix = params["assist_prefix"]
dataset_path = params["dataset_path"]
num_node = params["num_node"]
order = params["temporal_graph"]["order"]
lag = params["temporal_graph"]["lag"]
period = params["temporal_graph"]["period"]
sparsity = params["temporal_graph"]["sparsity"]

id2road = {v: k for k, v in road2id.items()}
initialize_seed(43)


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


def get_dtw(dataset_df, dtw_path):
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
            d[i, j] = compute_dtw(data[:, :, i], data[:, :, j], order=order, Ts=lag)
        t2 = time.time()
        print("Line", i, "finished in", t2 - t1, "seconds.")
    dtw = d + d.T
    np.save(dtw_path, dtw)
    print("The calculation of time series is done!")
    return dtw


def temporal_graph(dtw_path):
    adj = np.load(dtw_path)
    adj = adj + adj.T
    w_adj = np.zeros([num_node, num_node])
    w = np.zeros([num_node, num_node])
    adj_percent = sparsity
    top = int(num_node * adj_percent)
    print(top)
    min_val, max_val = adj.min(), adj.max()
    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1
            w[i, a[j]] = 1 - (adj[i, a[j]] - min_val) / (max_val - min_val)
            # w[i, a[j]] = adj[i, a[j]]
    # w = 1 - (w - w.min()) / (w.max() - w.min())
    for i in range(num_node):
        for j in range(num_node):
            if w_adj[i][j] == 1:
                w_adj[j][i] = 1
                w[j][i] = w[i][j]
            if i == j:
                w_adj[i][j] = 1
                w[i][j] = 1
    print("Total route number: ", num_node)
    print("Sparsity of adj: ", len(w_adj.nonzero()[0]) / (num_node * num_node))
    tg_prefix = os.path.join(dataset, "temporal_graph")
    ensure_dir(tg_prefix)
    tg_path = os.path.join(tg_prefix, "tg_s{}.csv".format(sparsity))
    pd.DataFrame(w_adj).to_csv(tg_path, index=False, header=None)
    print("The adjacent matrix of temporal graph is generated!")
    w_tg_path = os.path.join(tg_prefix, 'w_tg_s{}.csv'.format(sparsity))
    pd.DataFrame(w).to_csv(w_tg_path, index=False, header=None)
    print("The weight matrix of temporal graph is generated!")
    lst = []
    for i in range(num_node):
        for j in range(num_node):
            if i != j and w_adj[i][j] == 1:
                lst.append((id2road[i], 'adj_temp', id2road[j]))
    tri_tg_path = os.path.join(tg_prefix, 'tri_tg_s{}.txt'.format(sparsity))
    toTxt(lst, tri_tg_path)
    print("functional graph is generated!")


if __name__ == '__main__':
    dtw_path = os.path.join(assist_prefix, "{}-dtw-{}-{}-s{}.npy".format(dataset, period, order, sparsity))
    dataset_df = np.expand_dims(pd.read_csv(dataset_path).values, axis=2)
    dtw = get_dtw(dataset_df, dtw_path)
    temporal_graph(dtw_path)
