import os
import random

import numpy as np
import torch
from pandas import DataFrame


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def is_chinese(string):
    for char in string:
        if not ('\u4e00' <= char <= '\u9fff'):
            return False
    return True


def toCsv(lst, data_path):
    df = []
    for _tuple in lst:
        df.append(list(_tuple))
    DataFrame(df).to_csv(data_path, index=False, header=False)


def toTxt(lst, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        for _tuple in lst:
            f.write('\t'.join(list(_tuple)) + '\n')


def toDict(lst):
    dct = {ent: id for ent, id in lst }
    return dct


def load_dict(data_path):
    dct = {}
    with open(data_path, 'r') as f:
        for _line in f.readlines():
            _k, _v = _line.strip().split('\t')
            dct[_k] = int(_v)
    return dct


def load_txt(data_path):
    lst = []
    with open(data_path, 'r') as f:
        for _line in f.readlines():
            _h, _r, _t = _line.strip().split('\t')
            lst.append((_h, _t, _r))
    return lst


def save_key_vlist(dct, path):
    with open(path, 'w') as f:
        for _k, _v in dct.items():
            _v = list(map(str, _v))
            f.write(_k + '\t' + str(len(_v)) + '\n')
            f.write('\t'.join(_v) + '\n')


def read_emb(emb_path):
    lst = []
    with open(emb_path, 'r') as f:
        for _line in f.readlines():
            lst.append(list(map(float, _line.strip().split('\t'))))
    arr = torch.tensor(lst)
    return arr


def load_triples(triples_path):
    lst = []
    with open(triples_path, 'r') as f:
        for _line in f.readlines():
            tuple = list(map(int, _line.strip().split('\t')))
            lst.append(tuple)
    return torch.tensor(lst)


# def ent2id(kg: DataFrame):
#     ent = []
#     for _h, _r, _t in kg.values:
#         if (_r not in poi_type):
#             ent.append(int(_h))
#             ent.append(int(_t))
#         else:
#             ent.append(int(_h))
#     ent = list(sorted(set(ent)))
#     lst = [(str(_ent), str(_id)) for _id, _ent in enumerate(ent)]
#     return lst


# def rel2id(kg: DataFrame):
#     rel = []
#     for _h, _r, _t in kg.values:
#         if (_r not in poi_type):
#             rel.append(_r)
#     rel = list(sorted(set(rel)))
#     lst = [(_rel, str(_id)) for _id, _rel in enumerate(rel)]
#     return lst


# def att2id(kg: DataFrame):
#     att = []
#     for _h, _r, _t in kg.values:
#         if (_r in poi_type):
#             att.append(_r)
#     att = list(sorted(set(att)))
#     lst = [(_att, str(_id)) for _id, _att in enumerate(att)]
#     return lst


# def val2id(kg: DataFrame):
#     val = []
#     for _h, _r, _t in kg.values:
#         if (_r in poi_type):
#             val.append(int(_t))
#     val = list(sorted(set(val)))
#     lst = [(str(_val), str(_id)) for _id, _val in enumerate(val)]
#     return lst


# def triple2id(kg: DataFrame, ent2id_dict: dict, rel2id_dict: dict) -> list[tuple[str, str, str]]:
#     triple = []
#     for _h, _r, _t in kg.values:
#         triple.append((ent2id_dict[str(_h)], rel2id_dict[_r], ent2id_dict[str(_t)]))
#     return triple


# def att_val(kg: DataFrame):
#     dct = {}
#     for _h, _r, _t in kg.values:
#         if (_r in poi_type):
#             dct.setdefault(_r, []).append(int(_t))
#     for _k, _v in dct.items():
#         dct[_k] = list(map(str, sorted(set(_v))))
#     dct = dict(sorted(dct.items()))
#     return dct


# def save_att_val(dct: dict, att_val_path: str) -> None:
#     with open(att_val_path, 'w') as f:
#         for _k, _v in dct.items():
#             f.write(_k + '\t' + str(len(_v)) + '\n')
#             f.write('\t'.join(_v) + '\n')


# def save_ent_att_val(ent_att_val_dict: dict, ent_att_val_path: str) -> None:
#     with open(ent_att_val_path, 'w') as f:
#         for _k, _v in ent_att_val_dict.items():
#             f.write(_k + '\n')
#             _v = map(str, _v)
#             f.write('\t'.join(_v) + '\n')


def rel_att_train_split(kg: DataFrame):
    rel_triple, att_triple = [], []
    for _h, _r, _t in kg.values:
        if (_r.startswith('adj')):
            rel_triple.append((str(_h), str(_t), _r))
        else:
            att_triple.append((str(_h), str(_t), _r))
    print(rel_triple)
    print(att_triple)
    random.shuffle(rel_triple)
    random.shuffle(att_triple)
    print()
    print(rel_triple)
    print(att_triple)
    return rel_triple, att_triple


# def ent_att_val(kg: DataFrame, ent2id: dict, att2id: dict, val2id: dict):
#     lst = []
#     for _h, _r, _t in kg.values:
#         if (_r in poi_type):
#             lst.append((ent2id[str(_h)], att2id[_r], val2id[str(_t)]))
#             # dct.setdefault(int(ent2id[_h]), []).append((int(att2id[_r]), int(val2id[str(_t)])))
#     # for _k, _v in dct.items():
#     #     dct[_k] = sorted(_v)
#     # dct = dict(sorted(dct.items()))
#     # dct = dict(zip(map(str, dct.keys()), dct.values()))
#     # return dct
#     return lst


def ent_rel_ent(kg: DataFrame, ent2id: dict, rel2id: dict):
    lst = []
    for _h, _r, _t in kg.values:
        if (_r.startswith('adj')):
            lst.append((ent2id[str(_h)], rel2id[_r], ent2id[str(_t)]))
    return lst


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std