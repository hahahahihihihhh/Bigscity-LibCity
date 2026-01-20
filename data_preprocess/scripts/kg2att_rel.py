#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/kg2att_rel.py

将 raw_data/<dataset>/att_kg.csv (att 属性) 与 raw_data/<dataset>/rel_kg.csv (邻接关系)
转换为 KR-EAR 的输入文件（仅训练文件，不生成 test-*）：

输出到：kg/<dataset>/KR-EAR/

生成文件：
  - train-rel.txt      (e1, e2, rel)
  - train-attr.txt     (e1, val, attr)
  - entity2id.txt      (entity -> id)
  - relation2id.txt    (relation -> id)
  - attribute2id.txt   (attribute -> id)
  - val2id.txt         (value -> id)
  - attribute_val.txt  (each attribute's value set)

用法（在项目根目录运行）：
  python scripts/kg2att_rel.py --dataset NYCTAXI20140103

可选：
  python scripts/kg2att_rel.py --dataset NYCTAXI20140103 --project_root /path/to/your/project

说明：
- 本脚本会自动识别常见列名：
  att_kg.csv:  grid_id/entity_id, num/value, att_type/attr
  rel_kg.csv:  origin/head/src, destination/tail/dst, rel/relation
- 默认会对同一 (entity, attr) 的多条记录做合并：
  若 value 可转为数值 -> 求和；否则取第一个非空值。
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


dataset, model = "TDRIVE20150406", "KST_GCN"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset][model]
attribute_file_path = cfg["attribute_file_path"]
relation_file_path = cfg["relation_file_path"]
krear_dir_path = cfg["krear_dir_path"]
if model == "DMKG_GNN":
    sparsity = cfg["sparsity"]
    attribute_file_path = attribute_file_path.format(sparsity)
    relation_file_path = relation_file_path.format(sparsity, sparsity)
    krear_dir_path = krear_dir_path.format(sparsity)

# -----------------------------
# Column auto-detection helpers
# -----------------------------
def _pick_col(cols: List[str], candidates: List[str]) -> str:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    raise ValueError(f"Cannot find column among candidates={candidates} in columns={cols}")


def _normalize_token(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if float(x).is_integer():
            return str(int(x))
        return str(float(x))
    s = str(x).strip()
    # numeric string cleanup: "3.0" -> "3"
    try:
        fx = float(s)
        if fx.is_integer():
            return str(int(fx))
        return str(fx)
    except Exception:
        return s


def _sorted_tokens(tokens: Iterable[str]) -> List[str]:
    tokens = [t for t in tokens if t != ""]
    if tokens and all(t.lstrip("-").isdigit() for t in tokens):
        return sorted(tokens, key=lambda x: int(x))
    return sorted(tokens)


def _write_mapping(path: Path, tokens: List[str]) -> Dict[str, int]:
    mapping = {tok: i for i, tok in enumerate(tokens)}
    with path.open("w", encoding="utf-8") as f:
        for tok, i in mapping.items():
            f.write(f"{tok}\t{i}\n")
    return mapping


def _write_triples(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for a, b, c in rows:
            f.write(f"{a}\t{b}\t{c}\n")


def _write_attribute_val(path: Path, attr_to_vals: Dict[str, List[str]]) -> None:
    """
    attribute_val.txt 格式（与常见 KR-EAR 数据集一致）：
      attr \t K
      v1 \t v2 \t ... \t vK
    """
    with path.open("w", encoding="utf-8") as f:
        for attr in _sorted_tokens(attr_to_vals.keys()):
            vals = _sorted_tokens(set(attr_to_vals[attr]))
            f.write(f"{attr}\t{len(vals)}\n")
            f.write("\t".join(vals) + "\n")


def _merge_attr_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并同一 (entity, attr) 的多条 value：
    - 若 value 可转 float -> 求和
    - 否则取第一个非空
    """
    def _agg_value(s: pd.Series):
        # try numeric sum
        try:
            return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())
        except Exception:
            for x in s:
                t = str(x).strip()
                if t != "":
                    return t
            return ""

    g = df.groupby(["entity", "attr"], as_index=False)["value"].apply(_agg_value)
    g = g.rename(columns={"value": "value"}).copy()
    g["value"] = g["value"].map(_normalize_token)
    return g


def main():

    att_csv = attribute_file_path
    rel_csv = relation_file_path
    out_dir = Path(krear_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Load att attribute triples
    # -----------------------------
    att_df = pd.read_csv(att_csv)
    e_col = _pick_col(list(att_df.columns), ["grid_id", "entity_id", "entity", "node", "region", "id"])
    v_col = _pick_col(list(att_df.columns), ["num", "value", "val", "count"])
    a_col = _pick_col(list(att_df.columns), ["att_type", "attr", "attribute", "type", "category", "cate"])

    att_df = att_df[[e_col, v_col, a_col]].copy()
    att_df.columns = ["entity", "value", "attr"]

    att_df["entity"] = att_df["entity"].map(_normalize_token)
    att_df["value"] = att_df["value"].map(_normalize_token)
    att_df["attr"] = att_df["attr"].map(lambda x: str(x).strip())

    att_df = att_df[(att_df["entity"] != "") & (att_df["attr"] != "") & (att_df["value"] != "")].reset_index(drop=True)

    # 合并重复 (entity, attr)
    att_df = _merge_attr_duplicates(att_df)
    att_df = att_df[(att_df["entity"] != "") & (att_df["attr"] != "") & (att_df["value"] != "")].reset_index(drop=True)


    # -----------------------------
    # 2) Load adjacency relation triples
    # -----------------------------
    rel_df = pd.read_csv(rel_csv)
    h_col = _pick_col(list(rel_df.columns), ["origin", "head", "src", "source", "e1"])
    t_col = _pick_col(list(rel_df.columns), ["destination", "tail", "dst", "target", "e2"])
    r_col = _pick_col(list(rel_df.columns), ["rel", "relation", "edge_type", "type"])

    rel_df = rel_df[[h_col, t_col, r_col]].copy()
    rel_df.columns = ["e1", "e2", "rel"]

    rel_df["e1"] = rel_df["e1"].map(_normalize_token)
    rel_df["e2"] = rel_df["e2"].map(_normalize_token)
    rel_df["rel"] = rel_df["rel"].map(lambda x: str(x).strip())

    rel_df = rel_df[(rel_df["e1"] != "") & (rel_df["e2"] != "") & (rel_df["rel"] != "")].reset_index(drop=True)


    # -----------------------------
    # 3) Write train files only
    # -----------------------------
    _write_triples(out_dir / "train-attr.txt", att_df[["entity", "value", "attr"]].itertuples(index=False, name=None))
    _write_triples(out_dir / "train-rel.txt", rel_df[["e1", "e2", "rel"]].itertuples(index=False, name=None))

    # -----------------------------
    # 4) Build & write mappings
    # -----------------------------
    entities = _sorted_tokens(set(att_df["entity"]).union(set(rel_df["e1"])).union(set(rel_df["e2"])))
    relations = _sorted_tokens(rel_df["rel"].unique().tolist())
    attributes = _sorted_tokens(att_df["attr"].unique().tolist())
    values = _sorted_tokens(att_df["value"].unique().tolist())

    _write_mapping(out_dir / "entity2id.txt", entities)
    _write_mapping(out_dir / "relation2id.txt", relations)
    _write_mapping(out_dir / "attribute2id.txt", attributes)
    _write_mapping(out_dir / "val2id.txt", values)

    # -----------------------------
    # 5) attribute_val.txt
    # -----------------------------
    attr_to_vals: Dict[str, List[str]] = {}
    for attr, sub in att_df.groupby("attr"):
        attr_to_vals[attr] = sub["value"].astype(str).tolist()
    _write_attribute_val(out_dir / "attribute_val.txt", attr_to_vals)

    print(f"[OK] dataset={dataset}")
    print(f"     input : {att_csv} , {rel_csv}")
    print(f"     output: {out_dir.resolve()}")
    for name in [
        "train-rel.txt",
        "train-attr.txt",
        "entity2id.txt",
        "relation2id.txt",
        "attribute2id.txt",
        "val2id.txt",
        "attribute_val.txt",
    ]:
        p = out_dir / name
        print(f"  - {name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
