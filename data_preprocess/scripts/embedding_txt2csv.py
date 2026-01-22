#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/txt2csv.py

将 KR-EAR/OpenKE 输出的 entity2vec0.txt 转为 CSV，并保存到 kg_assist 目录，
文件名固定为：kg_embedding_d{dim}.csv（如 kg_embedding_d20.csv）。

默认路径（与你截图一致）：
  输入:
    data_preprocess/kg/<dataset>/KR-EAR/embedding/d{dim}/entity2vec0.txt
  输出:
    kg_assist/<dataset>/<target_model>/kg_embedding_d{dim}.csv

用法（在项目根目录运行）：
  python scripts/txt2csv.py --dataset NYCTAXI20140103 --dim 20 --target_model KST_GCN

也可手动指定路径：
  python scripts/txt2csv.py --src /abs/path/entity2vec0.txt --dst /abs/path/kg_embedding_d20.csv

说明（兼容常见格式）：
- 支持空格/Tab 分隔
- 支持首行是 "N D" 的头（会自动跳过）
- 支持每行是：
    (d0 d1 ... dD-1)                # 无实体id
  或
    (entity_id d0 d1 ... dD-1)      # 有实体id（首列非float或int）
- 默认输出 CSV 不写表头，避免影响你后续用 header=None 读取
"""

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


dataset, model = "TDRIVE20150406", "DMKG_GNN"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset][model]
dim = cfg["ke_dim"]
if model == "DMKG_GNN":
    sparsity = cfg["sparsity"]


def _is_int_token(s: str) -> bool:
    s = s.strip()
    if s.startswith(("+", "-")):
        s = s[1:]
    return s.isdigit()


def _try_parse_floats(tokens: List[str]) -> Optional[np.ndarray]:
    try:
        return np.array([float(t) for t in tokens], dtype=float)
    except Exception:
        return None


def read_embedding_txt(src: Path) -> Tuple[Optional[List[str]], np.ndarray]:
    """Read OpenKE/KR-EAR embedding txt."""
    lines = [ln.strip() for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"Empty file: {src}")

    # Header: "N D"
    first = lines[0].split()
    start_idx = 0
    expected_dim = None
    if len(first) == 2 and _is_int_token(first[0]) and _is_int_token(first[1]):
        start_idx = 1
        expected_dim = int(first[1])

    ids: Optional[List[str]] = None
    vecs: List[np.ndarray] = []

    for i in range(start_idx, len(lines)):
        toks = lines[i].split()
        if not toks:
            continue

        # Case A: all floats -> vector
        arr = _try_parse_floats(toks)
        if arr is not None:
            if expected_dim is None or arr.size == expected_dim:
                vecs.append(arr)
                continue
            # else: might be id+vector; fallthrough

        # Case B: first token is id
        if len(toks) >= 2:
            arr2 = _try_parse_floats(toks[1:])
            if arr2 is None:
                raise ValueError(f"Line {i+1} parse failed: {lines[i]}")
            if ids is None:
                ids = []
            ids.append(toks[0])
            vecs.append(arr2)
        else:
            raise ValueError(f"Line {i+1} parse failed: {lines[i]}")
    if not vecs:
        raise ValueError(f"No vectors parsed from: {src}")

    mat = np.vstack(vecs)
    if expected_dim is not None and mat.shape[1] != expected_dim:
        raise ValueError(f"Dimension mismatch: header D={expected_dim}, but parsed D={mat.shape[1]} in {src}")
    if ids is not None and len(ids) != mat.shape[0]:
        raise ValueError(f"Inconsistent id column: ids={len(ids)} vectors={mat.shape[0]} in {src}")
    return ids, mat


def write_csv(dst: Path, ids: Optional[List[str]], mat: np.ndarray, with_header: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    D = mat.shape[1]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if with_header:
            if ids is None:
                w.writerow([f"dim{i}" for i in range(D)])
            else:
                w.writerow(["entity_id"] + [f"dim{i}" for i in range(D)])
        if ids is None:
            for row in mat:
                w.writerow([f"{x:.10g}" for x in row])
        else:
            for eid, row in zip(ids, mat):
                w.writerow([eid] + [f"{x:.10g}" for x in row])


def main():

    project_root = Path(__file__).resolve().parents[1]
    parent_project_root = Path(__file__).resolve().parents[2]

    if model == "DMKG_GNN":
        src = project_root / "kg" / dataset / f"aug_kg_s{sparsity}" / "KR-EAR" /  f"d{dim}" / "entity2vec0.txt"
    else:
        src = project_root / "kg" / dataset / "kg" / "KR-EAR" / f"d{dim}" / "entity2vec0.txt"

    if model == "DMKG_GNN":
        dst = parent_project_root / "kg_assist" / dataset / model / f"{dim}d_s{sparsity}.csv"
    else:
        dst = parent_project_root / "kg_assist" / dataset / model / f"kg_embedding_d{dim}.csv"

    ids, mat = read_embedding_txt(src)
    write_csv(dst, ids, mat)

    print("[OK] converted")
    print("  src:", src)
    print("  dst:", dst)
    print("  shape:", mat.shape, ("with ids" if ids is not None else "no ids"))


if __name__ == "__main__":
    main()
