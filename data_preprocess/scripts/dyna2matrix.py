#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert LibCity-style *.dyna (with inflow/outflow) to SZ_TAXI-like matrix CSV.

- Input:  dyna CSV with columns: dyna_id,type,time,entity_id,inflow,outflow
- Output: two CSV files (inflow/outflow). Each file:
    * first row: ordered entity_id list (as header)
    * subsequent rows: values per timestamp (ordered by time)
    * no time column (same style as sz_speed.csv)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd


dataset, model = "NYCTAXI20140103", "KST_GCN"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
dataset_cfg = settings[dataset]
dataset_prefix = dataset_cfg["dataset_prefix"]
cfg = settings[dataset][model]


def _read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_dyna_path(cfg: dict, config_path: Path, dyna_path: Optional[Path]) -> Path:
    if dyna_path is not None:
        return dyna_path
    # default: dyna file in the same folder as config.json, name from info.data_files[0]
    data_files = cfg.get("info", {}).get("data_files", [])
    if not data_files:
        raise ValueError("config.json 中未找到 info.data_files，无法推断 dyna 文件名。请用 --dyna 显式指定。")
    base = str(data_files[0])
    return config_path.parent / f"{base}.dyna"


def _pick_columns(df: pd.DataFrame, cfg: dict) -> Tuple[str, str]:
    """Return (in_col, out_col) in the dyna file."""
    # Prefer info.data_col if it exists
    data_col = cfg.get("info", {}).get("data_col", None)
    candidates: List[str] = []
    if isinstance(data_col, list):
        candidates = [str(x) for x in data_col]

    # common naming variants
    variants = [
        ("inflow", "outflow"),
        ("in_flow", "out_flow"),
        ("in", "out"),
    ]

    # 1) try config info.data_col (must be length>=2 and in df)
    if len(candidates) >= 2 and candidates[0] in df.columns and candidates[1] in df.columns:
        return candidates[0], candidates[1]

    # 2) try common variants
    for a, b in variants:
        if a in df.columns and b in df.columns:
            return a, b

    raise ValueError(
        f"无法在 dyna 中找到 inflow/outflow 列。当前列名: {list(df.columns)}。"
        f"请检查 dyna 表头，或在 config.json 的 info.data_col 中写对列名。"
    )


def dyna_to_matrix_csv(
    dyna_path: Path,
    cfg: dict,
    out_dir: Path,
    out_prefix: str,
    # fill_value: float = 0.0,
    # keep_time_index: bool = False,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dyna_path)

    required = {"time", "entity_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"dyna 缺少必要列 {required}，当前列名: {list(df.columns)}")

    in_col, out_col = _pick_columns(df, cfg)

    # Parse time as UTC; your dyna has 'Z'
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        bad = df[df["time"].isna()].head(5)
        raise ValueError(f"time 列存在无法解析的值，示例:\n{bad}")

    # entity_id -> int
    df["entity_id"] = pd.to_numeric(df["entity_id"], errors="coerce")
    if df["entity_id"].isna().any():
        bad = df[df["entity_id"].isna()].head(5)
        raise ValueError(f"entity_id 列存在无法解析的值，示例:\n{bad}")
    df["entity_id"] = df["entity_id"].astype(int)

    # Order
    entity_ids = sorted(df["entity_id"].unique().tolist())
    t_min, t_max = df["time"].min(), df["time"].max()

    # Build expected full time index if time_intervals exists; else use observed times
    step = cfg.get("info", {}).get("time_intervals", None)
    if isinstance(step, (int, float)) and step > 0:
        freq = pd.Timedelta(seconds=int(step))
        time_index = pd.date_range(start=t_min, end=t_max, freq=freq, tz="UTC")
    else:
        time_index = pd.Index(sorted(df["time"].unique()), name="time")

    def _pivot(value_col: str) -> pd.DataFrame:
        mat = df.pivot_table(index="time", columns="entity_id", values=value_col, aggfunc="mean")
        mat = mat.reindex(index=time_index)
        mat = mat.reindex(columns=entity_ids)
        return mat

    inflow_mat = _pivot(in_col)
    outflow_mat = _pivot(out_col)

    inflow_path = out_dir / f"{out_prefix}_inflow.csv"
    outflow_path = out_dir / f"{out_prefix}_outflow.csv"

    inflow_mat.to_csv(inflow_path, index=False)
    outflow_mat.to_csv(outflow_path, index=False)
    return inflow_path, outflow_path


def main():
    config_path = Path(f"{dataset_prefix}/config.json")
    dyna_path = Path(f"{dataset_prefix}/{dataset}.dyna")
    out_dir = Path(dataset_prefix)

    cfg = _read_config(config_path)
    inflow_path, outflow_path = dyna_to_matrix_csv(
        dyna_path,
        cfg,
        out_dir,
        out_prefix=dataset
    )
    print(f"OK: wrote {inflow_path}")
    print(f"OK: wrote {outflow_path}")


if __name__ == "__main__":
    main()
