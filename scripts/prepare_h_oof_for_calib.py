#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser("merge H-chain OOF with labels for calibrator")
    ap.add_argument("--oof", required=True, help="H-chain OOF csv, columns like: idx,oof")
    ap.add_argument("--labels", required=True, help="labels csv, e.g. data/processed/pair_features.csv")
    ap.add_argument("--out", required=True, help="output csv with columns: ddg_true,ddg_pred")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    if "oof" in oof.columns and "ddg_pred" not in oof.columns:
        oof = oof.rename(columns={"oof": "ddg_pred"})
    if "idx" not in oof.columns or "ddg_pred" not in oof.columns:
        raise SystemExit(f"OOF需要包含列 idx, oof/ddg_pred，当前列: {list(oof.columns)}")

    lab = pd.read_csv(args.labels).reset_index().rename(columns={"index": "idx"})
    # 只用 H 链
    if "_chain" in lab.columns:
        lab = lab[lab["_chain"] == "H"].copy()

    # 标签列名兼容：优先 ddG / 其次 ddg / y
    label_col = None
    for c in ["ddG", "ddg", "y"]:
        if c in lab.columns:
            label_col = c
            break
    if label_col is None:
        raise SystemExit(f"在标签文件里找不到真值列 (ddG/ddg/y)，现有列: {list(lab.columns)}")

    merged = oof.merge(lab[["idx", label_col]], on="idx", how="inner").dropna()
    if merged.empty:
        raise SystemExit("merge 后为空，请检查 idx 是否一致。")

    out = merged[["ddg_pred", label_col]].rename(columns={label_col: "ddg_true"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ 保存: {args.out}  (n={len(out)})")

if __name__ == "__main__":
    main()
