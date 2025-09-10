#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, argparse
import numpy as np, pandas as pd
from pathlib import Path
from utils import load_cfg, normalize_labels

def parse_chain(mut):
    m = re.match(r'^\s*([A-Za-z0-9])\s*:', str(mut).strip())
    return m.group(1).upper() if m else None

def load_emb(p: Path): 
    with np.load(p.as_posix()) as d:
        return d["emb"].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-wt", action="store_true",
                    help="将 WT 的 H/L 向量也写入特征表 (wH_*/wL_*)；默认关闭以完全兼容旧版")
    args = ap.parse_args()

    cfg = load_cfg("configs/paths.yaml")
    labels = normalize_labels(pd.read_csv(cfg["labels_csv"], encoding="latin1")).copy()
    emb_dir = Path(cfg["embeddings_dir"])
    out_csv = Path("data/processed/pair_features.csv"); out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows, miss = [], 0
    for pdb, mut in zip(labels["#PDB"], labels["Mutation"]):
        ch = parse_chain(mut)
        if ch not in ("H", "L"):
            continue
        tag = str(mut).replace(":", "_")

        f_wt_H = emb_dir / f"{pdb}_H__WT.npz"
        f_wt_L = emb_dir / f"{pdb}_L__WT.npz"
        f_mut  = emb_dir / f"{pdb}_{ch}__{tag}.npz"
        if not (f_wt_H.is_file() and f_wt_L.is_file() and f_mut.is_file()):
            miss += 1; continue

        wtH, wtL = load_emb(f_wt_H), load_emb(f_wt_L)
        mut_vec  = load_emb(f_mut)

        if ch == "H":
            dH, dL = (mut_vec - wtH), np.zeros_like(wtL)
        else:
            dH, dL = np.zeros_like(wtH), (mut_vec - wtL)

        ddg = pd.to_numeric(
            labels.loc[(labels["#PDB"] == pdb) & (labels["Mutation"] == mut), "ddG"],
            errors="coerce"
        )
        ddg = float(ddg.values[0]) if len(ddg) else np.nan

        row = {"_pdb": pdb, "_chain": ch, "_mutation": mut, "ddG": ddg}
        row.update({f"zH_{i}": float(v) for i, v in enumerate(dH)})
        row.update({f"zL_{i}": float(v) for i, v in enumerate(dL)})

        if args.with_wt:
            row.update({f"wH_{i}": float(v) for i, v in enumerate(wtH)})
            row.update({f"wL_{i}": float(v) for i, v in enumerate(wtL)})

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv.as_posix(), index=False)
    print(f"✅ 写入 {out_csv.resolve()}  行数 {len(df)}  （跳过缺文件 {miss}）")

if __name__ == "__main__":
    main()
