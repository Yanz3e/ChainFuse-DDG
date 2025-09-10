#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict_ddg_overall_v2.py
--------------------------------
V2 版一体化预测脚本（与原脚本区别开）
- 生成 WT 的所有单点突变
- 用 ESM-1b 做句子级 mean embedding
- 载入线性能量头，按 E(mut) - E(wt) 得到 ΔΔG
- 可选：载入等距回归校准器，输出 pred_ddg_cal
- 写出 ALL 表与 TopK 表

依赖：
  pip install torch fair-esm pandas numpy scikit-learn joblib biopython

用法示例：
  python scripts/predict_ddg_overall_v2.py \
    --wt input/WT.fasta \
    --positions 22-148 \
    --esm-ckpt models/checkpoints/esm1b_t33_650M_UR50S.pt \
    --model models/energy_H_rank96_full.pt \
    --scaler-h models/scaler_H.pkl \
    --scaler-l models/scaler_L.pkl \
    --calib models/calib_H_iso.pkl \
    --mut-dir Sample_Mutant \
    --out-dir output \
    --tag VHH \
    --topk 10 \
    --top-mode neg \
    --device cuda \
    --batch-size 8
"""

from __future__ import annotations
import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import joblib

try:
    import esm
except Exception as e:
    print("[fatal] 未找到 fair-esm，请先 `pip install fair-esm`", file=sys.stderr)
    raise

# -------------------------------
# 1) 校准器：与训练时同名类 + 鲁棒加载
# -------------------------------

from sklearn.isotonic import IsotonicRegression

class DDGCalibrator:
    """与训练时一致的校准器封装：iso 回归 + x_min/x_max 裁剪。"""
    def __init__(self, x=None, y=None, x_min=None, x_max=None):
        self.iso = IsotonicRegression(out_of_bounds='clip')
        if x is not None and y is not None:
            self.iso.fit(np.asarray(x, float), np.asarray(y, float))
            self.x_min = float(np.min(x)) if x_min is None else float(x_min)
            self.x_max = float(np.max(x)) if x_max is None else float(x_max)
        else:
            self.x_min = float('-inf') if x_min is None else float(x_min)
            self.x_max = float('inf') if x_max is None else float(x_max)

    def predict(self, x):
        x = np.asarray(x, float)
        if np.ndim(x) == 0:
            x = x[None]
        x = np.clip(x, self.x_min, self.x_max)
        y = self.iso.predict(x)
        return np.asarray(y, float)

def load_calibrator(path: str | Path) -> DDGCalibrator:
    """兼容 obj/dict 两种存档；并解决 joblib 反序列化找不到类名问题。"""
    obj = joblib.load(path)
    # 字典版：{'iso': IsotonicRegression, 'x_min':..., 'x_max':...}
    if isinstance(obj, dict) and 'iso' in obj:
        cal = DDGCalibrator.__new__(DDGCalibrator)
        cal.iso = obj['iso']
        cal.x_min = float(obj.get('x_min', -np.inf))
        cal.x_max = float(obj.get('x_max',  np.inf))
        return cal
    # 对象版：直接返回
    if hasattr(obj, 'predict') and hasattr(obj, 'iso'):
        if not hasattr(obj, 'x_min'): obj.x_min = -np.inf
        if not hasattr(obj, 'x_max'): obj.x_max =  np.inf
        return obj
    raise ValueError(f"Unsupported calibrator object at {path}: {type(obj)}")

def apply_calibration(cal: DDGCalibrator, arr: np.ndarray) -> np.ndarray:
    y = cal.predict(arr)
    return np.asarray(y, float).reshape(-1)


# -------------------------------
# 2) 实用函数
# -------------------------------

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def read_single_fasta(path: str | Path) -> str:
    txt = Path(path).read_text().strip()
    # 支持简易 fasta 与纯序列
    if txt.startswith(">"):
        lines = [ln.strip() for ln in txt.splitlines() if ln and not ln.startswith(">")]
        seq = "".join(lines)
    else:
        seq = "".join([c for c in txt.splitlines() if c and not c.startswith(">")])
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return seq

def parse_positions(spec: str) -> List[int]:
    """'22-148' 或 '22,23,45' -> 1-based 位置 list"""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    parts = [int(x) for x in re.split(r"[,\s]+", spec) if x]
    return parts

def enumerate_mutants(seq: str, positions: List[int], alts: Iterable[str]) -> Tuple[List[str], List[Tuple[int,str,str]]]:
    """返回所有单突变的序列列表、以及 (pos, wt, mut) 列表"""
    seq = seq.upper()
    n = len(seq)
    alts = [a.upper() for a in alts]
    muts = []
    records = []
    for pos1 in positions:
        if pos1 < 1 or pos1 > n:
            continue
        wt = seq[pos1 - 1]
        for a in alts:
            if a == wt: 
                continue
            new_seq = seq[:pos1-1] + a + seq[pos1:]
            muts.append(new_seq)
            records.append((pos1, wt, a))
    return muts, records

def load_esm_model_and_alphabet(ckpt: str | Path):
    # 注意：local 接口，不能传 regression_location
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(ckpt))
    model.eval()
    return model, alphabet

@torch.no_grad()
def embed_mean(model, alphabet, seqs: List[str], device="cuda", batch_size=8) -> np.ndarray:
    """对每条序列计算 token 平均（去掉 BOS/EOS），返回 (N, D)"""
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    outs = []

    # 分批
    for i in range(0, len(seqs), batch_size):
        sub = seqs[i:i+batch_size]
        data = [("id", s) for s in sub]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        # ESM-1b: per-token representation at layer 33 by default
        # 直接用最后层表示
        rep = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]  # (B, L, D)
        # 去掉 BOS/EOS：alphabet.cls_idx / alphabet.eos_idx
        mask = (toks != alphabet.cls_idx) & (toks != alphabet.eos_idx) & (toks != alphabet.padding_idx)
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        rep = rep * mask
        lengths = mask.sum(dim=1)  # (B, 1)
        lengths = lengths.clamp(min=1)
        mean = rep.sum(dim=1) / lengths  # (B, D)
        outs.append(mean.detach().cpu().numpy())

    return np.vstack(outs) if outs else np.zeros((0, model.embed_dim), dtype=np.float32)

def safe_load_scaler(pkl_path: Optional[str], dim: int) -> callable:
    """读不到 scaler 就返回恒等映射。"""
    if not pkl_path:
        return lambda x: x
    try:
        sc = joblib.load(pkl_path)
        # 尝试 transform
        _ = sc.transform(np.zeros((1, dim), np.float32))
        return lambda x: sc.transform(x)
    except Exception as e:
        print(f"[warn] 读取 scaler 失败，使用恒等映射: {pkl_path} | {e}")
        return lambda x: x

def _flatten_to_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    """把各种 (D,), (1,D), (D,1) 等形状，稳妥压成 (D,)。"""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] != dim:
            raise RuntimeError(f"权重长度不匹配: {x.shape} vs {dim}")
        return x
    if x.ndim == 2:
        r, c = x.shape
        # 常见： (1, D) / (D, 1)
        if r == 1 and c == dim:
            return x.reshape(-1)
        if c == 1 and r == dim:
            return x.reshape(-1)
        # 有些导出会给 (D,) -> (1,D) 或 (D,1)，上面已涵盖
        # 退路：如果其中一维等于 dim，选这一维
        if r == dim:
            return x[:, 0].reshape(-1) if c == 1 else x[0:dim, 0].reshape(-1)
        if c == dim:
            return x[0, :].reshape(-1) if r == 1 else x[0, 0:dim].reshape(-1)
    # 其他维度，尽量 squeeze 后再判断
    xs = np.squeeze(x)
    if xs.ndim == 1 and xs.shape[0] == dim:
        return xs.astype(np.float32)
    raise RuntimeError(f"无法将权重 reshape 到 (D,): got {x.shape}, dim={dim}")


def infer_weight_heads(state: dict, dim: int) -> Tuple[np.ndarray, float]:
    """
    从 state 里尽可能找到一个线性头：
      - 优先 ['wH.weight', 'wH'] / 以及 bias 键
      - 其次 ['w.weight', 'head.weight', 'H.weight']
      - 若都没有，备选 ['U.weight', 'V.weight'] 中能展平成 (D,) 的一个
    """
    if not isinstance(state, dict):
        raise RuntimeError(f"state 不是 dict: {type(state)}")

    keys = list(state.keys())

    # 常见嵌套展开
    for nest_key in ["state", "state_dict", "model_state_dict"]:
        if nest_key in state and isinstance(state[nest_key], dict):
            state = state[nest_key]
            keys = list(state.keys())
            break

    # 1) 优先 H 头
    wt_candidates = ["wH.weight", "wH", "weightH", "linear.weightH"]
    bias_candidates = ["wH.bias", "biasH", "linear.biasH", "bias", "b"]

    for k in wt_candidates:
        if k in state:
            w = _flatten_to_dim(np.asarray(state[k]), dim)
            # 找 bias
            b = 0.0
            for kb in bias_candidates:
                if kb in state:
                    b = float(np.asarray(state[kb]).reshape(-1)[0])
                    break
            print(f"[info] 使用权重键: {k}, bias 键: {kb if 'kb' in locals() else '(none)'}")
            return w, b

    # 2) 退化到通用头
    wt_candidates2 = ["w.weight", "w", "head.weight", "H.weight", "linear.weight"]
    bias_candidates2 = ["bias", "head.bias", "H.bias", "linear.bias", "b"]

    for k in wt_candidates2:
        if k in state:
            w = _flatten_to_dim(np.asarray(state[k]), dim)
            b = 0.0
            for kb in bias_candidates2:
                if kb in state:
                    b = float(np.asarray(state[kb]).reshape(-1)[0])
                    break
            print(f"[info] 使用权重键: {k}, bias 键: {kb if 'kb' in locals() else '(none)'}")
            return w, b

    # 3) 再不行，尝试 U/V 里找能展平成 (D,) 的一个
    for k in ["U.weight", "V.weight", "U", "V"]:
        if k in state:
            try:
                w = _flatten_to_dim(np.asarray(state[k]), dim)
                print(f"[warn] 未找到显式线性头，退而用 {k} 作为 w（无 bias）")
                return w, 0.0
            except Exception:
                pass

    raise RuntimeError(f"未找到线性权重向量，state keys(示例)={keys[:20]} ...")


def energy_linear(z: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """ E = z @ w + b """
    return z @ w + b


# -------------------------------
# 3) 主流程
# -------------------------------

def main():
    import argparse, os, math, csv, json, pickle, joblib, torch
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from esm import pretrained

    # ---------- utils inlined ----------
    def parse_positions(spec: str) -> list[int]:
        """支持 '22-148' 或 '22,23,25-30' """
        out = []
        for token in spec.split(","):
            token = token.strip()
            if "-" in token:
                a, b = token.split("-")
                out.extend(list(range(int(a), int(b) + 1)))
            else:
                out.append(int(token))
        return out

    def read_single_fasta(path: str) -> str:
        seq = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                if line.startswith(">"): 
                    continue
                seq.append(line)
        if not seq:
            raise RuntimeError(f"没有在 {path} 读到序列")
        return "".join(seq).upper()

    AAs = list("ACDEFGHIKLMNPQRSTVWY")

    def build_mutants(wt: str, positions: list[int], alts: str) -> list[tuple[int, str, str]]:
        """
        返回 [(pos, wt_aa, mut_aa), ...]，pos 用 1-based（与输入一致）
        alts: 'ALL' 或 由 AAs 组成的字符串。
        """
        ret = []
        for pos in positions:
            if pos < 1 or pos > len(wt):
                raise RuntimeError(f"位点 {pos} 越界(1..{len(wt)})")
            wt_aa = wt[pos - 1]
            cand = AAs if alts.upper() == "ALL" else [c for c in alts if c in AAs]
            for aa in cand:
                if aa != wt_aa:
                    ret.append((pos, wt_aa, aa))
        return ret

    def write_all_mut_fasta(wt: str, muts: list[tuple[int, str, str]], out_fa: str):
        """把所有单突变写成 FASTA，方便复用/检查"""
        os.makedirs(os.path.dirname(out_fa), exist_ok=True)
        with open(out_fa, "w") as w:
            for i, (pos, wt_aa, mu) in enumerate(muts, 1):
                s = list(wt)
                s[pos - 1] = mu
                s = "".join(s)
                w.write(f">{i}|pos={pos}|{wt_aa}->{mu}\n")
                w.write(s + "\n")

    def load_esm1b_from_ckpt(ckpt: Path, device: str):
        model, alphabet = pretrained.load_model_and_alphabet_local(str(ckpt))
        model.eval()
        model = model.to(device)
        return model, alphabet

    @torch.no_grad()
    def esm_embed_mean(model, alphabet, seqs: list[str], device: str, batch_size: int) -> np.ndarray:
        """
        对每条序列提取 ESM-1b 最后一层 token 表征并取平均（去掉 BOS/EOS/PAD）。
        返回 [N, D]
        """
        batch_converter = alphabet.get_batch_converter()
        layer = model.num_layers  # ESM-1b: 33
        outs = []
        for i in range(0, len(seqs), batch_size):
            batch = [("seq", s) for s in seqs[i:i + batch_size]]
            _, _, toks = batch_converter(batch)
            toks = toks.to(device)
            out = model(toks, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer]   # [B, L, D]
            # 掩码：非 PAD/CLS/EOS
            mask = (toks != alphabet.padding_idx) & (toks != alphabet.cls_idx) & (toks != alphabet.eos_idx)
            # 对每个样本做 masked mean
            for b in range(reps.size(0)):
                m = mask[b]
                vec = reps[b][m].mean(dim=0)   # [D]
                outs.append(vec.cpu().numpy())
        return np.stack(outs, axis=0).astype(np.float32)

    def load_calibrator(path: str):
        """
        兼容两种保存方式：
          1) 直接 joblib.dump(cal_obj)    -> 反序列化 cal 对象
          2) joblib.dump({"iso": iso, "x_min":..., "x_max":...})
        返回 (cal, x_min, x_max)
        """
        obj = joblib.load(path)
        if isinstance(obj, dict) and "iso" in obj:
            iso = obj["iso"]
            x_min = float(obj.get("x_min", -np.inf))
            x_max = float(obj.get("x_max",  np.inf))
            return iso, x_min, x_max
        # 直接是对象：尝试从对象上读范围，读不到就用无穷
        x_min = getattr(obj, "x_min", -np.inf)
        x_max = getattr(obj, "x_max",  np.inf)
        return obj, float(x_min), float(x_max)

    def apply_calibrator(cal, x_min, x_max, x: np.ndarray) -> np.ndarray:
        xx = np.clip(x, x_min, x_max)
        return cal.predict(xx.reshape(-1, 1)).reshape(-1)

    def plot_volcano(x: np.ndarray, out_png: str, title: str):
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.figure(figsize=(5.0, 4.2), dpi=160)
        xs = np.arange(len(x))
        plt.scatter(xs, x, s=6, alpha=0.6)
        plt.axhline(0.0, color="gray", ls="--", lw=0.8)
        plt.xlabel("mutants")
        plt.ylabel("ΔΔG (pred)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    # ---------- args ----------
    p = argparse.ArgumentParser()
    p.add_argument("--wt", required=True, help="WT fasta")
    p.add_argument("--positions", required=True, help="如 22-148 或 22,23,25-30")
    p.add_argument("--alts", default="ALL", help="备选氨基酸集合, e.g. 'ALL' or 'ACDE...'; 默认 ALL")
    p.add_argument("--esm-ckpt", required=True, help="ESM-1b 权重 (esm1b_t33_650M_UR50S.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    # 线性头 + (可选) 校准器
    p.add_argument("--model", required=True, help="energy .pt (含线性头权重)")
    p.add_argument("--calib", default=None, help="可选：校准器 pkl")
    # 输出 & 选择
    p.add_argument("--mut-dir", default="Sample_Mutant")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--tag", default="VHH")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--top-mode", choices=["neg", "pos"], default="neg",
                   help="选 topk：neg=ΔΔG 最负; pos=ΔΔG 最正")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1. 读取 WT & 生成全部单突变 ----------
    wt = read_single_fasta(args.wt)
    pos_list = parse_positions(args.positions)
    muts = build_mutants(wt, pos_list, args.alts)
    print(f"✅ 总体模式 | 位点数={len(pos_list)} | 单突变数={len(muts)} | alts={args.alts}")

    # 记录所有突变序列到 FASTA（方便检查/复用）
    all_fa = Path(args.mut_dir) / args.tag / "ALL_mutants.fasta"
    write_all_mut_fasta(wt, muts, str(all_fa))
    print(f"🧬 所有突变序列保存：{all_fa}")

    # ---------- 2. ESM 嵌入 ----------
    print(f"🧠 使用本地 ESM-1b 权重：{args.esm_ckpt}")
    model, alphabet = load_esm1b_from_ckpt(Path(args.esm_ckpt), args.device)
    # WT + mutants 的均值嵌入
    mut_seqs = []
    for pos, wt_aa, mu in muts:
        s = list(wt); s[pos-1] = mu; mut_seqs.append("".join(s))

    wt_vec = esm_embed_mean(model, alphabet, [wt], args.device, batch_size=1)[0]   # [D]
    mut_vecs = esm_embed_mean(model, alphabet, mut_seqs, args.device, batch_size=args.batch_size)  # [N,D]
    dim = wt_vec.shape[0]
    print(f"[embed] D = {dim}, wt_vec norm={float(np.linalg.norm(wt_vec)):.4f}")

    # ---------- 3. 读取能量头 ----------
    state = torch.load(args.model, map_location="cpu")
    # 可能外层嵌套了一层 dict
    for nest_key in ["state", "state_dict", "model_state_dict"]:
        if isinstance(state, dict) and nest_key in state and isinstance(state[nest_key], dict):
            state = state[nest_key]
            break

    w, b = infer_weight_heads(state, dim)   # ← 用我给你的增强版
    # 统一为 numpy
    w = np.asarray(w, dtype=np.float32).reshape(-1)    # (D,)
    b = float(b)
    print(f"[head] |w|={float(np.linalg.norm(w)):.4f} , b={b:.4f}")

    # ---------- 4. 计算 ΔΔG ----------
    # ΔG(mut) = w·z_mut + b； ΔG(wt) = w·z_wt + b；ΔΔG = ΔG(mut)-ΔG(wt) = w·(z_mut - z_wt)
    base = float(np.dot(w, wt_vec))
    ddg = (mut_vecs @ w) - base          # (N,)

    # ---------- 5. 可选校准 ----------
    ddg_cal = None
    if args.calib:
        try:
            cal, x_min, x_max = load_calibrator(args.calib)
            ddg_cal = apply_calibrator(cal, x_min, x_max, ddg.copy())
            print(f"[calib] 使用 {args.calib} ，范围 [{x_min:.4g}, {x_max:.4g}]")
        except Exception as e:
            print(f"[warn] 加载校准器失败，跳过：{e}")

    # ---------- 6. 导出 CSV & TopK & 火山图 ----------
    rows = []
    for i, (pos, wt_aa, mu) in enumerate(muts):
        rows.append({
            "idx": i + 1,
            "pos": pos,
            "wt": wt_aa,
            "mut": mu,
            "pred_ddg": float(ddg[i]),
            "pred_ddg_cal": float(ddg_cal[i]) if ddg_cal is not None else 0.0
        })

    all_csv = Path(args.out_dir) / f"{args.tag}_ALL_ddg.csv"
    with open(all_csv, "w", newline="") as wcsv:
        w = csv.DictWriter(wcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"📄 全量结果保存：{all_csv}")

    # TopK
    key = "pred_ddg"
    sign = -1 if args.top_mode == "neg" else +1
    sorted_idx = sorted(range(len(rows)), key=lambda i: sign * rows[i][key])[:args.topk]
    top_rows = [{"rank": r+1, **{k: rows[i][k] for k in ["pos", "wt", "mut", key]}} for r, i in enumerate(sorted_idx)]
    top_csv = Path(args.out_dir) / f"{args.tag}_top{args.topk}_{args.top_mode}.csv"
    with open(top_csv, "w", newline="") as wcsv:
        w = csv.DictWriter(wcsv, fieldnames=list(top_rows[0].keys()))
        w.writeheader()
        for r in top_rows:
            w.writerow(r)
    print(f"🏁 Top{args.topk}（按 {args.top_mode}）保存：{top_csv}")

    # 火山图（可视化 ΔΔG 分布；严格说不是 logFC/−logP 的火山，只作快速概览）
    volcano_png = Path(args.out_dir) / f"{args.tag}_volcano.png"
    plot_volcano(ddg, str(volcano_png), f"{args.tag} ΔΔG")
    print(f"🖼️ 火山图保存：{volcano_png}")

    print("✅ 结束")


if __name__ == "__main__":
    main()
