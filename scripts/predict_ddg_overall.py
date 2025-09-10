# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import re
import csv
import math
import json
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --------- 工具函数：读写 ---------
def read_single_fasta(path):
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq.append(line)
    seq = "".join(seq).strip().upper()
    if not seq:
        raise ValueError(f"Empty FASTA: {path}")
    return seq

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------- 位点 / 突变枚举 ---------
AA20 = "ACDEFGHIKLMNPQRSTVWY"

def parse_positions(pos_str, length):
    """
    解析类似 '22-148' 或 '22,23,30-35' 这样的输入，返回 1-based 的整数列表。
    """
    pos = []
    for tok in re.split(r"[,\s]+", pos_str.strip()):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            pos.extend(list(range(a, b + 1)))
        else:
            pos.append(int(tok))
    # 限制在序列长度内
    pos = [p for p in pos if 1 <= p <= length]
    if not pos:
        raise ValueError("positions 解析为空，请检查范围是否超出序列长度")
    return sorted(sorted(set(pos)))

def make_all_mutants(seq, positions, alts="ALL"):
    """
    返回 [(pos, wt, mut, mut_seq), ...]；pos 为 1-based。
    """
    if alts == "ALL":
        alphabet = AA20
    else:
        alphabet = "".join([c for c in alts.upper() if c in AA20])

    seq = seq.upper()
    L = len(seq)
    out = []
    for pos in positions:
        wt = seq[pos - 1]
        for a in alphabet:
            if a == wt:
                continue
            ms = list(seq)
            ms[pos - 1] = a
            out.append((pos, wt, a, "".join(ms)))
    return out

def write_fasta(path, records):
    """
    records: [(name, seq)]
    """
    with open(path, "w", encoding="utf-8") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            # wrap 可选
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

# --------- ESM 嵌入 ---------
def load_esm1b_from_ckpt(ckpt_path):
    """
    本地加载 ESM-1b 权重（.pt），返回 (model, alphabet)。
    兼容不同 fair-esm 版本：有的支持 regression_location，有的不支持。
    """
    import esm
    fn = esm.pretrained.load_model_and_alphabet_local
    try:
        # 新版 fair-esm 支持 regression_location
        model, alphabet = fn(str(ckpt_path), regression_location=None)
    except TypeError:
        # 旧版不支持该参数，直接调用
        model, alphabet = fn(str(ckpt_path))
    model.eval()
    return model, alphabet

def esm_embed_mean(model, alphabet, seqs, device="cuda", batch_size=8, layer=33):
    """
    对每条序列取该层表示的均值（去掉 BOS/EOS）。
    返回 numpy array (N, 1280)
    """
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i+batch_size]
            data = [("seq", s) for s in chunk]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)
            out = model(toks, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer]  # (B, L, D)
            # 平均时去掉 <cls>/<eos>
            # alphabet.toks_to_ids['<cls>']==0, '<eos>'==2（一般如此）
            # 稳妥做法：按每条真实长度掩码
            for b, s in enumerate(chunk):
                # tokens: [cls] + s + [eos]
                v = reps[b, 1:1+len(s), :].mean(0).detach().cpu().numpy()
                all_vecs.append(v)
    return np.stack(all_vecs, axis=0)

# --------- 能量模型加载：健壮解嵌套，自动匹配 ---------
def _unwrap_nested_state(d):
    if not isinstance(d, dict):
        return d
    for _ in range(6):
        for k in ["state_dict", "state", "model_state", "module", "model", "net", "energy", "params", "weights"]:
            if isinstance(d, dict) and k in d and isinstance(d[k], dict):
                d = d[k]
                break
        else:
            break
    return d

def _walk_tensors(obj, prefix=""):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items += _walk_tensors(v, prefix + k + ".")
    else:
        if torch.is_tensor(obj) or hasattr(obj, "shape"):
            items.append((prefix[:-1], obj))
    return items

def load_energy_from_any_ckpt(ckpt_path, Dh, Dl):
    """
    从任意形态 ckpt 提取 wH/wL/UH/UL/b（若不存在某项则为 None/0）。
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    sd  = _unwrap_nested_state(raw)
    if not isinstance(sd, dict):
        raise RuntimeError("checkpoint 不是 dict/state_dict 结构，无法解析")

    tensors = _walk_tensors(sd)

    def to_np(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    # 1) wH/wL: 拉平后长度 == Dh / Dl
    cand_wH = [(k, to_np(v).reshape(-1)) for k, v in tensors if hasattr(v, "shape") and np.prod(v.shape)==Dh]
    cand_wL = [(k, to_np(v).reshape(-1)) for k, v in tensors if hasattr(v, "shape") and np.prod(v.shape)==Dl]

    def pick_best(cands, regex=None):
        if not cands: return None
        if regex:
            for k,v in cands:
                if re.search(regex, k, re.I):
                    return v
        return cands[0][1]

    wH = pick_best(cand_wH, r"(^|\.)(w_h|wh|heavy|h)(\.|$)")
    wL = pick_best(cand_wL, r"(^|\.)(w_l|wl|light|l)(\.|$)")

    # 2) UH/UL: (Dh, r)/(Dl, r)
    cand_UH = [(k, to_np(v)) for k,v in tensors if hasattr(v,"shape") and len(v.shape)==2 and v.shape[0]==Dh]
    cand_UL = [(k, to_np(v)) for k,v in tensors if hasattr(v,"shape") and len(v.shape)==2 and v.shape[0]==Dl]
    UH = UL = None
    if cand_UH and cand_UL:
        by_rank = {}
        for k,v in cand_UH:
            by_rank.setdefault(v.shape[1], []).append(v)
        for k,v in cand_UL:
            r = v.shape[1]
            if r in by_rank:
                UH, UL = by_rank[r][0], v
                break
        if UH is None:
            UH = cand_UH[0][1]
        if UL is None:
            UL = cand_UL[0][1]

    # 3) bias：取第一个标量
    b = 0.0
    for k, v in tensors:
        arr = to_np(v)
        if arr.size == 1:
            b = float(arr.reshape(-1)[0]); break

    print("[ckpt] parsed:",
          f"wH={None if wH is None else wH.shape}",
          f"wL={None if wL is None else wL.shape}",
          f"UH={None if UH is None else UH.shape}",
          f"UL={None if UL is None else UL.shape}",
          f"b={b}")
    if (wH is None) and (wL is None) and (UH is None) and (UL is None):
        raise RuntimeError("未在 ckpt 中匹配到任何可用权重（请检查 Dh/Dl 与 ckpt 是否一致）")
    return {"wH": wH, "wL": wL, "UH": UH, "UL": UL, "b": b}

# --------- 能量/ΔΔG 计算 ---------
def standardize(x, scaler):
    # x: (N,D)
    return scaler.transform(x)

def energy_from_parts(zH, zL, scalerH, scalerL, wH=None, wL=None, UH=None, UL=None, b=0.0, device="cpu"):
    """
    zH/zL: numpy (N, D)
    返回 numpy (N,)
    """
    e = np.zeros((zH.shape[0],), dtype=np.float32) + float(b)

    if scalerH is not None and zH is not None:
        zHn = standardize(zH, scalerH)  # (N,Dh)
    else:
        zHn = None
    if scalerL is not None and zL is not None:
        zLn = standardize(zL, scalerL)  # (N,Dl)
    else:
        zLn = None

    if (wH is not None) and (zHn is not None):
        e += (zHn @ wH.astype(np.float32))
    if (wL is not None) and (zLn is not None):
        e += (zLn @ wL.astype(np.float32))

    if (UH is not None) and (zHn is not None):
        t = zHn @ UH.astype(np.float32)           # (N,r)
        e += np.sum(t*t, axis=1)
    if (UL is not None) and (zLn is not None):
        t = zLn @ UL.astype(np.float32)
        e += np.sum(t*t, axis=1)

    return e

def ddg_from_energy(zH_wt, zL_wt, zH_mut, zL_mut, scalerH, scalerL, parts, device="cpu"):
    wH, wL, UH, UL, b = parts["wH"], parts["wL"], parts["UH"], parts["UL"], parts["b"]
    e_wt  = energy_from_parts(zH_wt, zL_wt, scalerH, scalerL, wH, wL, UH, UL, b, device=device)  # (1,)
    e_mut = energy_from_parts(zH_mut, zL_mut, scalerH, scalerL, wH, wL, UH, UL, b, device=device) # (N,)
    return e_mut - e_wt[0]

# --------- 校准（可选） ---------
def safe_load_calibrator(path):
    """
    避免 joblib 反序列化时因类路径不在当前 __main__ 报错。
    只要求对象有 predict(X) 接口。
    """
    obj = joblib.load(path)
    # 兜底：np.ndarray/常量 就当恒等
    if hasattr(obj, "predict"):
        return obj
    # 如果是 sklearn 的 IsotonicRegression，也有 predict
    return None

# --------- 主流程 ---------
def main():
    ap = argparse.ArgumentParser("Predict ΔΔG for single mutants with ESM-1b + energy model")
    ap.add_argument("--wt", required=True, help="WT FASTA 路径")
    ap.add_argument("--positions", required=True, help="突变位点，如 '22-148' 或 '22,35,50-60'")
    ap.add_argument("--alts", default="ALL", help="备选氨基酸集合，默认 ALL=20AA")
    ap.add_argument("--esm-ckpt", required=True, help="ESM-1b 本地权重 .pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--model", required=True, help="能量模型 ckpt（任意嵌套/扁平格式都可）")
    ap.add_argument("--scaler-h", required=True, help="StandardScaler H pkl")
    ap.add_argument("--scaler-l", required=True, help="StandardScaler L pkl")
    ap.add_argument("--calib", default=None, help="可选：校准器 pkl（不传则不校准）")

    ap.add_argument("--mut-dir", default="Sample_Mutant", help="变体与嵌入保存根目录")
    ap.add_argument("--out-dir", default="output", help="结果输出目录")
    ap.add_argument("--tag", default="VHH", help="样本 tag，用于归档")

    ap.add_argument("--topk", type=int, default=10, help="导出 Top-K 条目")
    ap.add_argument("--top-mode", choices=["neg","pos","abs"], default="neg",
                    help="Top-K 的排序模式：最负/最正/按绝对值")

    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # 1) 读 WT 并枚举所有突变
    wt = read_single_fasta(args.wt)
    positions = parse_positions(args.positions, len(wt))
    muts = make_all_mutants(wt, positions, args.alts)

    mut_root = ensure_dir(Path(args.mut_dir) / args.tag)
    out_root = ensure_dir(args.out_dir)
    emb_root = ensure_dir(mut_root / "emb")

    fasta_all = mut_root / "ALL_mutants.fasta"
    write_fasta(fasta_all, [(f"{p}_{w}>{a}", s) for (p,w,a,s) in muts])
    print(f"🧬 所有突变序列保存：{fasta_all}")

    # 2) ESM 嵌入
    print(f"🧠 使用本地 ESM-1b 权重：{args.esm_ckpt}")
    model, alphabet = load_esm1b_from_ckpt(Path(args.esm_ckpt))
    model = model.eval()

    wt_vec = esm_embed_mean(model, alphabet, [wt], device=device, batch_size=1)  # (1,1280)
    mut_vecs = esm_embed_mean(model, alphabet, [s for (_,_,_,s) in muts], device=device, batch_size=args.batch_size)

    # 保存嵌入
    np.save(emb_root / "WT.npy", wt_vec.astype(np.float32))
    np.save(emb_root / "ALL_mutants.npy", mut_vecs.astype(np.float32))

    # 3) 加载 scaler，确定 Dh/Dl
    scH = joblib.load(args.scaler_h)
    scL = joblib.load(args.scaler_l)
    Dh = int(getattr(scH, "n_features_in_", None) or len(scH.mean_))
    Dl = int(getattr(scL, "n_features_in_", None) or len(scL.mean_))

    if wt_vec.shape[1] != Dh:
        # 常见：ESM-1b 是 1280 维；Scaler 也是 1280 维
        print(f"[warn] WT 嵌入维度 {wt_vec.shape[1]} 与 scaler_H 维度 {Dh} 不一致。若确实不匹配会报错。")
    if mut_vecs.shape[1] != Dh:
        print(f"[warn] Mut 嵌入维度 {mut_vecs.shape[1]} 与 scaler_H 维度 {Dh} 不一致。")

    # VHH：没有 L 链，zL 置为 None 或 0 向量都可以；
    # 这里统一走接口，若模型里没有 wL/UL 会被自动跳过。
    zH_wt  = wt_vec
    zH_mut = mut_vecs
    zL_wt  = None
    zL_mut = None

    # 4) 加载能量模型（健壮解析）
    parts = load_energy_from_any_ckpt(args.model, Dh, Dl)
    # 转 numpy float32
    for k in ["wH","wL","UH","UL"]:
        if parts[k] is not None:
            parts[k] = parts[k].astype(np.float32)

    # 5) 预测 ΔΔG
    ddg = ddg_from_energy(zH_wt, zL_wt, zH_mut, zL_mut, scH, scL, parts, device=device)  # (N,)

    # 6) 可选校准
    ddg_cal = None
    if args.calib:
        try:
            cal_obj = safe_load_calibrator(args.calib)
            if cal_obj is not None:
                ddg_cal = cal_obj.predict(ddg.reshape(-1,1)).astype(np.float32)
                print(f"📐 已应用校准器：{args.calib}")
            else:
                print(f"[warn] 校准器 {args.calib} 不可用（无 predict），将跳过。")
        except Exception as e:
            print(f"[warn] 校准器加载失败：{e}，将跳过。")

    # 7) 写出 CSV
    out_csv = out_root / f"{args.tag}_ALL_ddg.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["idx","pos","wt","mut","pred_ddg"]
        if ddg_cal is not None:
            header += ["pred_ddg_cal"]
        w.writerow(header)
        for i, (p,w0,a,_) in enumerate(muts):
            row = [i, p, w0, a, float(ddg[i])]
            if ddg_cal is not None:
                row += [float(ddg_cal[i])]
            w.writerow(row)
    print(f"💾 结果保存：{out_csv}")

    # 8) Top-K
    ddg_for_rank = ddg_cal if ddg_cal is not None else ddg
    if args.top_mode == "neg":
        order = np.argsort(ddg_for_rank)
    elif args.top_mode == "pos":
        order = np.argsort(-ddg_for_rank)
    else:
        order = np.argsort(-np.abs(ddg_for_rank))
    top_idx = order[:args.topk]

    out_top = out_root / f"{args.tag}_topk.csv"
    with open(out_top, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank","pos","wt","mut","pred_ddg"] + (["pred_ddg_cal"] if ddg_cal is not None else []))
        for r, i in enumerate(top_idx, 1):
            p, wt_aa, mut_aa, _ = muts[i]
            row = [r, p, wt_aa, mut_aa, float(ddg[i])]
            if ddg_cal is not None:
                row += [float(ddg_cal[i])]
            w.writerow(row)
    print(f"🏅 Top-{args.topk} 保存：{out_top}（模式：{args.top_mode}）")

    # 9) 火山图（简单版：x=ΔΔG，y=|ΔΔG| 的 rank-based 近似）
    x = ddg_for_rank
    y = -np.log10((np.argsort(np.argsort(-np.abs(x)))+1) / (len(x)+1.0))  # 一个近似可视化
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.axvline(0, color="gray", lw=1)
    plt.xlabel("ΔΔG (model)")
    plt.ylabel("-log10(rank(|ΔΔG|))")
    plt.title(f"Volcano-like plot: {args.tag}")
    fig_path = out_root / f"{args.tag}_volcano.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"📈 火山图：{fig_path}")

if __name__ == "__main__":
    main()

