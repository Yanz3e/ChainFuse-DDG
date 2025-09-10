#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
import numpy as np, pandas as pd, torch, esm
from pathlib import Path
from typing import List, Dict, Any
from utils import (
    load_cfg, ensure_dir, normalize_labels, parse_mut_str,
    read_map_json, apply_point_mutation
)

def mean_pool(rep: torch.Tensor, seq_len: int) -> torch.Tensor:
    # rep: [1+L+1, C]
    return rep[1:1+seq_len].mean(0)

def weighted_pool(rep: torch.Tensor, weights) -> torch.Tensor:
    """
    加权池化，稳定版：
      - weights 长度为 L；会自动归一化
      - 若权重异常或和为 0，退回 mean
    """
    x = rep[1:1+len(weights)]
    try:
        w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
        s = w.sum()
        if float(s) <= 0 or w.numel() != x.shape[0]:
            return x.mean(0)
        w = w / s
        return (x * w[:, None]).sum(0)
    except Exception:
        return x.mean(0)

def build_cdr_mask_from_index_map(index_map: Dict[str, int], seq_len: int, chain_type: str):
    """
    稳定版 CDR 掩码：
      - 默认返回全 1（与 mean 相同，不炸）
      - 能解析到粗略 IMGT 号段时：CDR=1.0，FR=0.2
      - 后续你接 ANARCI/IMGT 精确号段时，替换本函数即可
    index_map: 'resseq|icode' → 序列索引
    """
    try:
        import re as _re
        # 先假定全 1，确保稳定
        mask = [1.0] * seq_len

        # 粗略号段（保守默认）
        if str(chain_type).upper().startswith("H"):
            ranges = [(26, 32), (52, 56), (95, 102)]
        else:
            # L 链粗略范围
            ranges = [(24, 34), (50, 56), (89, 97)]

        # resseq→idx
        r2i = {}
        for k, v in index_map.items():
            m = _re.match(r"^(\d+)\|", k)
            if m:
                r2i[int(m.group(1))] = int(v)

        # 先给 FR=0.2
        mask = [0.2] * seq_len
        for a, b in ranges:
            for r in range(a, b + 1):
                if r in r2i and 0 <= r2i[r] < seq_len:
                    mask[r2i[r]] = 1.0
        return mask
    except Exception:
        # 任意异常都走全 1，绝不影响流程
        return [1.0] * seq_len

def parse_multi_same_chain(mut_str: str):
    """
    支持同链多点写法：
      - "H:Y33F+W52A"
      - "H:Y33F;H:W52A"
      - "H:Y33F, H:W52A"
      - "D:L483T,D:V486P,..."
      - 单点 "H:Y33F" 仍然可用
    只要都在同一条链上就行；跨链请拆两行。
    返回: chain, [(reskey,newAA), ...], tag
    """
    s = str(mut_str).strip()
    if not s:
        return None, [], ""
    if '+' not in s and ';' not in s and ',' not in s:
        c, reskey, naa = parse_mut_str(s)
        return c, [(reskey, naa)], f"{c}_{s.split(':',1)[1]}".replace(':','_')

    tokens = re.split(r'[;,]\s*', s)
    chain_set = set(); per_chain = {}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r'^([A-Za-z0-9])\s*:\s*(.+)$', tok)
        if m:
            c = m.group(1).upper()
            rest = m.group(2).strip()
            sites = re.split(r'\s*\+\s*', rest)
            for site in sites:
                mm = re.match(r'^([A-Za-z])(\d+)([A-Za-z]?)([A-Za-z])$', site.strip())
                if not mm:
                    return None, [], ""
                resseq = int(mm.group(2))
                icode  = mm.group(3).upper() if mm.group(3) else ""
                naa    = mm.group(4).upper()
                per_chain.setdefault(c, []).append((f"{resseq}|{icode}", naa))
            chain_set.add(c)
        else:
            try:
                c1, rk1, na1 = parse_mut_str(tok)
                chain_set.add(c1)
                per_chain.setdefault(c1, []).append((rk1, na1))
            except Exception:
                return None, [], ""

    if len(chain_set) != 1:
        return None, [], ""  # 多链请拆行

    chain = next(iter(chain_set))
    sites_str = "+".join([rk.split("|",1)[0] + naa for rk, naa in per_chain[chain]])
    tag = f"{chain}_{sites_str}"
    return chain, per_chain[chain], tag

def main():
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cdr"],
                        help="特征池化方式：mean(默认) 或 cdr(加权)")
    parser.add_argument("--layers", type=str, default="33",
                        help="取哪些层，逗号分隔，例如 33 或 31,33")
    parser.add_argument("--add-cls", action="store_true",
                        help="是否将 CLS 向量拼接到序列池化向量之后")
    args = parser.parse_args()

    cfg = load_cfg("configs/paths.yaml")
    fasta = Path(cfg["wt_fasta"])
    map_dir = Path(cfg["map_dir"])
    out_dir = Path(cfg["embeddings_dir"]); ensure_dir(out_dir.as_posix())
    labels = normalize_labels(pd.read_csv(cfg["labels_csv"], encoding="latin1"))

    # 读 WT 序列
    wt = {}; name = None
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip(); wt[name] = ""
        else:
            wt[name] += line.strip()

    # 模型
    weights = Path(cfg["esm1b_weights"]).resolve()
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(weights.as_posix())
    model.eval()
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_fp16 = bool(cfg.get("fp16", True))
    if use_fp16:
        model.half()
    batch_converter = alphabet.get_batch_converter()
    layer_ids = [int(x) for x in str(args.layers).split(",") if str(x).strip()]

    # 先缓存 WT 向量
    for name, seq in wt.items():
        out_path = out_dir / f"{name}__WT.npz"
        if out_path.is_file():
            continue
        toks = batch_converter([(name, seq)])[2].to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16), torch.no_grad():
            outs = model(toks, repr_layers=layer_ids, return_contacts=False)["representations"]
        rep = sum(outs[L][0] for L in layer_ids) / len(layer_ids)

        # 默认 mean；cdr 需要 index_map 和链类型
        if args.pool == "cdr":
            try:
                pdb_id, chain_type = name.split("_", 1)
                idx_map = read_map_json((map_dir / f"{pdb_id}_{chain_type}.json").as_posix())
                weights = build_cdr_mask_from_index_map(idx_map, len(seq), chain_type)
                seq_vec = weighted_pool(rep, weights)
            except Exception:
                seq_vec = mean_pool(rep, len(seq))
        else:
            seq_vec = mean_pool(rep, len(seq))

        if args.add_cls:
            cls_vec = rep[0]
            emb_t = torch.cat([seq_vec, cls_vec], dim=0)
        else:
            emb_t = seq_vec
        emb = emb_t.float().cpu().numpy()
        np.savez_compressed(out_path.as_posix(), emb=emb.astype(np.float32))

    # 突变体（同链多点 OK，跨链请拆行）
    bsz = int(cfg.get("batch_size", 8))
    jobs = []
    for pdb, mut_str in zip(labels["#PDB"], labels["Mutation"]):
        chain, sites, tag = parse_multi_same_chain(mut_str)
        if not chain:
            continue
        name = f"{pdb}_{chain}"
        if name not in wt:
            continue
        mapp = map_dir / f"{pdb}_{chain}.json"
        if not mapp.is_file():
            continue
        idx_map = read_map_json(mapp.as_posix())
        mut_seq = wt[name]
        ok = True
        for reskey, naa in sites:
            if reskey not in idx_map:
                ok = False; break
            mut_seq = apply_point_mutation(mut_seq, idx_map[reskey], naa)
        if not ok:
            continue
        out_path = out_dir / f"{name}__{tag}.npz"
        if out_path.is_file():
            continue
        jobs.append((f"{name}__{tag}", mut_seq, out_path, chain))

    for i in range(0, len(jobs), bsz):
        batch = jobs[i:i+bsz]
        names = [j[0] for j in batch]
        seqs  = [j[1] for j in batch]
        toks = batch_converter(list(zip(names, seqs)))[2].to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16), torch.no_grad():
            outs = model(toks, repr_layers=layer_ids, return_contacts=False)["representations"]
        # 每条序列独立处理
        for k in range(len(batch)):
            rep = sum(outs[L][k] for L in layer_ids) / len(layer_ids)
            name = names[k]
            seq  = seqs[k]
            chain_type = name.split("_", 1)[1].split("__", 1)[0].split("_", 1)[0]  # "H"/"L" 等

            if args.pool == "cdr":
                try:
                    pdb_id = name.split("_", 1)[0]
                    idx_map = read_map_json((map_dir / f"{pdb_id}_{chain_type}.json").as_posix())
                    weights = build_cdr_mask_from_index_map(idx_map, len(seq), chain_type)
                    seq_vec = weighted_pool(rep, weights)
                except Exception:
                    seq_vec = mean_pool(rep, len(seq))
            else:
                seq_vec = mean_pool(rep, len(seq))

            if args.add_cls:
                cls_vec = rep[0]
                emb_t = torch.cat([seq_vec, cls_vec], dim=0)
            else:
                emb_t = seq_vec

            emb = emb_t.float().cpu().numpy()
            np.savez_compressed(batch[k][2].as_posix(), emb=emb.astype(np.float32))

    print(f"✅ 生成完成：嵌入已写入 {out_dir.resolve()}")

if __name__ == "__main__":
    main()
