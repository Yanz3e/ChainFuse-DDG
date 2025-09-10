#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_full_energy_siamese.py
在整份训练集（按链筛选）上拟合最终能量模型，并保存：
- models/energy_{CHAIN}_rank{RANK}_full.pt
- models/scaler_H.pkl / models/scaler_L.pkl（以及兼容命名副本）
- models/meta_{CHAIN}.json（包含维度、超参、训练轮次等）
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import joblib


# ------------------------- utils -------------------------
def pick(df: pd.DataFrame, prefix: str):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"缺少列: {prefix}")
    return cols


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return dict(
        spearman=float(spearmanr(y_true, y_pred, nan_policy="omit").statistic),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# ------------------------- model -------------------------
class EnergyBilinear(nn.Module):
    """
    低秩双线性能量：
      E(H, L) = wH·H + wL·L + sum_r <U_r, H> * <V_r, L>
    预测：ddG = E(mut) - E(wt)
    """
    def __init__(self, dH: int, dL: int, rank: int = 32):
        super().__init__()
        self.wH = nn.Linear(dH, 1, bias=False)
        self.wL = nn.Linear(dL, 1, bias=False)
        self.U  = nn.Linear(dH, rank, bias=False)
        self.V  = nn.Linear(dL, rank, bias=False)

        # 稳定初始化
        nn.init.zeros_(self.wH.weight)
        nn.init.zeros_(self.wL.weight)
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def energy(self, H, L):
        main = self.wH(H) + self.wL(L)                             # [N,1]
        inter = (self.U(H) * self.V(L)).sum(dim=1, keepdim=True)   # [N,1]
        return main + inter

    def forward(self, Hm, Lm, Hw, Lw):
        return self.energy(Hm, Lm) - self.energy(Hw, Lw)


# ------------------------- training -------------------------
def fit_full(Hm, Lm, Hw, Lw, y,
             rank=96, lr=5e-3, wd=1e-4, epochs=1500, patience=150, device="cuda", seed=27):
    torch.manual_seed(seed)
    dH, dL = Hm.shape[1], Lm.shape[1]
    model = EnergyBilinear(dH, dL, rank=rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    X = [torch.from_numpy(a).float().to(device) for a in [Hm, Lm, Hw, Lw]]
    Y = torch.from_numpy(y.astype(np.float32)).to(device)[:, None]

    best = np.inf
    best_state = None
    bad = 0
    last_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(*X)
        loss = F.smooth_l1_loss(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        val = loss.item()  # 使用训练损失作为早停指标（全量拟合，不切验证）
        if val < best - 1e-6:
            best = val
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                last_epoch = ep
                break
    if last_epoch == 0:
        last_epoch = epochs

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        yhat = model(*X).detach().cpu().numpy().ravel()

    rep = metrics(y, yhat)
    rep.update(dict(best_train_loss=float(best), epochs_run=int(last_epoch)))
    return model, rep


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=str, default="H", choices=["H", "L"], help="选择训练链（默认 H）")
    ap.add_argument("--rank", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--csv", type=str, default="data/processed/pair_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ["ddG", "_pdb", "_chain"]:
        assert c in df.columns, f"缺少列: {c}"

    # 取列
    zH = pick(df, "zH_"); zL = pick(df, "zL_")
    wH = pick(df, "wH_"); wL = pick(df, "wL_")

    # 复原突变后绝对表征（mut = wt + delta）
    mH = (df[wH].values + df[zH].values).astype(np.float32)
    mL = (df[wL].values + df[zL].values).astype(np.float32)
    wH_arr = df[wH].values.astype(np.float32)
    wL_arr = df[wL].values.astype(np.float32)
    y = df["ddG"].values.astype(np.float32)
    cflag = df["_chain"].str.upper().values

    # 筛选链
    sub = (cflag == args.chain)
    mH, mL, wH_arr, wL_arr, y = mH[sub], mL[sub], wH_arr[sub], wL_arr[sub], y[sub]

    # 标准化：在 WT+MUT 的联合分布上拟合
    scH = StandardScaler().fit(np.vstack([mH, wH_arr]))
    scL = StandardScaler().fit(np.vstack([mL, wL_arr]))
    Hm = scH.transform(mH); Hw = scH.transform(wH_arr)
    Lm = scL.transform(mL); Lw = scL.transform(wL_arr)

    # 拟合
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, rep = fit_full(Hm, Lm, Hw, Lw, y,
                          rank=args.rank, lr=args.lr, wd=args.wd,
                          epochs=args.epochs, patience=args.patience,
                          device=device, seed=args.seed)

    # 保存
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = Path(f"models/energy_{args.chain}_rank{args.rank}_full.pt")
    torch.save(model.state_dict(), model_path)

    # 保存 scaler（主名）以及若干兼容别名
    scH_path = Path("models/scaler_H.pkl")
    scL_path = Path("models/scaler_L.pkl")
    joblib.dump(scH, scH_path)
    joblib.dump(scL, scL_path)
    # 兼容命名（若你后面脚本用到了不同名字，这里一并写出去）
    joblib.dump(scH, Path("models/scaler_H_w.pkl"))
    joblib.dump(scH, Path("models/scaler_H_m.pkl"))
    joblib.dump(scL, Path("models/scaler_L_w.pkl"))
    joblib.dump(scL, Path("models/scaler_L_m.pkl"))

    meta = dict(
        time=datetime.now().isoformat(timespec="seconds"),
        chain=args.chain,
        rank=args.rank,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=device,
        dims=dict(dH=int(Hm.shape[1]), dL=int(Lm.shape[1])),
        paths=dict(
            model=str(model_path.as_posix()),
            scaler_H=str(scH_path.as_posix()),
            scaler_L=str(scL_path.as_posix()),
        ),
        train_report=rep,
        feature_prefix=dict(wH="wH_", zH="zH_", wL="wL_", zL="zL_"),
        data_csv=args.csv,
        n_samples=int(len(y)),
    )
    with open(f"models/meta_{args.chain}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 全量训练完成：", {args.chain: rep})
    print("📦 保存：", model_path, scH_path, scL_path)


if __name__ == "__main__":
    main()
