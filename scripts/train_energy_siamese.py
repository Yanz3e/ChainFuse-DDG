#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils ----------
def pick(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"缺列: {prefix}")
    return cols

def metrics(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return dict(
        spearman=float(spearmanr(y, yhat, nan_policy="omit").statistic),
        mae=float(mean_absolute_error(y, yhat)),
        rmse=float(np.sqrt(mean_squared_error(y, yhat))),
    )

# ---------- model ----------
class EnergyBilinear(nn.Module):
    """
    E(H, L) = wH·H + wL·L + sum_r <U_r, H> * <V_r, L>   （低秩双线性交互 + 线性项）
    """
    def __init__(self, dH, dL, rank=32):
        super().__init__()
        self.wH = nn.Linear(dH, 1, bias=False)
        self.wL = nn.Linear(dL, 1, bias=False)
        self.U  = nn.Linear(dH, rank, bias=False)
        self.V  = nn.Linear(dL, rank, bias=False)

        # 参数初始化更稳一些
        nn.init.zeros_(self.wH.weight)
        nn.init.zeros_(self.wL.weight)
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def energy(self, H, L):
        # H:[N,dH], L:[N,dL]
        main = self.wH(H) + self.wL(L)                 # [N,1]
        inter = (self.U(H) * self.V(L)).sum(dim=1, keepdim=True)  # [N,1]
        return main + inter

    def forward(self, Hm, Lm, Hw, Lw):
        # 预测 ddG = E(mut) - E(wt)
        return self.energy(Hm, Lm) - self.energy(Hw, Lw)

# ---------- training one fold ----------
def train_fold(Hm_tr, Lm_tr, Hw_tr, Lw_tr, y_tr,
               Hm_va, Lm_va, Hw_va, Lw_va, y_va,
               rank, lr, wd, epochs, patience, device, seed):
    torch.manual_seed(seed)
    dH, dL = Hm_tr.shape[1], Lm_tr.shape[1]
    model = EnergyBilinear(dH, dL, rank=rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    Xtr = [torch.from_numpy(a).float().to(device) for a in [Hm_tr, Lm_tr, Hw_tr, Lw_tr]]
    Xva = [torch.from_numpy(a).float().to(device) for a in [Hm_va, Lm_va, Hw_va, Lw_va]]
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)[:, None]
    yva = torch.from_numpy(y_va.astype(np.float32)).to(device)[:, None]

    best = np.inf; bad = 0; best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(*Xtr)
        loss = F.smooth_l1_loss(pred, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val = F.smooth_l1_loss(model(*Xva), yva).item()
        if val < best - 1e-6:
            best = val; bad = 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yhat_va = model(*Xva).squeeze(1).cpu().numpy()
    return model, yhat_va

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chain", type=str, default="both", choices=["H","L","both"],
                    help="按链分别训练或合并训练。默认 both=分别训练更稳")
    args = ap.parse_args()

    df = pd.read_csv("data/processed/pair_features.csv")
    for c in ["ddG","_pdb","_chain"]:
        assert c in df.columns, f"缺少列: {c}"

        # ===== 新增：确保有 OOF 目录 =====
    out_oof_dir = Path("OOF")
    out_oof_dir.mkdir(parents=True, exist_ok=True)

    zH = pick(df, "zH_"); zL = pick(df, "zL_")
    wH = pick(df, "wH_"); wL = pick(df, "wL_")

    # 复原“突变后绝对表征” mH/mL
    mH = (df[wH].values + df[zH].values).astype(np.float32)
    mL = (df[wL].values + df[zL].values).astype(np.float32)
    wH_arr = df[wH].values.astype(np.float32)
    wL_arr = df[wL].values.astype(np.float32)
    y = df["ddG"].values.astype(np.float32)
    g = df["_pdb"].values
    cflag = df["_chain"].str.upper().values  # 'H' or 'L'

    def run_one(split_flag):
        sub = (cflag == split_flag) if split_flag in ["H","L"] else np.ones(len(df), dtype=bool)
        Hm = mH[sub]; Lm = mL[sub]; Hw = wH_arr[sub]; Lw = wL_arr[sub]
        yy = y[sub]; gg = g[sub]

        # 每折内 scaler：在 WT+MUT 的联合分布上拟合，保证差分发生在输出不在尺度
        gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(gg)))))
        oof = np.zeros_like(yy, dtype=np.float32)

        for fold, (tr, va) in enumerate(gkf.split(Hm, yy, gg), 1):
            scH = StandardScaler().fit(np.vstack([Hm[tr], Hw[tr]]))
            scL = StandardScaler().fit(np.vstack([Lm[tr], Lw[tr]]))

            Hm_tr, Hw_tr = scH.transform(Hm[tr]), scH.transform(Hw[tr])
            Lm_tr, Lw_tr = scL.transform(Lm[tr]), scL.transform(Lw[tr])
            Hm_va, Hw_va = scH.transform(Hm[va]), scH.transform(Hw[va])
            Lm_va, Lw_va = scL.transform(Lm[va]), scL.transform(Lw[va])

            _, yhat_va = train_fold(
                Hm_tr, Lm_tr, Hw_tr, Lw_tr, yy[tr],
                Hm_va, Lm_va, Hw_va, Lw_va, yy[va],
                rank=args.rank, lr=args.lr, wd=args.wd,
                epochs=args.epochs, patience=args.patience,
                device=args.device, seed=args.seed + fold
            )
            oof[va] = yhat_va

        rep = metrics(yy, oof)

        idx_sub = np.flatnonzero(sub)  # 映射回原 df 的行号以便融合对齐
        oof_path = out_oof_dir / f"oof_energy_siamese_{split_flag}_rank{args.rank}_seed{args.seed}.csv"
        pd.DataFrame({"idx": idx_sub, "oof": oof}).to_csv(oof_path, index=False)
        print(f"💾 OOF 已保存到: {oof_path}")
        
        return rep

    if args.chain == "both":
        repH = run_one("H"); repL = run_one("L")
        out = {"H": repH, "L": repL}
    elif args.chain == "H":
        out = {"H": run_one("H")}
    else:
        out = {"L": run_one("L")}

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/metrics_energy_siamese.json","w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("✅ 训练完成：", out)

if __name__ == "__main__":
    main()
