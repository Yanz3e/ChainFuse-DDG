#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit a calibrator for ΔΔG regression from OOF predictions.
- Supports isotonic (default) and linear calibrations.
- Auto-detect monotonic direction; flips x when needed.
- Saves a pickle with a .transform()/.__call__() API.

Usage examples:
  # H-chain, isotonic
  python scripts/fit_ddg_calibrator.py \
    --oof OOF/H_oof.csv \
    --y-col ddg_true --pred-col ddg_pred \
    --out-pkl models/calib_H_iso.pkl \
    --plot OOF/calib_H_iso.png

  # L-chain, linear
  python scripts/fit_ddg_calibrator.py \
    --oof OOF/L_oof.csv \
    --y-col ddg_true --pred-col ddg_pred \
    --method linear \
    --out-pkl models/calib_L_lin.pkl \
    --plot OOF/calib_L_lin.png
"""

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pickle
from typing import Optional, Tuple

from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------- Calibrator wrapper ----------------------

class DDGCalibrator:
    """
    Wrapper exposing a stable .transform(x) for both isotonic and linear.
    Handles flipping x if the monotonic direction is negative.
    """
    def __init__(self, kind: str, model, flip_x: bool, x_min: float, x_max: float):
        self.kind = kind              # "isotonic" or "linear"
        self.model = model
        self.flip_x = bool(flip_x)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

    def _prep_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        # optional clip to training domain to avoid insane extrapolation
        x = np.clip(x, self.x_min, self.x_max)
        if self.flip_x:
            x = -x
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        xv = self._prep_x(x)
        if self.kind == "isotonic":
            y = self.model.predict(xv)
        elif self.kind == "linear":
            y = self.model.predict(xv.reshape(-1, 1)).reshape(-1)
        else:
            raise ValueError(f"Unknown calibrator kind: {self.kind}")
        return y.astype(np.float32)

    __call__ = transform


# ---------------------- Utils ----------------------

def load_oof(paths, y_col: str, pred_col: str) -> pd.DataFrame:
    dfs = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            raise FileNotFoundError(f"OOF file not found: {pth}")
        df = pd.read_csv(pth)
        if y_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"Columns missing in {pth}: need '{y_col}' and '{pred_col}'. "
                             f"Have: {list(df.columns)}")
        dfs.append(df[[y_col, pred_col]].dropna())
    out = pd.concat(dfs, axis=0, ignore_index=True)
    if len(out) == 0:
        raise ValueError("Empty OOF after dropna.")
    return out.rename(columns={y_col: "y", pred_col: "p"})


def metrics(y, p, prefix=""):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    sp = spearmanr(y, p).correlation
    pr = pearsonr(y, p)[0]
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {
        f"{prefix}spearman": float(sp),
        f"{prefix}pearson": float(pr),
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
    }


def fit_isotonic(y, p) -> Tuple[DDGCalibrator, np.ndarray]:
    # decide direction by Spearman sign
    sp = spearmanr(y, p).correlation
    flip_x = False
    x = p.copy()
    if np.isnan(sp):
        # degenerate; force increasing
        sp = 0.0
    if sp < 0:
        x = -x
        flip_x = True
    # isotonic with safe clipping outside train domain
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(x, y)
    cal = DDGCalibrator(kind="isotonic", model=ir, flip_x=flip_x, x_min=float(np.min(p)), x_max=float(np.max(p)))
    yhat = cal.transform(p)
    return cal, yhat


def fit_linear(y, p) -> Tuple[DDGCalibrator, np.ndarray]:
    # linear regression; no flipping, the slope can be negative as needed
    lr = LinearRegression()
    lr.fit(p.reshape(-1, 1), y.reshape(-1, 1))
    cal = DDGCalibrator(kind="linear", model=lr, flip_x=False, x_min=float(np.min(p)), x_max=float(np.max(p)))
    yhat = cal.transform(p)
    return cal, yhat


def make_plot(y, p, ycal, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6,6))
    # raw
    ax.scatter(p, y, s=10, alpha=0.35, label="raw", edgecolors="none")
    # calibrated
    ax.scatter(ycal, y, s=10, alpha=0.35, label="calibrated", edgecolors="none")
    lims = np.array([np.min([p.min(), y.min(), ycal.min()]) - 0.5,
                     np.max([p.max(), y.max(), ycal.max()]) + 0.5])
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("prediction / calibrated")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser("Fit ΔΔG calibrator from OOF.")
    ap.add_argument("--oof", nargs="+", required=True, help="OOF csv paths (one or more)")
    ap.add_argument("--y-col", default="ddg_true", help="ground-truth column name")
    ap.add_argument("--pred-col", default="ddg_pred", help="prediction column name")
    ap.add_argument("--method", choices=["isotonic", "linear"], default="isotonic")
    ap.add_argument("--out-pkl", required=True, help="output pickle path for the calibrator")
    ap.add_argument("--plot", default="", help="optional: save a comparison scatter png")
    args = ap.parse_args()

    df = load_oof(args.oof, y_col=args.y_col, pred_col=args.pred_col)
    y = df["y"].to_numpy(dtype=np.float64)
    p = df["p"].to_numpy(dtype=np.float64)

    pre = metrics(y, p, prefix="pre_")
    print("[pre ]", pre)

    if args.method == "isotonic":
        cal, yhat = fit_isotonic(y, p)
        method_name = "isotonic"
    else:
        cal, yhat = fit_linear(y, p)
        method_name = "linear"

    post = metrics(y, yhat, prefix="post_")
    print("[post]", post)

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(cal, f)
    print(f"[save] calibrator -> {out_pkl}")

    if args.plot:
        out_png = Path(args.plot)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        title = f"Calibration ({method_name})"
        make_plot(y, p, yhat, out_png, title=title)
        print(f"[plot] saved -> {out_png}")


if __name__ == "__main__":
    main()
