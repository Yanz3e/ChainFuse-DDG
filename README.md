<!-- README.md -->

# Antibody-Lite: ΔΔG Prediction with ESM-1b + Energy (Siamese) Model

This repository provides a lightweight, end-to-end workflow to:
1) build features from ESM-1b embeddings,  
2) train a Siamese energy model with out-of-fold (OOF) validation,  
3) optionally fit an isotonic **ΔΔG calibrator**, and  
4) predict ΔΔG for **all single-point mutants** of a given antibody WT sequence (e.g., VHH).

> Minimal, reproducible, GPU-friendly. Keeps heavy/light chain separated (H/L). VHH uses **H-only**.

---

## 1) Environment

```bash
conda env create -f environment.yml
conda activate antibody-lite

# optional check
python - <<'PY'
import torch; print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
PY
