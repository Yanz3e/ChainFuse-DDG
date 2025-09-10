import re, os, json, yaml, numpy as np, pandas as pd
from typing import Tuple, Dict

AA_3to1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"
}

def load_cfg(path:str)->dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def parse_mut_str(s:str):
    """
    支持 'H:Y33F' 或 'A:W100aG'；返回 (chain, reskey 'resseq|icode', newAA)
    """
    s = str(s).strip()
    m = re.match(r"^([A-Za-z0-9])\s*:\s*([A-Za-z])(\d+)([A-Za-z]?)([A-Za-z])$", s)
    if not m:
        raise ValueError(f"无法解析突变格式: {s}")
    chain = m.group(1).upper()
    resseq = int(m.group(3))
    icode = m.group(4).upper() if m.group(4) else ""
    newaa = m.group(5).upper()
    return chain, f"{resseq}|{icode}", newaa

def normalize_labels(df: pd.DataFrame)->pd.DataFrame:
    """
    允许两种输入：
    1) '#PDB','Mutation','ddG(kcal/mol)'
    2) '_pdb','_chain','_mutation','ddG'
    """
    cols = set(df.columns)
    if {"#PDB","Mutation"}.issubset(cols):
        out = pd.DataFrame({
            "#PDB": df["#PDB"].astype(str).str.upper().str.strip(),
            "Mutation": df["Mutation"].astype(str).str.strip(),
            "ddG": pd.to_numeric(df.get("ddG(kcal/mol)", df.get("ddG")), errors="coerce")
        })
        return out.dropna(subset=["#PDB","Mutation"])
    elif {"_pdb","_chain","_mutation"}.issubset(cols):
        out = pd.DataFrame({
            "#PDB": df["_pdb"].astype(str).str.upper().str.strip(),
            "Mutation": df["_chain"].astype(str).str.upper().str.strip() + ":" + df["_mutation"].astype(str).str.strip(),
            "ddG": pd.to_numeric(df.get("ddG"), errors="coerce")
        })
        return out.dropna(subset=["#PDB","Mutation"])
    else:
        raise KeyError("labels.csv 需包含 (#PDB,Mutation[,ddG(kcal/mol)]) 或 (_pdb,_chain,_mutation[,ddG])")

def read_map_json(path:str)->Dict[str,int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_point_mutation(seq:str, idx:int, new_aa:str)->str:
    assert 0 <= idx < len(seq), (idx, len(seq))
    return seq[:idx] + new_aa.upper() + seq[idx+1:]
