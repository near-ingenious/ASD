#!/usr/bin/env python3
"""
experiments/track_c_missing.py — Track C: Missing-Modality Robustness.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from asd_multimodal.data.preprocessing import MultimodalPreprocessor
from asd_multimodal.models.unimodal import make_random_forest
from asd_multimodal.training.missing_modality import (
    run_missing_modality_experiment, summarise_track_c, SCENARIOS, STRATEGIES)

DEFAULT=dict(connectivity_matrix="data/processed/connectivity_matrix.npy",
             connectivity_metadata="data/processed/connectivity_metadata.csv",
             pheno_path="data/raw/Phenotypic_V1_0b.csv",
             results_dir="results/track_c",fmri_pca=100,
             test_size=0.20,train_vae=True,train_mae=False,n_boot=500,seed=42)


def run_track_c(cfg):
    out_dir=Path(cfg["results_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    X_fmri=np.load(cfg["connectivity_matrix"]); meta=pd.read_csv(cfg["connectivity_metadata"])
    y=( meta["DX_GROUP"].values==1).astype(int)
    pheno=pd.read_csv(cfg["pheno_path"],low_memory=False).replace(-9999,np.nan)
    meta["SUB_ID"]=meta["SUB_ID"].astype(int); pheno["SUB_ID"]=pheno["SUB_ID"].astype(int)
    mg=meta.merge(pheno,on="SUB_ID",how="left")
    X_pheno=mg[["FIQ","VIQ","PIQ"]].values.astype(float)
    X_demo=mg[["AGE_AT_SCAN","SEX"]].values.astype(float)
    prep=MultimodalPreprocessor(fmri_pca=cfg["fmri_pca"],seed=cfg["seed"])
    X_full=prep.fit_transform(X_fmri,X_pheno,X_demo)
    print(f"Full feature matrix: {X_full.shape}")
    df=run_missing_modality_experiment(
        lambda:make_random_forest(seed=cfg["seed"]),X_full,y,
        test_size=cfg["test_size"],train_vae=cfg["train_vae"],
        train_mae=cfg["train_mae"],n_boot=cfg["n_boot"],seed=cfg["seed"])
    df.to_csv(out_dir/"track_c_full_results.csv",index=False)
    summary=summarise_track_c(df); summary.to_csv(out_dir/"track_c_summary.csv",index=False)
    print("\n=== TRACK C SUMMARY ===")
    print(summary[["Scenario","Description","Strategy","AUC","Delta_AUC","Pct_Retained","Degradation"]].to_string(index=False))
    print(f"\n✓ Track C → {out_dir}/")
    return df


def main():
    p=argparse.ArgumentParser(description="Track C: Missing-Modality Robustness")
    p.add_argument("--scenarios",nargs="+",default=None,choices=list(SCENARIOS.keys()))
    p.add_argument("--strategies",nargs="+",default=None,choices=STRATEGIES)
    p.add_argument("--no-vae",action="store_true"); p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    cfg={**DEFAULT,"seed":args.seed,"train_vae":not args.no_vae}
    run_track_c(cfg)

if __name__=="__main__": main()
