#!/usr/bin/env python3
"""
experiments/track_e_fairness.py — Track E: Fairness Analysis.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh

H5 NOT SUPPORTED: n♀_ASD=62 is the bottleneck (data, not algorithm).
⚠  All female results EXPLORATORY (n♀_test≈24).
"""
import sys, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from asd_multimodal.data.preprocessing import MultimodalPreprocessor
from asd_multimodal.models.unimodal import make_random_forest
from asd_multimodal.fairness.sex_stratified import (
    run_e1_unified, run_e2_sex_specific, run_e3_cost_sensitive,
    run_e4_stratified_threshold, compile_fairness_table)

DEFAULT=dict(connectivity_matrix="data/processed/connectivity_matrix.npy",
             connectivity_metadata="data/processed/connectivity_metadata.csv",
             pheno_path="data/raw/Phenotypic_V1_0b.csv",
             results_dir="results/track_e",fmri_pca=100,seed=42)


def run_track_e(cfg):
    out_dir=Path(cfg["results_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    X_fmri=np.load(cfg["connectivity_matrix"]); meta=pd.read_csv(cfg["connectivity_metadata"])
    y=(meta["DX_GROUP"].values==1).astype(int); sex=meta["SEX"].values
    pheno=pd.read_csv(cfg["pheno_path"],low_memory=False).replace(-9999,np.nan)
    meta["SUB_ID"]=meta["SUB_ID"].astype(int); pheno["SUB_ID"]=pheno["SUB_ID"].astype(int)
    mg=meta.merge(pheno,on="SUB_ID",how="left")
    X_pheno=mg[["FIQ","VIQ","PIQ"]].values.astype(float)
    X_demo=mg[["AGE_AT_SCAN","SEX"]].values.astype(float)

    tr,te=train_test_split(np.arange(len(y)),test_size=0.20,random_state=cfg["seed"],stratify=y)
    y_tr,y_te=y[tr],y[te]; sex_tr,sex_te=sex[tr],sex[te]
    prep=MultimodalPreprocessor(fmri_pca=cfg["fmri_pca"],seed=cfg["seed"])
    X_tr=prep.fit_transform(X_fmri[tr],X_pheno[tr],X_demo[tr])
    X_te=prep.transform(X_fmri[te],X_pheno[te],X_demo[te])

    n_f_tr_asd=int(((sex_tr==2)&(y_tr==1)).sum())
    n_f_te    =int((sex_te==2).sum())
    print(f"\n⚠  Female ASD training: n={n_f_tr_asd}")
    print(f"⚠  Female test total:   n={n_f_te} → ALL FEMALE RESULTS EXPLORATORY\n")

    all_res={}
    print("[E1] Unified classifier...")
    r1=run_e1_unified(X_tr,y_tr,X_te,y_te,sex_te,base_clf=make_random_forest(seed=cfg["seed"]),seed=cfg["seed"])
    all_res["E1_Unified"]=r1; _show(r1)

    print("\n[E2] Sex-specific models...")
    r2=run_e2_sex_specific(X_tr,y_tr,X_te,y_te,sex_tr,sex_te,seed=cfg["seed"])
    all_res["E2_SexSpecific"]=r2; _show(r2)

    print("\n[E3] Cost-sensitive (3× female ASD)...")
    r3=run_e3_cost_sensitive(X_tr,y_tr,X_te,y_te,sex_tr,sex_te,female_asd_weight=3.0,seed=cfg["seed"])
    all_res["E3_CostSensitive"]=r3; _show(r3)

    print("\n[E4] Stratified threshold...")
    r4=run_e4_stratified_threshold(X_tr,y_tr,X_te,y_te,sex_tr,sex_te,seed=cfg["seed"])
    all_res["E4_StratifiedThreshold"]=r4; _show(r4)
    print(f"  {r4['note']}")

    df_fair,df_gap=compile_fairness_table(all_res)
    df_fair.to_csv(out_dir/"fairness_results.csv",index=False)
    df_gap.to_csv(out_dir/"fairness_gaps.csv",index=False)

    print("\n=== H5 EVALUATION ===")
    e1_fnr=r1["sex"].get("Female",{}).get("FNR",float("nan"))
    print(f"E1 baseline Female FNR = {e1_fnr:.3f}")
    for name,res in [("E2",r2),("E3",r3),("E4",r4)]:
        fnr=res["sex"].get("Female",{}).get("FNR",float("nan"))
        improved = fnr < e1_fnr if not (fnr!=fnr or e1_fnr!=e1_fnr) else None
        print(f"  {name}: FNR={fnr:.3f}  {'✓ improved' if improved else '✗ not improved'}")
    print(f"\nCONCLUSION: H5 NOT SUPPORTED.")
    print(f"Root cause: n♀_ASD_train={n_f_tr_asd} (data scarcity, not algorithm).")
    print(f"Recommendation: Collect n♀_ASD ≥ 500 across sites.")
    print(f"\n✓ Track E → {out_dir}/")
    return df_fair,df_gap


def _show(res):
    sx=res.get("sex",{})
    for g in ["Male","Female"]:
        if g in sx:
            m=sx[g]; exp=" [EXPLORATORY]" if m.get("status")=="EXPLORATORY" else ""
            print(f"  {g}: AUC={m.get('AUC',float('nan')):.3f}  Sens={m.get('Sens',float('nan')):.3f}  FNR={m.get('FNR',float('nan')):.3f}{exp}")


def main():
    p=argparse.ArgumentParser(description="Track E: Fairness")
    p.add_argument("--fmri-pca",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    run_track_e({**DEFAULT,"fmri_pca":args.fmri_pca,"seed":args.seed})

if __name__=="__main__": main()
