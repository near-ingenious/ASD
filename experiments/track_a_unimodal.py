#!/usr/bin/env python3
"""
experiments/track_a_unimodal.py — Track A: Unimodal Baseline Models.

Authors:    Jarin Alam Prity (222-115-005)  jarinprity438@gmail.com
            Popy Rani Boidya (007)           popyboidya@gmail.com
Supervisor: Md Mahfujul Hasan — Metropolitan University, Sylhet
Clinical:   Prof. Imdadul Magfur — Sylhet MAG Osmani Medical College
"""
import sys, os, argparse, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from asd_multimodal.models.unimodal import (
    make_random_forest, make_xgboost, make_lightgbm, make_mlp,
    TabTransformer, BrainGNN, GraphTransformer, SklearnWrapper)
from asd_multimodal.training.cv_engine import run_cv, format_results
from asd_multimodal.utils.metrics import format_ci

DEFAULT = dict(
    connectivity_matrix   = "data/processed/connectivity_matrix.npy",
    connectivity_metadata = "data/processed/connectivity_metadata.csv",
    pheno_path            = "data/raw/Phenotypic_V1_0b.csv",
    results_dir           = "results/track_a",
    fmri_pca=100, n_splits=10, n_splits_neural=5, n_boot=1000, seed=42,
)


def load_data(cfg):
    X_fmri = np.load(cfg["connectivity_matrix"])
    meta   = pd.read_csv(cfg["connectivity_metadata"])
    y      = (meta["DX_GROUP"].values==1).astype(int)
    sex    = meta["SEX"].values
    pheno  = pd.read_csv(cfg["pheno_path"],low_memory=False).replace(-9999,np.nan)
    meta["SUB_ID"]=meta["SUB_ID"].astype(int)
    pheno["SUB_ID"]=pheno["SUB_ID"].astype(int)
    mg     = meta.merge(pheno,on="SUB_ID",how="left")
    X_pheno= mg[["FIQ","VIQ","PIQ"]].values.astype(float)
    X_demo = mg[["AGE_AT_SCAN","SEX"]].values.astype(float)
    print(f"Loaded: fMRI={X_fmri.shape} ASD={y.sum()} TDC={(1-y).sum()}")
    return X_fmri, X_pheno, X_demo, y, sex


def run_track_a(cfg):
    seed    = cfg["seed"]
    out_dir = Path(cfg["results_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    X_fmri, X_pheno, X_demo, y, sex = load_data(cfg)
    pca = cfg["fmri_pca"]

    models = {
        "RandomForest":    lambda: make_random_forest(seed=seed),
        "XGBoost":         lambda: make_xgboost(y_train=y,seed=seed),
        "LightGBM":        lambda: make_lightgbm(seed=seed),
        "MLP":             lambda: make_mlp(seed=seed),
        "TabTransformer":  lambda: SklearnWrapper(TabTransformer,{"d_model":64,"n_heads":4,"dropout":0.2},epochs=50,seed=seed),
        "BrainGNN":        lambda: SklearnWrapper(BrainGNN,{"hidden":128,"dropout":0.35},epochs=50,seed=seed),
        "GraphTransformer":lambda: SklearnWrapper(GraphTransformer,{"d_model":64,"n_heads":4,"dropout":0.2},epochs=50,seed=seed),
    }
    modalities = {
        "fmri":  (X_fmri,  {"pca_components":pca,"impute_strategy":"mean"}),
        "pheno": (X_pheno, {"pca_components":None,"impute_strategy":"median"}),
        "demo":  (X_demo,  {"pca_components":None,"impute_strategy":"median"}),
    }
    results=[]
    for mod, (X_mod, kw) in modalities.items():
        for clf_name, clf_fn in models.items():
            t0  = time.time()
            is_n= clf_name in ["TabTransformer","BrainGNN","GraphTransformer"]
            ns  = cfg["n_splits_neural"] if is_n else cfg["n_splits"]
            print(f"\n[{mod.upper()}] {clf_name} ({ns}-fold)...")
            try:
                res = run_cv(clf_fn,X_mod,y,n_splits=ns,sex=sex,
                              n_boot=cfg["n_boot"],seed=seed,**kw)
                ov  = res["overall"]; ci=res["ci"]; sx=res["sex_metrics"]
                print(format_results(res,f"{mod} {clf_name}"))
                np.save(out_dir/f"oof_{mod}_{clf_name}.npy",
                        np.stack([res["oof_true"],res["oof_score"]],1))
                row = {"Modality":mod,"Model":clf_name,"CV_splits":ns,
                       "AUC":round(ov["AUC"],4),"AUC_lo":round(ci["AUC"][0],4),
                       "AUC_hi":round(ci["AUC"][1],4),"BAC":round(ov["BAC"],4),
                       "F1":round(ov["F1"],4),"Sens":round(ov["Sens"],4),
                       "Spec":round(ov["Spec"],4),"FNR":round(ov["FNR"],4),
                       "AUC_Male":round(sx.get("Male",{}).get("AUC",float("nan")),4),
                       "AUC_Female":round(sx.get("Female",{}).get("AUC",float("nan")),4),
                       "MNAR_flag":(mod=="behav"),"status":"OK",
                       "elapsed_s":round(time.time()-t0,1)}
            except Exception as e:
                print(f"  ERROR: {e}")
                row={"Modality":mod,"Model":clf_name,"status":f"ERROR:{e}","AUC":float("nan")}
            results.append(row)

    df=pd.DataFrame(results)
    df.to_csv(out_dir/"track_a_results.csv",index=False)
    print(f"\n✓ Track A → {out_dir}/track_a_results.csv")
    ok=df[df["status"]=="OK"].sort_values(["Modality","AUC"],ascending=[True,False])
    print(ok[["Modality","Model","AUC","AUC_lo","AUC_hi","BAC","Sens","Spec"]].to_string(index=False))
    return df


def main():
    p=argparse.ArgumentParser(description="Track A: Unimodal Baselines")
    p.add_argument("--modality",    default=None)
    p.add_argument("--n-splits",    type=int,default=10)
    p.add_argument("--seed",        type=int,default=42)
    p.add_argument("--audit-only",  action="store_true")
    args=p.parse_args()
    cfg={**DEFAULT,"n_splits":args.n_splits,"n_splits_neural":min(args.n_splits,5),"seed":args.seed}
    if args.audit_only:
        from asd_multimodal.data.abide_loader import load_abide1_phenotypic,check_motion_confound
        df1=load_abide1_phenotypic(cfg["pheno_path"])
        meta=pd.read_csv(cfg["connectivity_metadata"])
        print(check_motion_confound(meta))
    else:
        run_track_a(cfg)

if __name__=="__main__": main()
