#!/usr/bin/env python3
"""
experiments/track_b_fusion.py — Track B: Multimodal Fusion.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys, argparse, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from asd_multimodal.models.fusion import LateFusionStacked, LateFusionAverage, EarlyFusion, CMAWrapper
from asd_multimodal.training.cv_engine import run_multimodal_cv, format_results
from asd_multimodal.utils.metrics import bootstrap_permutation_auc


def load_data(cfg):
    X_fmri=np.load(cfg["connectivity_matrix"]); meta=pd.read_csv(cfg["connectivity_metadata"])
    y=( meta["DX_GROUP"].values==1).astype(int); sex=meta["SEX"].values
    pheno=pd.read_csv(cfg["pheno_path"],low_memory=False).replace(-9999,np.nan)
    meta["SUB_ID"]=meta["SUB_ID"].astype(int); pheno["SUB_ID"]=pheno["SUB_ID"].astype(int)
    mg=meta.merge(pheno,on="SUB_ID",how="left")
    return X_fmri,mg[["FIQ","VIQ","PIQ"]].values.astype(float),mg[["AGE_AT_SCAN","SEX"]].values.astype(float),y,sex


DEFAULT=dict(connectivity_matrix="data/processed/connectivity_matrix.npy",
             connectivity_metadata="data/processed/connectivity_metadata.csv",
             pheno_path="data/raw/Phenotypic_V1_0b.csv",
             results_dir="results/track_b",fmri_pca=100,n_splits=10,
             n_splits_neural=5,n_boot=1000,seed=42)


def run_track_b(cfg):
    seed=cfg["seed"]; pca=cfg["fmri_pca"]
    out_dir=Path(cfg["results_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    X_fmri,X_pheno,X_demo,y,sex=load_data(cfg); results=[]

    experiments=[
        ("LF_Stacked",    lambda:LateFusionStacked(seed=seed),         cfg["n_splits"]),
        ("LF_Average",    lambda:LateFusionAverage(weights=[0.7,0.3],seed=seed), cfg["n_splits"]),
        ("CMA",           lambda:CMAWrapper(fmri_dim=pca,pheno_dim=3,demo_dim=2,seed=seed), cfg["n_splits_neural"]),
    ]
    try:
        import xgboost as xgb
        from asd_multimodal.models.fusion import EarlyFusion
        experiments.append(("EF_XGBoost",
            lambda:EarlyFusion(xgb.XGBClassifier(n_estimators=300,max_depth=4,
                learning_rate=0.05,scale_pos_weight=(1-y).sum()/y.sum(),
                verbosity=0,random_state=seed,n_jobs=-1)),cfg["n_splits"]))
    except ImportError: pass

    for name,clf_fn,ns in experiments:
        print(f"\n[{name}] ({ns}-fold)...")
        t0=time.time()
        try:
            res=run_multimodal_cv(clf_fn,X_fmri,X_pheno,X_demo,y,
                                   fmri_pca=pca,n_splits=ns,sex=sex,
                                   n_boot=cfg["n_boot"],seed=seed)
            print(format_results(res,name))
            np.save(out_dir/f"oof_B_{name}.npy",np.stack([res["oof_true"],res["oof_score"]],1))
            ov=res["overall"]; ci=res["ci"]
            results.append({"Strategy":name,"AUC":round(ov["AUC"],4),
                             "AUC_lo":round(ci["AUC"][0],4),"AUC_hi":round(ci["AUC"][1],4),
                             "BAC":round(ov["BAC"],4),"Sens":round(ov["Sens"],4),
                             "Spec":round(ov["Spec"],4),"elapsed_s":round(time.time()-t0,1)})
        except Exception as e:
            print(f"  ERROR: {e}"); results.append({"Strategy":name,"status":f"ERROR:{e}"})

    df=pd.DataFrame(results); df.to_csv(out_dir/"track_b_results.csv",index=False)
    print(f"\n✓ Track B → {out_dir}/track_b_results.csv")
    print(df[["Strategy","AUC","AUC_lo","AUC_hi","BAC","Sens","Spec"]].to_string(index=False))
    return df


def main():
    p=argparse.ArgumentParser(description="Track B: Multimodal Fusion")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    run_track_b({**DEFAULT,"seed":args.seed})

if __name__=="__main__": main()
