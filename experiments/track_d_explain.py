#!/usr/bin/env python3
"""
experiments/track_d_explain.py — Track D: Explainability Analysis.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from asd_multimodal.data.preprocessing import MultimodalPreprocessor
from asd_multimodal.models.unimodal import make_random_forest
from asd_multimodal.explainability.shap_analysis import (
    compute_shap_values, compute_permutation_importance,
    compute_integrated_gradients, method_agreement_analysis,
    pc_backproject_to_networks)
from asd_multimodal.utils.metrics import expected_calibration_error

DEFAULT=dict(connectivity_matrix="data/processed/connectivity_matrix.npy",
             connectivity_metadata="data/processed/connectivity_metadata.csv",
             pheno_path="data/raw/Phenotypic_V1_0b.csv",
             results_dir="results/track_d",fmri_pca=100,
             shap_samples=80,ig_samples=50,seed=42)


def run_track_d(cfg):
    out_dir=Path(cfg["results_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    X_fmri=np.load(cfg["connectivity_matrix"]); meta=pd.read_csv(cfg["connectivity_metadata"])
    y=(meta["DX_GROUP"].values==1).astype(int)
    pheno=pd.read_csv(cfg["pheno_path"],low_memory=False).replace(-9999,np.nan)
    meta["SUB_ID"]=meta["SUB_ID"].astype(int); pheno["SUB_ID"]=pheno["SUB_ID"].astype(int)
    mg=meta.merge(pheno,on="SUB_ID",how="left")
    X_pheno=mg[["FIQ","VIQ","PIQ"]].values.astype(float)
    X_demo=mg[["AGE_AT_SCAN","SEX"]].values.astype(float)

    tr,te=train_test_split(np.arange(len(y)),test_size=0.20,random_state=cfg["seed"],stratify=y)
    y_tr,y_te=y[tr],y[te]
    prep=MultimodalPreprocessor(fmri_pca=cfg["fmri_pca"],seed=cfg["seed"])
    X_tr=prep.fit_transform(X_fmri[tr],X_pheno[tr],X_demo[tr])
    X_te=prep.transform(X_fmri[te],X_pheno[te],X_demo[te])

    print("[D1] Training RF + SHAP...")
    rf=make_random_forest(seed=cfg["seed"]); rf.fit(X_tr,y_tr)
    sv,shap_df=compute_shap_values(rf,X_tr,X_te,max_samples=cfg["shap_samples"],seed=cfg["seed"])
    np.save(out_dir/"shap_values.npy",sv); shap_df.to_csv(out_dir/"shap_summary.csv",index=False)
    print(f"  Top-5: {shap_df['Feature'].head(5).tolist()}")

    print("[D2] Permutation Importance...")
    pi_df=compute_permutation_importance(rf,X_te,y_te,n_repeats=15,seed=cfg["seed"])
    pi_df.to_csv(out_dir/"pi_summary.csv",index=False)

    print("[D3] Integrated Gradients...")
    ig_df=compute_integrated_gradients(None,X_tr,X_te,y_tr,n_steps=30,
                                        n_samples=cfg["ig_samples"],epochs=60,seed=cfg["seed"])
    ig_df.to_csv(out_dir/"ig_summary.csv",index=False)

    print("[D4] Method agreement...")
    agreement=method_agreement_analysis(shap_df,pi_df,ig_df)
    pd.DataFrame([agreement]).to_csv(out_dir/"method_agreement.csv",index=False)

    print("[D5] PC back-projection...")
    top_pcs=(shap_df[shap_df["Modality"]=="fMRI"].head(5)["Feature"]
             .str.replace("fMRI_PC","").astype(int).values-1).tolist()
    pca_fitted=prep._fmri_prep._pca
    if pca_fitted is not None:
        net_df=pc_backproject_to_networks(pca_fitted.components_,top_pcs)
        net_df.to_csv(out_dir/"pc_network_attribution.csv",index=False)
        print(f"  Network attribution saved ({len(net_df)} rows)")

    print("[D6] Calibration...")
    ys_raw=rf.predict_proba(X_te)[:,1]
    ece_raw=expected_calibration_error(y_te,ys_raw)
    rf_platt=CalibratedClassifierCV(make_random_forest(seed=cfg["seed"]),cv=5,method="sigmoid")
    rf_platt.fit(X_tr,y_tr); ys_cal=rf_platt.predict_proba(X_te)[:,1]
    ece_cal=expected_calibration_error(y_te,ys_cal)
    calib_df=pd.DataFrame([{"Model":"RF_uncalibrated",**{k:v for k,v in ece_raw.items() if not isinstance(v,list)}},
                             {"Model":"RF_Platt",       **{k:v for k,v in ece_cal.items() if not isinstance(v,list)}}])
    calib_df.to_csv(out_dir/"calibration_results.csv",index=False)
    print(f"  RF uncalibrated: ECE={ece_raw['ECE']:.4f}  Brier={ece_raw['Brier']:.4f}")
    print(f"  RF + Platt:      ECE={ece_cal['ECE']:.4f}  Brier={ece_cal['Brier']:.4f}")
    print(f"\n✓ Track D → {out_dir}/")
    return shap_df, pi_df, ig_df


def main():
    p=argparse.ArgumentParser(description="Track D: Explainability")
    p.add_argument("--fmri-pca",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    run_track_d({**DEFAULT,"fmri_pca":args.fmri_pca,"seed":args.seed})

if __name__=="__main__": main()
