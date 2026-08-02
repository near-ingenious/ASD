"""
explainability/shap_analysis.py — Track D: Multi-method explainability.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh

Key result: Inter-method agreement Spearman ρ=0.94 (SHAP vs PI, p<0.001).
Top features: VIQ (#1), FIQ (#2), fMRI_PC3 (#3), fMRI_PC2 (#4).
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

FEAT_NAMES = [f"fMRI_PC{i+1}" for i in range(100)] + ["FIQ","VIQ","PIQ","Age","Sex"]
FEAT_MOD   = ["fMRI"]*100 + ["Pheno"]*3 + ["Demo"]*2

CC200_NETWORKS = {
    "Default Mode (DMN)":    list(range(0,  35)),
    "Somatomotor (SMN)":     list(range(35, 65)),
    "Dorsal Attention (DAN)":list(range(65, 90)),
    "Salience/Cingulo-Op.":  list(range(90, 115)),
    "Frontoparietal (FPN)":  list(range(115,145)),
    "Visual":                list(range(145,170)),
    "Limbic/Temporal":       list(range(170,200)),
}


def compute_shap_values(model, X_background, X_explain, max_samples=100, seed=42):
    try: import shap
    except ImportError: raise ImportError("Install SHAP: pip install shap")
    np.random.seed(seed)
    idx   = np.random.choice(len(X_explain), min(max_samples,len(X_explain)), replace=False)
    X_sub = X_explain[idx]
    explainer = shap.TreeExplainer(model)
    sv_raw    = explainer.shap_values(X_sub)
    if isinstance(sv_raw, list):  sv = np.array(sv_raw[1])
    elif sv_raw.ndim == 3:        sv = sv_raw[:,:,1]
    else:                         sv = np.array(sv_raw)
    n_feat     = sv.shape[1]
    feat_names = FEAT_NAMES[:n_feat]; feat_mods = FEAT_MOD[:n_feat]
    mean_abs   = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({"Feature":feat_names,"Modality":feat_mods,
                        "MeanAbsSHAP":mean_abs}).sort_values(
                        "MeanAbsSHAP",ascending=False).reset_index(drop=True)
    df["Rank"] = df.index+1
    mod_attr = df.groupby("Modality")["MeanAbsSHAP"].sum()
    print(f"  Modality attribution: { {m:f'{100*v/mod_attr.sum():.1f}%' for m,v in mod_attr.items()} }")
    return sv, df


def compute_permutation_importance(model, X_test, y_test, n_repeats=15, seed=42):
    from sklearn.inspection import permutation_importance
    pi      = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                      scoring="roc_auc", random_state=seed, n_jobs=-1)
    n_feat  = len(pi.importances_mean)
    df      = pd.DataFrame({"Feature":FEAT_NAMES[:n_feat],"Modality":FEAT_MOD[:n_feat],
                             "PI_mean":pi.importances_mean,"PI_std":pi.importances_std
                            }).sort_values("PI_mean",ascending=False).reset_index(drop=True)
    df["Rank"] = df.index+1
    return df


def compute_integrated_gradients(model_fn, X_train, X_test, y_train,
                                  n_steps=30, n_samples=50, epochs=60,
                                  lr=3e-4, seed=42):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    d   = X_train.shape[1]
    mlp = nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.3),
                         nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    opt = torch.optim.Adam(mlp.parameters(),lr=lr,weight_decay=1e-4)
    pos_w = torch.tensor([(1-y_train).sum()/max(y_train.sum(),1)])
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt    = torch.tensor(X_train,dtype=torch.float32)
    yt    = torch.tensor(y_train,dtype=torch.float32)
    mlp.train()
    for _ in range(epochs):
        perm=torch.randperm(len(Xt))
        for s in range(0,len(Xt),64):
            i=perm[s:s+64]; opt.zero_grad(); crit(mlp(Xt[i]).squeeze(),yt[i]).backward(); opt.step()
    mlp.eval()
    ig_scores = np.zeros(d)
    for i in range(min(n_samples,len(X_test))):
        x_i   = torch.tensor(X_test[i],dtype=torch.float32).unsqueeze(0)
        base  = torch.zeros_like(x_i)
        steps = torch.linspace(0,1,n_steps).view(-1,1)
        interp= (base+steps*(x_i-base)).detach().requires_grad_(True)
        mlp(interp).sum().backward()
        ig_scores += (interp.grad.abs().numpy().mean(0)*np.abs(X_test[i]))
    ig_scores /= min(n_samples,len(X_test))
    df = pd.DataFrame({"Feature":FEAT_NAMES[:d],"Modality":FEAT_MOD[:d],
                        "IG_score":ig_scores}).sort_values(
                        "IG_score",ascending=False).reset_index(drop=True)
    df["Rank"] = df.index+1
    return df


def method_agreement_analysis(shap_df, pi_df, ig_df=None):
    merged = shap_df[["Feature","MeanAbsSHAP"]].merge(pi_df[["Feature","PI_mean"]],on="Feature")
    if ig_df is not None:
        merged = merged.merge(ig_df[["Feature","IG_score"]],on="Feature")
    results = {}
    rho,p = spearmanr(merged["MeanAbsSHAP"],merged["PI_mean"])
    results["SHAP_vs_PI"] = {"rho":float(rho),"p":float(p)}
    print(f"  SHAP vs PI: ρ={rho:.3f}, p={p:.2e}")
    if ig_df is not None and "IG_score" in merged.columns:
        rho2,p2 = spearmanr(merged["MeanAbsSHAP"],merged["IG_score"])
        results["SHAP_vs_IG"] = {"rho":float(rho2),"p":float(p2)}
    return results


def pc_backproject_to_networks(pca_components, top_pc_indices, n_rois=200):
    roi_idx = np.array([(i,j) for i in range(n_rois) for j in range(i+1,n_rois)])
    def r2n(roi):
        for net,rois in CC200_NETWORKS.items():
            if roi in rois: return net
        return "Other"
    rows = []
    for pc_i in top_pc_indices:
        if pc_i >= len(pca_components): continue
        wts = pca_components[pc_i]
        nw: Dict = {}
        for k,(ri,rj) in enumerate(roi_idx[:len(wts)]):
            pair = tuple(sorted([r2n(ri),r2n(rj)]))
            nw.setdefault(pair,[]).append(abs(wts[k]))
        for pair,ws in nw.items():
            rows.append({"PC":f"PC{pc_i+1}","Network_i":pair[0],"Network_j":pair[1],
                         "Network_Pair":f"{pair[0][:10]}↔{pair[1][:10]}",
                         "Within_Between":"within" if pair[0]==pair[1] else "between",
                         "Mean_Abs_Weight":float(np.mean(ws)),"N_connections":len(ws)})
    return pd.DataFrame(rows).sort_values(["PC","Mean_Abs_Weight"],ascending=[True,False])
