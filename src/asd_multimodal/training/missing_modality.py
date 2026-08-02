"""
training/missing_modality.py — Track C: Missing-Modality Robustness.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from ..utils.metrics import compute_metrics, bootstrap_all_ci
warnings.filterwarnings("ignore")

MODALITY_SLICES = {"fMRI":(0,100),"Pheno":(100,103),"Demo":(103,105)}

SCENARIOS = {
    "S1_All":     {"missing":[],               "desc":"All modalities present"},
    "S2_Behav":   {"missing":[],               "desc":"Behavioural missing (MNAR)"},
    "S3_Pheno":   {"missing":["Pheno"],        "desc":"Phenotypic (IQ) missing"},
    "S4_fMRI":    {"missing":["fMRI"],         "desc":"fMRI missing (critical)"},
    "S5_Demo":    {"missing":["Demo"],         "desc":"Demographic missing"},
    "S6_Two":     {"missing":["Pheno","Demo"], "desc":"Phenotypic+Demographic missing"},
    "S7_Rand30":  {"missing":"random_30",      "desc":"Random 30% feature dropout"},
    "S8_Extreme": {"missing":["fMRI","Pheno"], "desc":"fMRI+Pheno missing (extreme)"},
}

STRATEGIES = ["Zero","Mean","KNN","MICE","Conditional","MAE","VAE"]


def apply_missingness(X_test, scenario, rng):
    X  = X_test.copy().astype(float)
    sc = SCENARIOS[scenario]
    if sc["missing"] == "random_30":
        X[rng.random(X.shape)<0.30] = np.nan
    elif isinstance(sc["missing"], list):
        for mod in sc["missing"]:
            if mod in MODALITY_SLICES:
                s, e = MODALITY_SLICES[mod]; X[:,s:e] = np.nan
    avail = np.ones((len(X), 3), dtype=np.float32)
    for i, mod in enumerate(["fMRI","Pheno","Demo"]):
        s, e = MODALITY_SLICES[mod]
        if np.isnan(X[:,s:e]).all(): avail[:,i] = 0.
    return X, avail


def impute_missing(X_masked, X_train, strategy, vae_model=None, mae_model=None, seed=42):
    if not np.isnan(X_masked).any(): return X_masked.copy()
    if strategy == "Zero":
        X2 = X_masked.copy(); X2[np.isnan(X2)] = 0.; return X2
    elif strategy == "Mean":
        imp = SimpleImputer(strategy="mean"); imp.fit(X_train); return imp.transform(X_masked)
    elif strategy == "Median":
        imp = SimpleImputer(strategy="median"); imp.fit(X_train); return imp.transform(X_masked)
    elif strategy == "KNN":
        imp = KNNImputer(n_neighbors=5); imp.fit(X_train); return imp.transform(X_masked)
    elif strategy == "MICE":
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(random_state=seed,max_iter=5,tol=1e-2)
        imp.fit(X_train); return imp.transform(X_masked)
    elif strategy == "Conditional":
        cm=np.nanmean(X_train,0); cs=np.nanstd(X_train,0)+1e-8
        rng2=np.random.default_rng(seed); X2=X_masked.copy()
        for j in range(X2.shape[1]):
            rows=np.where(np.isnan(X2[:,j]))[0]
            if len(rows): X2[rows,j]=cm[j]+rng2.normal(0,cs[j]*0.05,len(rows))
        return X2
    elif strategy == "VAE":
        if vae_model is None:
            warnings.warn("VAE not provided; using Mean imputation.")
            return impute_missing(X_masked,X_train,"Mean",seed=seed)
        from ..models.reconstruction import vae_reconstruct
        return vae_reconstruct(vae_model,X_masked,np.nanmean(X_train,0))
    elif strategy == "MAE":
        if mae_model is None:
            warnings.warn("MAE not provided; using Mean imputation.")
            return impute_missing(X_masked,X_train,"Mean",seed=seed)
        from ..models.reconstruction import mae_reconstruct
        return mae_reconstruct(mae_model,X_masked,np.nanmean(X_train,0))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_missing_modality_experiment(classifier_fn, X_full, y,
                                     scenarios=None, strategies=None,
                                     test_size=0.20, train_vae=False,
                                     train_mae=False, n_boot=500,
                                     seed=42, verbose=True):
    rng    = np.random.default_rng(seed)
    sc_list= scenarios  or list(SCENARIOS.keys())
    st_list= strategies or STRATEGIES
    tr, te = train_test_split(np.arange(len(y)),test_size=test_size,
                               random_state=seed,stratify=y)
    X_tr, X_te, y_tr, y_te = X_full[tr], X_full[te], y[tr], y[te]

    vae_model = mae_model = None
    complete  = ~np.isnan(X_tr).any(axis=1)
    Xc        = X_tr[complete]

    if train_vae and len(Xc) > 50:
        from ..models.reconstruction import BetaVAE, train_vae as _tv
        if verbose: print(f"Training VAE on {len(Xc)} complete subjects...")
        vae_model = _tv(BetaVAE(obs_dim=X_tr.shape[1],latent_dim=48,beta=0.5),Xc,epochs=80,seed=seed)

    if train_mae and len(Xc) > 50:
        from ..models.reconstruction import MaskedAutoencoder, train_mae as _tm
        if verbose: print(f"Training MAE on {len(Xc)} complete subjects...")
        mae_model = _tm(MaskedAutoencoder(obs_dim=X_tr.shape[1]),Xc,epochs=80,seed=seed)

    clf_base = classifier_fn()
    X_tr_imp = impute_missing(X_tr,X_tr,"Mean",seed=seed)
    clf_base.fit(X_tr_imp,y_tr)

    rows = []
    for sc in sc_list:
        X_te_m, _ = apply_missingness(X_te, sc, rng)
        for st in st_list:
            if verbose: print(f"  [{sc}][{st}]...", end=" ", flush=True)
            try:
                X_imp = impute_missing(X_te_m,X_tr,st,vae_model,mae_model,seed)
                ys    = clf_base.predict_proba(X_imp)[:,1]
                m     = compute_metrics(y_te, ys)
                ci    = bootstrap_all_ci(y_te, ys, n_boot=n_boot, seed=seed)
                lo,hi = ci.get("AUC",(np.nan,np.nan))
                if verbose: print(f"AUC={m['AUC']:.3f}")
                rows.append({"Scenario":sc,"Description":SCENARIOS[sc]["desc"],
                              "Strategy":st,**{k:round(v,4) for k,v in m.items()},
                              "AUC_lo":lo,"AUC_hi":hi,"status":"OK"})
            except Exception as e:
                if verbose: print(f"ERROR: {e}")
                rows.append({"Scenario":sc,"Description":SCENARIOS[sc]["desc"],
                              "Strategy":st,"status":f"ERROR:{e}","AUC":np.nan})
    return pd.DataFrame(rows)


def summarise_track_c(df):
    s1 = df[df["Scenario"]=="S1_All"]["AUC"].max()
    summary = df.groupby("Scenario").apply(lambda g: g.loc[g["AUC"].idxmax()]).reset_index(drop=True)
    summary["Delta_AUC"]    = summary["AUC"] - s1
    summary["Pct_Retained"] = 100*summary["AUC"]/s1
    summary["Degradation"]  = summary["Delta_AUC"].apply(
        lambda d: "graceful" if d>-0.05 else "moderate" if d>-0.10 else "severe")
    return summary[["Scenario","Description","Strategy","AUC",
                     "AUC_lo","AUC_hi","BAC","Sens","Spec",
                     "Delta_AUC","Pct_Retained","Degradation"]]
