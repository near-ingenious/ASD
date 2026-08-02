"""
training/cv_engine.py — Stratified CV engine with bootstrap CIs.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
from __future__ import annotations
import time, warnings
from typing import Callable, Dict, List, Optional
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from ..utils.metrics import compute_metrics, compute_sex_metrics, bootstrap_all_ci
from ..data.preprocessing import ModalityPreprocessor, MultimodalPreprocessor
warnings.filterwarnings("ignore")


def run_cv(model_fn, X, y, n_splits=10, pca_components=None,
           impute_strategy="mean", sex=None, n_boot=1000, seed=42, verbose=True):
    t0  = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_t, oof_s, fold_rows = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        prep = ModalityPreprocessor(impute_strategy=impute_strategy,
                                     n_pca=pca_components, scale=True, seed=seed)
        X_tr = prep.fit_transform(X_tr); X_te = prep.transform(X_te)
        clf  = model_fn()
        clf.fit(X_tr, y_tr)
        ys   = clf.predict_proba(X_te)[:, 1]
        m    = compute_metrics(y_te, ys); m["fold"] = fold; fold_rows.append(m)
        if verbose:
            print(f"  Fold {fold+1}/{n_splits}: AUC={m['AUC']:.3f} "
                  f"Sens={m['Sens']:.3f}", flush=True)
        oof_t.extend(y_te); oof_s.extend(ys)
    oof_t = np.array(oof_t); oof_s = np.array(oof_s)
    overall = compute_metrics(oof_t, oof_s)
    ci      = bootstrap_all_ci(oof_t, oof_s, n_boot=n_boot, seed=seed)
    sex_m   = compute_sex_metrics(oof_t, oof_s, sex) if sex is not None else {}
    return dict(overall=overall, ci=ci, fold_metrics=pd.DataFrame(fold_rows),
                sex_metrics=sex_m, oof_true=oof_t, oof_score=oof_s,
                elapsed_s=time.time()-t0, n_splits=n_splits, pca_components=pca_components)


def run_multimodal_cv(model_fn, X_fmri, X_pheno, X_demo, y,
                      fmri_pca=100, n_splits=10, sex=None,
                      n_boot=1000, seed=42, verbose=True):
    t0  = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_t, oof_s, fold_rows = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_fmri, y)):
        y_tr, y_te = y[tr], y[te]
        prep = MultimodalPreprocessor(fmri_pca=fmri_pca, seed=seed)
        X_tr_full = prep.fit_transform(X_fmri[tr], X_pheno[tr], X_demo[tr])
        X_te_full = prep.transform(X_fmri[te],  X_pheno[te],  X_demo[te])
        fs, fe    = prep.get_slice("fMRI");  ps, pe = prep.get_slice("Pheno")
        clf = model_fn()
        try:
            clf.fit(X_tr_full[:,fs:fe], X_tr_full[:,ps:pe], y_tr)
            ys  = clf.predict_proba(X_te_full[:,fs:fe], X_te_full[:,ps:pe])[:, 1]
        except TypeError:
            clf.fit(X_tr_full, y_tr)
            ys  = clf.predict_proba(X_te_full)[:, 1]
        m = compute_metrics(y_te, ys); m["fold"] = fold; fold_rows.append(m)
        if verbose: print(f"  Fold {fold+1}/{n_splits}: AUC={m['AUC']:.3f}", flush=True)
        oof_t.extend(y_te); oof_s.extend(ys)
    oof_t = np.array(oof_t); oof_s = np.array(oof_s)
    overall = compute_metrics(oof_t, oof_s)
    ci      = bootstrap_all_ci(oof_t, oof_s, n_boot=n_boot, seed=seed)
    sex_m   = compute_sex_metrics(oof_t, oof_s, sex) if sex is not None else {}
    return dict(overall=overall, ci=ci, fold_metrics=pd.DataFrame(fold_rows),
                sex_metrics=sex_m, oof_true=oof_t, oof_score=oof_s,
                elapsed_s=time.time()-t0)


def format_results(res, model_name=""):
    ov = res["overall"]; ci = res["ci"]
    lo, hi = ci.get("AUC", (np.nan, np.nan))
    lines  = [f"{'='*55}", f"  {model_name or 'Model'}", f"{'='*55}",
              f"  AUC:  {ov['AUC']:.3f} [{lo:.3f}–{hi:.3f}]",
              f"  BAC:  {ov['BAC']:.3f}   F1: {ov['F1']:.3f}",
              f"  Sens: {ov['Sens']:.3f}  Spec: {ov['Spec']:.3f}",
              f"  Time: {res['elapsed_s']:.1f}s"]
    sx = res.get("sex_metrics", {})
    if "Female" in sx:
        f = sx["Female"]
        lines.append(f"  Female [{f.get('status','?')}]: "
                     f"AUC={f.get('AUC',np.nan):.3f}  FNR={f.get('FNR',np.nan):.3f}")
    return "\n".join(lines)
