"""
utils/metrics.py — Core evaluation metrics with bootstrap CIs.

Authors:    Jarin Alam Prity (222-115-005)  jarinprity438@gmail.com
            Popy Rani Boidya (007)           popyboidya@gmail.com
Supervisor: Md Mahfujul Hasan — Metropolitan University, Sylhet
Clinical:   Prof. Imdadul Magfur — Sylhet MAG Osmani Medical College
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, brier_score_loss, log_loss,
)
from scipy.stats import mannwhitneyu, chi2_contingency
from typing import Optional


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in
                ["AUC","BAC","F1","Sens","Spec","PPV","NPV","FNR","FPR","Brier","NLL"]}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    eps = 1e-9
    return dict(
        AUC   = float(roc_auc_score(y_true, y_score)),
        BAC   = float(balanced_accuracy_score(y_true, y_pred)),
        F1    = float(f1_score(y_true, y_pred, zero_division=0)),
        Sens  = float(tp / (tp + fn + eps)),
        Spec  = float(tn / (tn + fp + eps)),
        PPV   = float(tp / (tp + fp + eps)),
        NPV   = float(tn / (tn + fn + eps)),
        FNR   = float(fn / (fn + tp + eps)),
        FPR   = float(fp / (fp + tn + eps)),
        Brier = float(brier_score_loss(y_true, y_score)),
        NLL   = float(log_loss(y_true, y_score)),
    )


def compute_sex_metrics(y_true, y_score, sex, threshold=0.5):
    out = {}
    for sx, label in [(1,'Male'),(2,'Female')]:
        mask = sex == sx
        n    = int(mask.sum())
        if n < 5:
            out[label] = {"N": n, "status": "INSUFFICIENT"}
            continue
        m = compute_metrics(y_true[mask], y_score[mask], threshold)
        m["N"]      = n
        m["status"] = "EXPLORATORY" if n < 30 else "OK"
        out[label]  = m
    if "Male" in out and "Female" in out:
        for k in ["AUC","Sens","Spec","FNR"]:
            vm = out["Male"].get(k,  np.nan)
            vf = out["Female"].get(k,np.nan)
            out[f"Gap_{k}"] = float(vm-vf) if not (np.isnan(vm) or np.isnan(vf)) else np.nan
    return out


def bootstrap_ci(y_true, y_score, metric="AUC", n_boot=1000, alpha=0.05, seed=42):
    rng    = np.random.default_rng(seed)
    scores = []
    n      = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            m = compute_metrics(y_true[idx], y_score[idx])
            scores.append(m[metric])
        except Exception:
            pass
    scores = np.array(scores)
    return float(np.nanpercentile(scores, 100*alpha/2)), \
           float(np.nanpercentile(scores, 100*(1-alpha/2)))


def bootstrap_all_ci(y_true, y_score, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    rows = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            rows.append(compute_metrics(y_true[idx], y_score[idx]))
        except Exception:
            pass
    df  = pd.DataFrame(rows)
    lo  = df.quantile(0.025).to_dict()
    hi  = df.quantile(0.975).to_dict()
    return {k: (lo[k], hi[k]) for k in lo}


def bootstrap_permutation_auc(y_true1, y_score1, y_true2, y_score2,
                               n_boot=1000, alternative="two-sided", seed=42):
    rng      = np.random.default_rng(seed)
    auc1     = roc_auc_score(y_true1, y_score1)
    auc2     = roc_auc_score(y_true2, y_score2)
    delta_obs= auc1 - auc2
    diffs    = []
    for _ in range(n_boot):
        i1 = rng.integers(0, len(y_true1), len(y_true1))
        i2 = rng.integers(0, len(y_true2), len(y_true2))
        try:
            diffs.append(roc_auc_score(y_true1[i1], y_score1[i1]) -
                         roc_auc_score(y_true2[i2], y_score2[i2]))
        except Exception:
            pass
    diffs = np.array(diffs)
    se    = diffs.std()
    if alternative == "two-sided":
        p = 2 * min((diffs >= delta_obs).mean(), (diffs <= delta_obs).mean())
    elif alternative == "greater":
        p = (diffs <= delta_obs).mean()
    else:
        p = (diffs >= delta_obs).mean()
    return dict(auc1=float(auc1), auc2=float(auc2), delta=float(delta_obs),
                se=float(se), p_value=float(p),
                significant_05=bool(p<0.05), significant_01=bool(p<0.01),
                significant_001=bool(p<0.001), n_boot=n_boot,
                note="Approximate: independent bootstrap (not paired DeLong)")


def mcnemar_test(y_true, y_pred1, y_pred2):
    from scipy.stats import chi2
    b = int(((y_pred1==y_true)&(y_pred2!=y_true)).sum())
    c = int(((y_pred1!=y_true)&(y_pred2==y_true)).sum())
    if b+c == 0:
        return dict(chi2=0.0, p_value=1.0, b=b, c=c, note="No disagreements")
    chi2_stat = (abs(b-c)-1)**2 / (b+c)
    return dict(chi2=float(chi2_stat), p_value=float(1-chi2.cdf(chi2_stat,df=1)),
                b=b, c=c, note="Continuity-corrected McNemar (Yates)")


def mann_whitney_test(x, y):
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return dict(U=float(stat), p_value=float(p),
                mean_x=float(np.nanmean(x)), mean_y=float(np.nanmean(y)),
                std_x=float(np.nanstd(x)),  std_y=float(np.nanstd(y)))


def expected_calibration_error(y_true, y_score, n_bins=10, strategy="uniform"):
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_score,
                                                 n_bins=n_bins, strategy=strategy)
    except ValueError:
        return dict(ECE=np.nan, frac_pos=[], mean_pred=[], bin_counts=[])
    edges      = np.linspace(0,1,n_bins+1)
    bin_counts = [int(((y_score>=edges[i])&(y_score<edges[i+1])).sum())
                  for i in range(n_bins)]
    return dict(ECE=float(np.mean(np.abs(frac_pos-mean_pred))),
                MCE=float(np.max(np.abs(frac_pos-mean_pred))),
                Brier=float(brier_score_loss(y_true, y_score)),
                NLL=float(log_loss(y_true, y_score)),
                frac_pos=frac_pos.tolist(), mean_pred=mean_pred.tolist(),
                bin_counts=bin_counts, n_bins=n_bins, strategy=strategy)


def format_ci(mean, lo, hi, decimals=3):
    fmt = f".{decimals}f"
    return f"{mean:{fmt}} [{lo:{fmt}}–{hi:{fmt}}]"


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"
