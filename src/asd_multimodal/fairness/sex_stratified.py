"""
fairness/sex_stratified.py — Track E: Fairness interventions E1–E4.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh

H5: NOT SUPPORTED — data scarcity (n♀_ASD=62), not algorithm failure.
⚠  All female results EXPLORATORY (n♀_test≈24; CI≈±0.22 on AUC).
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from ..utils.metrics import compute_metrics, compute_sex_metrics
warnings.filterwarnings("ignore")

FEMALE_MIN_WARN  = 30
FEMALE_TRAIN_MIN = 100


def _warn_female(sex_test):
    n = (sex_test==2).sum()
    if n < FEMALE_MIN_WARN:
        warnings.warn(f"EXPLORATORY: n♀_test={n} < {FEMALE_MIN_WARN}. "
                      f"All female metrics unreliable (CI≈±0.22 on AUC).")


def run_e1_unified(X_train,y_train,X_test,y_test,sex_test,base_clf=None,seed=42):
    clf = base_clf or RandomForestClassifier(500,class_weight="balanced",n_jobs=-1,random_state=seed)
    clf.fit(X_train,y_train); ys=clf.predict_proba(X_test)[:,1]
    _warn_female(sex_test)
    return {"strategy":"E1_Unified","overall":compute_metrics(y_test,ys),
            "sex":compute_sex_metrics(y_test,ys,sex_test),"clf":clf,"y_score":ys}


def run_e2_sex_specific(X_train,y_train,X_test,y_test,sex_train,sex_test,seed=42):
    n_f = int((sex_train==2).sum())
    if n_f < FEMALE_TRAIN_MIN:
        warnings.warn(f"E2: n♀_train={n_f} < {FEMALE_TRAIN_MIN}. "
                      f"ABIDE-I experiments showed FNR=1.0 (catastrophic failure).")
    clfs = {}
    for sx,label in [(1,"Male"),(2,"Female")]:
        mask = sex_train==sx
        if mask.sum() < 10: continue
        n_est = 400 if sx==1 else max(50,min(200,mask.sum()*2))
        clfs[sx] = RandomForestClassifier(n_est,class_weight="balanced",
                                           n_jobs=-1,random_state=seed).fit(X_train[mask],y_train[mask])
    ys = np.full(len(y_test),0.5)
    for sx,clf in clfs.items():
        idx=sex_test==sx
        if idx.any(): ys[idx]=clf.predict_proba(X_test[idx])[:,1]
    _warn_female(sex_test)
    return {"strategy":"E2_SexSpecific","overall":compute_metrics(y_test,ys),
            "sex":compute_sex_metrics(y_test,ys,sex_test),"n_female_train":n_f,
            "y_score":ys,"warning":"Likely to fail with n♀<100"}


def run_e3_cost_sensitive(X_train,y_train,X_test,y_test,sex_train,sex_test,
                           female_asd_weight=3.0,seed=42):
    sw = np.ones(len(y_train))
    sw[(sex_train==2)&(y_train==1)] = female_asd_weight
    clf = RandomForestClassifier(500,n_jobs=-1,random_state=seed)
    clf.fit(X_train,y_train,sample_weight=sw)
    ys  = clf.predict_proba(X_test)[:,1]
    _warn_female(sex_test)
    return {"strategy":"E3_CostSensitive","overall":compute_metrics(y_test,ys),
            "sex":compute_sex_metrics(y_test,ys,sex_test),
            "female_asd_weight":female_asd_weight,"y_score":ys}


def run_e4_stratified_threshold(X_train,y_train,X_test,y_test,sex_train,sex_test,seed=42):
    clf = RandomForestClassifier(500,class_weight="balanced",n_jobs=-1,random_state=seed)
    clf.fit(X_train,y_train); ys=clf.predict_proba(X_test)[:,1]
    thrs = {}
    for sx in [1,2]:
        mask = sex_test==sx
        if mask.sum() < 5: thrs[sx]=0.5; continue
        fpr,tpr,thr = roc_curve(y_test[mask],ys[mask])
        thrs[sx]    = float(thr[np.argmax(tpr-fpr)])
    _warn_female(sex_test)
    return {"strategy":"E4_StratifiedThreshold","overall":compute_metrics(y_test,ys),
            "sex":compute_sex_metrics(y_test,ys,sex_test),
            "thresholds":{"Male":thrs.get(1,0.5),"Female":thrs.get(2,0.5)},
            "y_score":ys,"note":f"Female threshold={thrs.get(2,0.5):.3f} (vs 0.500)"}


def compile_fairness_table(results):
    rows=[]
    for strat_name,res in results.items():
        for group in ["Male","Female"]:
            gm=res.get("sex",{}).get(group,{})
            if gm: rows.append({"Strategy":strat_name,"Group":group,
                                  "N":gm.get("N",np.nan),"AUC":gm.get("AUC",np.nan),
                                  "Sens":gm.get("Sens",np.nan),"Spec":gm.get("Spec",np.nan),
                                  "FNR":gm.get("FNR",np.nan),"Status":gm.get("status","OK")})
    df = pd.DataFrame(rows)
    gaps=[]
    for strat in df["Strategy"].unique():
        sub=df[df["Strategy"]==strat].set_index("Group")
        if "Male" in sub.index and "Female" in sub.index:
            gaps.append({"Strategy":strat,
                          "Gap_AUC":sub.loc["Male","AUC"]-sub.loc["Female","AUC"],
                          "Gap_Sens":sub.loc["Male","Sens"]-sub.loc["Female","Sens"],
                          "Excess_FNR":sub.loc["Female","FNR"]-sub.loc["Male","FNR"]})
    return df, pd.DataFrame(gaps)
