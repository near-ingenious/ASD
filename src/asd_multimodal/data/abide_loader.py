"""
data/abide_loader.py — ABIDE-I/II loading with documented quality fixes.

Authors:    Jarin Alam Prity (222-115-005)  jarinprity438@gmail.com
            Popy Rani Boidya (007)           popyboidya@gmail.com
Supervisor: Md Mahfujul Hasan — Metropolitan University, Sylhet
Clinical:   Prof. Imdadul Magfur — Sylhet MAG Osmani Medical College

Critical fixes applied automatically:
  1. ABIDE-II: encoding='latin1' (not UTF-8)
  2. ABIDE-II: strip trailing whitespace from all column names
  3. Both: recode -9999 → NaN
  4. MNAR: ADOS/ADI-R flagged (not imputed for TDC)
"""
from __future__ import annotations
import re, warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ABIDE1_MISSING = -9999
MNAR_COLS = [
    "ADOS_TOTAL","ADOS_COMM","ADOS_SOCIAL","ADOS_STEREO_BEHAV",
    "ADOS_GOTHAM_TOTAL","ADI_R_SOCIAL_TOTAL_A","ADI_R_VERBAL_TOTAL_BV",
    "ADI_RRB_TOTAL_C","ADI_R_ONSET_TOTAL_D","SCQ_TOTAL",
]


def load_abide1_phenotypic(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).replace(ABIDE1_MISSING, np.nan)
    if "DX_GROUP" in df.columns:
        df["DX_BIN"] = (df["DX_GROUP"] == 1).astype(int)
    for col in MNAR_COLS:
        if col in df.columns:
            df[f"{col}_PRESENT"] = df[col].notna().astype(int)
    print(f"[ABIDE-I] {len(df)} subjects | "
          f"ASD={int((df['DX_GROUP']==1).sum())} "
          f"TDC={int((df['DX_GROUP']==2).sum())} | "
          f"Male={(df['SEX']==1).sum()} Female={(df['SEX']==2).sum()}")
    return df


def load_abide2_phenotypic(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()          # Fix: trailing whitespace
    df = df.replace(ABIDE1_MISSING, np.nan)
    if "DX_GROUP" in df.columns:
        df["DX_BIN"] = (df["DX_GROUP"] == 1).astype(int)
    print(f"[ABIDE-II] {len(df)} subjects | "
          f"ASD={int((df['DX_GROUP']==1).sum())} "
          f"TDC={int((df['DX_GROUP']==2).sum())} | "
          f"{df.shape[1]} columns")
    return df


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def load_roi_timeseries(path: Union[str, Path]) -> Optional[np.ndarray]:
    try:
        ts = np.loadtxt(path)
        return ts.reshape(-1, 1) if ts.ndim == 1 else ts.astype(np.float32)
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None


def compute_pearson_fc(timeseries: np.ndarray,
                        fisher_transform: bool = True) -> Optional[np.ndarray]:
    n_rois = timeseries.shape[1]
    if n_rois < 2: return None
    corr = np.corrcoef(timeseries.T)
    np.fill_diagonal(corr, 0.0)
    if fisher_transform:
        corr = fisher_z(corr)
    idx = np.triu_indices(n_rois, k=1)
    return corr[idx].astype(np.float32)


def build_connectivity_matrix(roi_dir, phenotypic_df, subject_col="SUB_ID",
                               roi_suffix="_rois_cc200.1D",
                               fisher_transform=True, min_timepoints=78):
    roi_dir = Path(roi_dir)
    rows, meta = [], []
    for _, subj_row in phenotypic_df.iterrows():
        subj_id = str(int(subj_row[subject_col]))
        site_id = str(subj_row.get("SITE_ID", "UNKNOWN"))
        pattern = re.compile(
            rf"{re.escape(site_id)}.*{re.escape(subj_id)}.*{re.escape(roi_suffix)}",
            re.IGNORECASE)
        matches = [f for f in roi_dir.glob("*.1D") if pattern.search(f.name)]
        if not matches: continue
        ts = load_roi_timeseries(matches[0])
        if ts is None or ts.shape[0] < min_timepoints: continue
        fc = compute_pearson_fc(ts, fisher_transform)
        if fc is None: continue
        rows.append(fc)
        meta.append({"SUB_ID": subj_row[subject_col], "SITE_ID": site_id,
                     "DX_GROUP": subj_row.get("DX_GROUP", np.nan),
                     "AGE_AT_SCAN": subj_row.get("AGE_AT_SCAN", np.nan),
                     "SEX": subj_row.get("SEX", np.nan),
                     "n_timepoints": ts.shape[0], "file": matches[0].name})
    X_fc    = np.stack(rows, axis=0)
    meta_df = pd.DataFrame(meta)
    meta_df["func_mean_fd"] = np.nan
    print(f"[ConnMatrix] {X_fc.shape[0]} subjects × {X_fc.shape[1]} features | "
          f"ASD={(meta_df['DX_GROUP']==1).sum()} TDC={(meta_df['DX_GROUP']==2).sum()}")
    return X_fc.astype(np.float32), meta_df


def flag_zero_variance_subjects(X, threshold=1e-10):
    return np.var(X, axis=1) < threshold


def check_motion_confound(metadata_df, fd_col="func_mean_fd", dx_col="DX_GROUP"):
    from scipy.stats import mannwhitneyu
    if fd_col not in metadata_df.columns:
        return {"status": "FD column not found"}
    df       = metadata_df.dropna(subset=[fd_col, dx_col])
    asd_fd   = df[df[dx_col]==1][fd_col].values
    tdc_fd   = df[df[dx_col]==2][fd_col].values
    u, p     = mannwhitneyu(asd_fd, tdc_fd, alternative="two-sided")
    return dict(ASD_mean_FD=float(asd_fd.mean()), ASD_std_FD=float(asd_fd.std()),
                TDC_mean_FD=float(tdc_fd.mean()), TDC_std_FD=float(tdc_fd.std()),
                U_stat=float(u), p_value=float(p), significant=bool(p<0.05),
                recommendation=("Include mean FD as ComBat covariate."
                                if p < 0.05 else "No significant motion confound."))
