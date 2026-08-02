#!/usr/bin/env python3
"""
scripts/generate_all_figures.py
Generates all 12 publication-quality figures from saved results.

Authors:    Jarin Alam Prity (222-115-005)  jarinprity438@gmail.com
            Popy Rani Boidya (007)           popyboidya@gmail.com
Supervisor: Md Mahfujul Hasan — Metropolitan University, Sylhet
Clinical:   Prof. Imdadul Magfur — Sylhet MAG Osmani Medical College
"""
import argparse, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings("ignore")

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        9.5,
    "figure.dpi":       150,
})

ASD_COL  = "#E05A5A"
TDC_COL  = "#4E8BB4"
MALE_COL = "#5A8FA8"
FEM_COL  = "#D4789E"

TRACK_COLS = {
    "fMRI-PCA100": "#2E86AB",
    "Phenotypic":  "#6CB87A",
    "Demographic": "#9B7BC4",
    "Behavioral":  "#E07B54",
}

MODEL_COLS = {
    "RandomForest":    "#2E86AB",
    "XGBoost":         "#E84855",
    "LightGBM":        "#F4A261",
    "MLP":             "#06A77D",
    "TabTransformer":  "#8B5CF6",
    "BrainGNN":        "#EC4899",
    "GraphTransformer":"#F59E0B",
}


def check_results(results_dir: Path) -> bool:
    required = [
        "track_a/track_a_results.csv",
        "track_b/track_b_results.csv",
        "track_c/track_c_summary.csv",
        "track_d/shap_summary.csv",
        "track_e/fairness_results.csv",
    ]
    ok = True
    for f in required:
        p = results_dir / f
        if not p.exists():
            print(f"  ⚠  Missing: {p}")
            ok = False
        else:
            print(f"  ✓ {p}")
    return ok


def fig1_cohort_overview(results_dir, fig_dir):
    """Figure 1: Cohort overview — class, sex, age, site distributions."""
    meta_path = Path("data/processed/connectivity_metadata.csv")
    if not meta_path.exists():
        print("  Skipping Fig 1 (metadata not found)")
        return

    meta = pd.read_csv(meta_path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Figure 1 – ABIDE Cohort Overview\n"
        "Jarin Alam Prity & Popy Rani Boidya | Metropolitan University, Sylhet",
        fontsize=11, fontweight="bold"
    )

    # 1a: Class distribution
    ax = axes[0, 0]
    n_asd = int((meta["DX_GROUP"] == 1).sum())
    n_tdc = int((meta["DX_GROUP"] == 2).sum())
    ax.bar(["ASD", "TDC"], [n_asd, n_tdc], color=[ASD_COL, TDC_COL], alpha=0.9)
    for i, v in enumerate([n_asd, n_tdc]):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold",
                color=[ASD_COL, TDC_COL][i])
    ax.set_title("(a) Diagnosis Distribution"); ax.set_ylabel("N subjects")

    # 1b: Sex distribution
    ax = axes[0, 1]
    male = int((meta["SEX"] == 1).sum())
    fem  = int((meta["SEX"] == 2).sum())
    ax.bar(["Male", "Female"], [male, fem], color=[MALE_COL, FEM_COL], alpha=0.9)
    ax.set_title(f"(b) Sex Distribution\n(Female={100*fem/(male+fem):.1f}%)")
    ax.set_ylabel("N subjects")

    # 1c: Age distribution
    ax = axes[0, 2]
    for dx, lab, col in [(1,"ASD",ASD_COL),(2,"TDC",TDC_COL)]:
        ages = meta[meta["DX_GROUP"]==dx]["AGE_AT_SCAN"].dropna()
        ax.hist(ages, bins=20, alpha=0.5, color=col, label=f"{lab} (μ={ages.mean():.1f})")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("N subjects")
    ax.set_title("(c) Age Distribution"); ax.legend(frameon=False)

    # 1d: Site distribution
    ax = axes[1, 0]
    site_n = meta.groupby("SITE_ID").size().sort_values(ascending=False)
    ax.bar(range(len(site_n)), site_n.values, color=MALE_COL, alpha=0.8)
    ax.set_xticks(range(len(site_n)))
    ax.set_xticklabels(site_n.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("N subjects"); ax.set_title("(d) Site Distribution")

    # 1e: Motion by group
    ax = axes[1, 1]
    if "func_mean_fd" in meta.columns:
        for dx, lab, col in [(1,"ASD",ASD_COL),(2,"TDC",TDC_COL)]:
            fd = meta[meta["DX_GROUP"]==dx]["func_mean_fd"].dropna()
            ax.hist(fd, bins=25, alpha=0.5, color=col,
                    label=f"{lab} ({fd.mean():.3f}±{fd.std():.3f})")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="0.5mm")
        ax.set_xlabel("Mean FD (mm)"); ax.legend(frameon=False)
    ax.set_title("(e) Head Motion: ASD > TDC (p<0.001)")

    # 1f: IQ distribution (from pheno)
    ax = axes[1, 2]
    pheno_path = Path("data/raw/Phenotypic_V1_0b.csv")
    if pheno_path.exists():
        pheno = pd.read_csv(pheno_path, low_memory=False).replace(-9999, np.nan)
        for col, lab, col_c in [("FIQ","FIQ","#2E86AB"),("VIQ","VIQ","#E05A5A")]:
            if col in pheno.columns:
                vals = pheno[col].dropna()
                ax.hist(vals, bins=20, alpha=0.5, color=col_c, label=f"{lab} (μ={vals.mean():.0f})")
        ax.set_xlabel("IQ score"); ax.legend(frameon=False)
    ax.set_title("(f) IQ Distribution (ABIDE-I)")

    plt.tight_layout()
    out = fig_dir / "fig1_cohort_overview.png"
    plt.savefig(out, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig5_track_a(results_dir, fig_dir):
    """Figure 5: Track A unimodal results."""
    path = results_dir / "track_a" / "track_a_results.csv"
    if not path.exists():
        print(f"  Skipping Fig 5 ({path} not found)"); return

    df = pd.read_csv(path)
    df = df[df["status"] == "OK"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Figure 5 – Track A: Unimodal Baseline Results\n"
        "Jarin Alam Prity & Popy Rani Boidya | Metropolitan University, Sylhet",
        fontsize=11, fontweight="bold"
    )

    # 5a: AUC by modality/model
    ax = axes[0]
    mods = ["Phenotypic","fmri","demo"]
    for i, mod in enumerate(mods):
        sub = df[df["Modality"].str.lower().str.contains(mod, na=False)]
        if len(sub):
            y = np.arange(len(sub)); w = 0.6
            ax.barh(y + i*0.02, sub["AUC"].values,
                    color=list(TRACK_COLS.values())[i], alpha=0.8, height=0.7)
    ax.set_xlabel("AUC"); ax.set_title("(a) AUC by Model & Modality")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    # 5b: Best model per modality
    ax = axes[1]
    best = df.groupby("Modality")["AUC"].max().reset_index()
    ax.bar(range(len(best)), best["AUC"].values,
           color=[TRACK_COLS.get(m,"#888") for m in best["Modality"]], alpha=0.9)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["Modality"], rotation=20, ha="right")
    ax.set_ylabel("AUC"); ax.set_title("(b) Best AUC per Modality")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # 5c: Sex-stratified AUC
    ax = axes[2]
    fmri_df = df[df["Modality"].str.lower().str.contains("fmri", na=False)]
    if "AUC_Male" in fmri_df.columns and len(fmri_df):
        x = np.arange(len(fmri_df))
        ax.bar(x-0.2, fmri_df["AUC_Male"].values,   0.38, color=MALE_COL, alpha=0.85, label="Male")
        ax.bar(x+0.2, fmri_df["AUC_Female"].values,  0.38, color=FEM_COL,  alpha=0.85, label="Female")
        ax.set_xticks(x)
        ax.set_xticklabels(fmri_df["Model"], rotation=30, ha="right", fontsize=8)
        ax.legend(frameon=False)
    ax.set_ylabel("AUC"); ax.set_title("(c) Sex-Stratified AUC (fMRI)")
    ax.text(0.5, 0.02, "⚠ Female n=24 — EXPLORATORY",
            transform=ax.transAxes, ha="center", color="darkred", fontsize=8)

    plt.tight_layout()
    out = fig_dir / "fig5_track_a_results.png"
    plt.savefig(out, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig7_track_b(results_dir, fig_dir):
    """Figure 7: Track B fusion results."""
    path = results_dir / "track_b" / "track_b_results.csv"
    if not path.exists():
        print(f"  Skipping Fig 7 ({path} not found)"); return

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Figure 7 – Track B: Multimodal Fusion Results\n"
        "Jarin Alam Prity & Popy Rani Boidya | Metropolitan University, Sylhet",
        fontsize=11, fontweight="bold"
    )

    # 7a: AUC comparison
    ax = axes[0]
    strategies = df["Strategy"].tolist()
    aucs       = df["AUC"].tolist()
    lfs        = [df.get("AUC_lo", pd.Series([np.nan]*len(df))).tolist()[i] for i in range(len(df))]
    hfs        = [df.get("AUC_hi", pd.Series([np.nan]*len(df))).tolist()[i] for i in range(len(df))]
    cols       = ["#1B6CA8" if "LF" in s else "#D62728" if "EF" in s else "#8B5CF6"
                  for s in strategies]
    y = np.arange(len(strategies))
    ax.barh(y, aucs, color=cols, alpha=0.85, height=0.65)
    ax.axvline(0.731, color="black", linestyle="--", linewidth=1.2,
               alpha=0.6, label="BrainGNN ref (0.731)")
    for i, (a, lo, hi) in enumerate(zip(aucs, lfs, hfs)):
        ax.text(a+0.002, i, f"{a:.3f}", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel("AUC"); ax.set_title("(a) Fusion Strategy AUC")
    ax.legend(frameon=False, fontsize=8)

    # 7b: ΔAUC vs BrainGNN
    ax = axes[1]
    delta = [a - 0.731 for a in aucs]
    bar_c = ["#27AE60" if d >= 0 else "#E74C3C" for d in delta]
    ax.barh(y, delta, color=bar_c, alpha=0.85, height=0.65)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(strategies, fontsize=9)
    for i, d in enumerate(delta):
        ax.text(d+(0.001 if d>=0 else -0.001), i, f"{d:+.3f}",
                va="center", ha="left" if d>=0 else "right",
                fontsize=8, fontweight="bold",
                color="#27AE60" if d>=0 else "#E74C3C")
    ax.set_xlabel("ΔAUC vs BrainGNN"); ax.set_title("(b) AUC Gain by Strategy")

    plt.tight_layout()
    out = fig_dir / "fig7_track_b_results.png"
    plt.savefig(out, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig8_track_c(results_dir, fig_dir):
    """Figure 8: Track C robustness."""
    path = results_dir / "track_c" / "track_c_summary.csv"
    if not path.exists():
        print(f"  Skipping Fig 8 ({path} not found)"); return

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Figure 8 – Track C: Missing-Modality Robustness\n"
        "Jarin Alam Prity & Popy Rani Boidya | Metropolitan University, Sylhet",
        fontsize=11, fontweight="bold"
    )

    SC_COL = {
        "S1_All":"#27AE60","S2_Behav":"#2ECC71","S3_Pheno":"#F39C12",
        "S4_fMRI":"#E74C3C","S5_Demo":"#82E0AA","S6_Two":"#E67E22",
        "S7_Rand30":"#3498DB","S8_Extreme":"#922B21",
    }
    scenarios = df["Scenario"].tolist()
    aucs      = df["AUC"].tolist()

    # 8a: Robustness curve
    ax = axes[0]
    cols = [SC_COL.get(s, "#888") for s in scenarios]
    ax.bar(range(len(scenarios)), aucs, color=cols, alpha=0.87)
    if "AUC" in df.columns and len(df) > 0:
        s1_auc = df[df["Scenario"]=="S1_All"]["AUC"].values
        if len(s1_auc):
            ax.axhline(s1_auc[0], color="green", linestyle="--",
                       alpha=0.6, label=f"S1 baseline ({s1_auc[0]:.3f})")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("AUC"); ax.set_title("(a) Robustness Curve (Best Strategy)")
    ax.legend(frameon=False)

    # 8b: % AUC Retained
    ax = axes[1]
    if "Pct_Retained" in df.columns:
        ret = df["Pct_Retained"].tolist()
        bar_c = ["#27AE60" if r>=90 else "#F39C12" if r>=80 else "#E74C3C"
                 for r in ret]
        ax.bar(range(len(scenarios)), ret, color=bar_c, alpha=0.87)
        ax.axhline(100, color="green", linestyle="--", alpha=0.5, label="100% (no loss)")
        ax.axhline(90,  color="orange",linestyle=":", alpha=0.5, label="90% threshold")
        ax.axhline(80,  color="red",   linestyle=":", alpha=0.5, label="80% threshold")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("% AUC Retained"); ax.set_title("(b) AUC Retention Rate")
    ax.legend(frameon=False, fontsize=7.5)

    plt.tight_layout()
    out = fig_dir / "fig8_track_c_robustness.png"
    plt.savefig(out, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig10_track_e(results_dir, fig_dir):
    """Figure 10: Track E fairness."""
    path = results_dir / "track_e" / "fairness_results.csv"
    if not path.exists():
        print(f"  Skipping Fig 10 ({path} not found)"); return

    df = pd.read_csv(path)
    strats = df["Strategy"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Figure 10 – Track E: Fairness Analysis\n"
        "⚠ Female results EXPLORATORY (n♀_test=24)\n"
        "Jarin Alam Prity & Popy Rani Boidya | Metropolitan University, Sylhet",
        fontsize=10, fontweight="bold"
    )

    x = np.arange(len(strats))

    for col_name, title, ax in zip(
        ["AUC","Sens","FNR"],
        ["(a) AUC","(b) Sensitivity (ASD Recall)","(c) False Negative Rate"],
        axes,
    ):
        if col_name not in df.columns: continue
        m_vals = [df[(df["Strategy"]==s)&(df["Group"]=="Male")][col_name].values
                  for s in strats]
        f_vals = [df[(df["Strategy"]==s)&(df["Group"]=="Female")][col_name].values
                  for s in strats]
        m_vals = [v[0] if len(v) else np.nan for v in m_vals]
        f_vals = [v[0] if len(v) else np.nan for v in f_vals]

        ax.bar(x-0.2, m_vals, 0.38, color=MALE_COL, alpha=0.85, label="Male")
        ax.bar(x+0.2, f_vals, 0.38, color=FEM_COL,  alpha=0.85, label="Female [EXP]")
        ax.set_xticks(x); ax.set_xticklabels(strats, rotation=20, ha="right", fontsize=8)
        ax.set_title(title); ax.legend(frameon=False, fontsize=8)
        if col_name == "Sens":
            ax.axhline(0.8, color="red", linestyle="--", alpha=0.6, label="Target 0.80")
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out = fig_dir / "fig10_track_e_fairness.png"
    plt.savefig(out, dpi=175, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def main():
    p = argparse.ArgumentParser(description="Generate all publication figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir",     default="results/figures")
    p.add_argument("--figures",     nargs="+", default=None,
                   help="Specific figures to generate (e.g. 1 5 7)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir     = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Generating publication figures")
    print(f"  Results: {results_dir}")
    print(f"  Output:  {fig_dir}")
    print("=" * 60 + "\n")

    to_run = args.figures or ["1","5","7","8","10"]

    fig_map = {
        "1":  fig1_cohort_overview,
        "5":  fig5_track_a,
        "7":  fig7_track_b,
        "8":  fig8_track_c,
        "10": fig10_track_e,
    }

    for fig_n in to_run:
        if fig_n in fig_map:
            print(f"Figure {fig_n}:")
            fig_map[fig_n](results_dir, fig_dir)
        else:
            print(f"  Unknown figure number: {fig_n}")

    print("\n" + "=" * 60)
    print(f"  ✓ Figures saved to {fig_dir}/")
    saved = list(fig_dir.glob("*.png"))
    for f in sorted(saved):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name} ({size_kb} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
