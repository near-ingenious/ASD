#!/usr/bin/env python3
"""
scripts/compute_connectivity.py
Builds (989, 19900) Fisher-z Pearson connectivity matrix from CC200 .1D files.

Authors:    Jarin Alam Prity (222-115-005)  jarinprity438@gmail.com
            Popy Rani Boidya (007)           popyboidya@gmail.com
Supervisor: Md Mahfujul Hasan — Metropolitan University, Sylhet
Clinical:   Prof. Imdadul Magfur — Sylhet MAG Osmani Medical College

Usage:
    python scripts/compute_connectivity.py
    python scripts/compute_connectivity.py --roi-dir data/raw/abide_rois_cc200
    python scripts/compute_connectivity.py --max-subjects 50   # quick test
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asd_multimodal.data.abide_loader import (
    load_abide1_phenotypic,
    build_connectivity_matrix,
    check_motion_confound,
    flag_zero_variance_subjects,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build ABIDE-I CC200 functional connectivity matrix"
    )
    p.add_argument("--pheno-path",   default="data/raw/Phenotypic_V1_0b.csv")
    p.add_argument("--roi-dir",      default="data/raw/abide_rois_cc200")
    p.add_argument("--out-dir",      default="data/processed")
    p.add_argument("--min-timepoints", type=int, default=78)
    p.add_argument("--max-subjects",   type=int, default=None,
                   help="Limit subjects (for quick test runs)")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ABIDE-I CC200 Connectivity Matrix Builder")
    print("  Metropolitan University, Sylhet-3104")
    print("=" * 60)

    # ── Load phenotypic data ──────────────────────────────────────────────────
    print(f"\nLoading phenotypic data from {args.pheno_path}...")
    df_pheno = load_abide1_phenotypic(args.pheno_path)

    if args.max_subjects:
        df_pheno = df_pheno.head(args.max_subjects)
        print(f"  [TEST MODE] Limited to {args.max_subjects} subjects")

    # ── Build connectivity matrix ─────────────────────────────────────────────
    print(f"\nBuilding connectivity matrix from {args.roi_dir}...")
    print(f"  Min timepoints: {args.min_timepoints}")

    roi_dir = Path(args.roi_dir)
    if not roi_dir.exists():
        print(f"\nERROR: ROI directory not found: {roi_dir}")
        print("  Run: bash scripts/download_abide.sh")
        sys.exit(1)

    n_1d = len(list(roi_dir.glob("*.1D")))
    print(f"  Found {n_1d} .1D files in {roi_dir}")
    if n_1d == 0:
        print("  No .1D files found. Download ROI files first.")
        sys.exit(1)

    X_fc, meta_df = build_connectivity_matrix(
        roi_dir       = roi_dir,
        phenotypic_df = df_pheno,
        subject_col   = "SUB_ID",
        roi_suffix    = "_rois_cc200.1D",
        fisher_transform = True,
        min_timepoints   = args.min_timepoints,
    )

    # ── Quality control ───────────────────────────────────────────────────────
    print("\nRunning quality control...")

    # Flag zero-variance subjects
    zv_mask = flag_zero_variance_subjects(X_fc)
    n_zv    = zv_mask.sum()
    if n_zv > 0:
        print(f"  ⚠  Excluding {n_zv} subjects with zero-variance ROIs")
        X_fc   = X_fc[~zv_mask]
        meta_df= meta_df[~zv_mask].reset_index(drop=True)

    print(f"  Final matrix: {X_fc.shape[0]} subjects × {X_fc.shape[1]} features")
    print(f"  ASD: {(meta_df['DX_GROUP']==1).sum()}  "
          f"TDC: {(meta_df['DX_GROUP']==2).sum()}")

    # Check motion confound
    print("\nMotion confound analysis:")
    if "func_mean_fd" in meta_df.columns:
        motion = check_motion_confound(meta_df)
        print(f"  ASD FD: {motion['ASD_mean_FD']:.3f} ± {motion['ASD_std_FD']:.3f}")
        print(f"  TDC FD: {motion['TDC_mean_FD']:.3f} ± {motion['TDC_std_FD']:.3f}")
        print(f"  p-value: {motion['p_value']:.4f}  "
              f"({'SIGNIFICANT ⚠' if motion['significant'] else 'ns'})")
        print(f"  Recommendation: {motion['recommendation']}")
    else:
        print("  func_mean_fd not available in metadata")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "connectivity_matrix.npy"
    meta_path   = out_dir / "connectivity_metadata.csv"

    np.save(matrix_path, X_fc)
    meta_df.to_csv(meta_path, index=False)

    print(f"\n✓ Saved:")
    print(f"  Matrix:   {matrix_path}  "
          f"({X_fc.nbytes / 1024**3:.2f} GB)")
    print(f"  Metadata: {meta_path}")

    # ── Validation ────────────────────────────────────────────────────────────
    print("\nValidation:")
    X_reload = np.load(matrix_path)
    assert X_reload.shape == X_fc.shape, "Shape mismatch after reload!"
    nan_pct = 100 * np.isnan(X_reload).mean()
    print(f"  Shape:   {X_reload.shape}  ✓")
    print(f"  NaN:     {nan_pct:.2f}%")
    print(f"  Range:   [{X_reload.min():.3f}, {X_reload.max():.3f}]")
    print(f"  Mean:    {X_reload.mean():.4f}")
    print(f"  Std:     {X_reload.std():.4f}")

    print("\n" + "=" * 60)
    print("  Connectivity matrix ready.")
    print("  Next step: python experiments/track_a_unimodal.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
