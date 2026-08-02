#!/usr/bin/env bash
# scripts/run_all_experiments.sh
# Full experimental pipeline — Tracks A through E
#
# Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
# Supervisor: Md Mahfujul Hasan, Metropolitan University, Sylhet
# Clinical:   Prof. Imdadul Magfur, Sylhet MAG Osmani Medical College
#
# Usage:
#   bash scripts/run_all_experiments.sh              # full run
#   bash scripts/run_all_experiments.sh --track a    # single track
#   bash scripts/run_all_experiments.sh --fast        # reduced CV for speed

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TRACK="all"
CONFIG="configs/default_config.yaml"
SEED=42
FAST=false
START_TIME=$(date +%s)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --track)   TRACK="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --seed)    SEED="$2";   shift 2 ;;
        --fast)    FAST=true;   shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SPLITS=10
SPLITS_NEURAL=5
[[ "$FAST" == "true" ]] && SPLITS=3 && SPLITS_NEURAL=3

# ── Header ───────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Robust ASD Diagnosis Under Missing Clinical Modalities"
echo "  Metropolitan University, Department of CSE, Sylhet-3104"
echo ""
echo "  Authors:    Jarin Alam Prity (222-115-005)"
echo "              Popy Rani Boidya (007)"
echo "  Supervisor: Md Mahfujul Hasan"
echo "  Clinical:   Prof. Imdadul Magfur"
echo ""
echo "  Config:  ${CONFIG}"
echo "  Seed:    ${SEED}"
echo "  Splits:  ${SPLITS} (classical) / ${SPLITS_NEURAL} (neural)"
echo "  Track:   ${TRACK}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "── Preflight checks ──"
python3 -c "import asd_multimodal; print(f'  ✓ Package: v{asd_multimodal.__version__}')"

for f in \
    "data/processed/connectivity_matrix.npy" \
    "data/processed/connectivity_metadata.csv" \
    "data/raw/Phenotypic_V1_0b.csv"; do
    [[ -f "$f" ]] && echo "  ✓ $f" || { echo "  ✗ MISSING: $f"; echo "    Run: bash scripts/download_abide.sh"; exit 1; }
done

mkdir -p results/{track_a,track_b,track_c,track_d,track_e,figures}
echo "  ✓ Results directories ready"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
run_if() {
    local track=$1; shift
    if [[ "$TRACK" == "all" || "$TRACK" == "$track" ]]; then
        echo "══════════════════════════════════════════════"
        echo "  TRACK ${track^^}: $*"
        echo "══════════════════════════════════════════════"
        "$@"
        echo ""
    fi
}

# ── Track A: Unimodal Baselines ───────────────────────────────────────────────
run_if a python3 experiments/track_a_unimodal.py \
    --config  "${CONFIG}" \
    --n-splits "${SPLITS}" \
    --seed    "${SEED}"

# ── Track B: Multimodal Fusion ────────────────────────────────────────────────
run_if b python3 experiments/track_b_fusion.py \
    --config "${CONFIG}" \
    --seed   "${SEED}"

# ── Track C: Missing-Modality Robustness ──────────────────────────────────────
run_if c python3 experiments/track_c_missing.py \
    --config "${CONFIG}" \
    --seed   "${SEED}" \
    $([[ "$FAST" == "true" ]] && echo "--no-vae" || echo "")

# ── Track D: Explainability ───────────────────────────────────────────────────
run_if d python3 experiments/track_d_explain.py \
    --fmri-pca 100 \
    --seed "${SEED}"

# ── Track E: Fairness ─────────────────────────────────────────────────────────
run_if e python3 experiments/track_e_fairness.py \
    --fmri-pca 100 \
    --seed "${SEED}"

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Pipeline complete in ${MINS}m ${SECS}s"
echo ""
echo "  Output files:"
find results/ -name "*.csv" 2>/dev/null | sort | while read f; do
    echo "    $f"
done
echo ""
echo "  Key result files:"
echo "    results/track_a/track_a_results.csv"
echo "    results/track_b/track_b_results.csv"
echo "    results/track_c/track_c_summary.csv"
echo "    results/track_d/shap_summary.csv"
echo "    results/track_e/fairness_results.csv"
echo ""
echo "  To generate figures:"
echo "    python scripts/generate_all_figures.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
