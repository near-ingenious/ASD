#!/usr/bin/env bash
# scripts/download_abide.sh
# Guide to downloading ABIDE-I and ABIDE-II data
# Metropolitan University ASD Research Team
# Authors: Jarin Alam Prity & Popy Rani Boidya

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ABIDE Data Download Guide"
echo "  Metropolitan University, Sylhet, Bangladesh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DATA_DIR="data/raw"
mkdir -p "${DATA_DIR}/abide_rois_cc200"

echo ""
echo "STEP 1: Download ABIDE-I Phenotypic Data"
echo "─────────────────────────────────────────"
echo "URL: https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
echo "     Phenotypic/Phenotypic_V1_0b.csv"
echo ""
echo "  wget -P ${DATA_DIR} \\"
echo "    'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic/Phenotypic_V1_0b.csv'"
echo ""

# Attempt download
if command -v wget &>/dev/null; then
    echo "Attempting download..."
    wget -q --show-progress -P "${DATA_DIR}" \
        "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic/Phenotypic_V1_0b.csv" \
        || echo "  [Manual download required — see URL above]"
else
    echo "  wget not available. Please download manually from:"
    echo "  https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic/Phenotypic_V1_0b.csv"
fi

echo ""
echo "STEP 2: Download ABIDE-II Phenotypic Data"
echo "──────────────────────────────────────────"
echo "URL: https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
echo "     ABIDEII/Phenotypic/ABIDEII_Composite_Phenotypic.csv"
echo ""
echo "  ⚠  CRITICAL: File uses latin1 encoding and has trailing"
echo "     whitespace in 'AGE_AT_SCAN ' column name."
echo "     Pipeline handles this automatically."
echo ""

echo "STEP 3: Download ABIDE-I Preprocessed ROI Time-Series (CC200)"
echo "──────────────────────────────────────────────────────────────"
cat << 'DPARSF_INFO'
The CC200 ROI time-series (.1D files) were generated using CPAC pipeline
(filt_noglobal strategy). They are available from:

  http://preprocessed-connectomes-project.org/abide/download.html

Use the Preprocessed Connectomes Project download script:

  # Install downloader
  pip install boto3

  # Download CC200 CPAC filt_noglobal for all sites
  python -c "
import boto3, os
from botocore import UNSIGNED
from botocore.config import Config

s3  = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'fcp-indi'
prefix = 'data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/'

paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key  = obj['Key']
        dest = os.path.join('data/raw/abide_rois_cc200', os.path.basename(key))
        if not os.path.exists(dest):
            print(f'Downloading {os.path.basename(key)}...')
            s3.download_file(bucket, key, dest)
print('Done.')
  "
DPARSF_INFO

echo ""
echo "STEP 4: Verify downloaded files"
echo "───────────────────────────────"
python3 -c "
import os, pandas as pd
data_dir = 'data/raw'

checks = {
    'ABIDE-I phenotypic': 'Phenotypic_V1_0b.csv',
    'ABIDE-II phenotypic': 'ABIDEII_Composite_Phenotypic.csv',
}
for label, fn in checks.items():
    path = os.path.join(data_dir, fn)
    if os.path.exists(path):
        try:
            enc = 'latin1' if 'II' in fn else 'utf-8'
            df  = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip()
            print(f'  ✓ {label}: n={len(df)} subjects, {df.shape[1]} columns')
        except Exception as e:
            print(f'  ✗ {label}: ERROR — {e}')
    else:
        print(f'  ✗ {label}: NOT FOUND at {path}')

roi_dir = os.path.join(data_dir, 'abide_rois_cc200')
n_roi   = len([f for f in os.listdir(roi_dir) if f.endswith('.1D')]) if os.path.exists(roi_dir) else 0
status  = '✓' if n_roi > 900 else '⚠' if n_roi > 0 else '✗'
print(f'  {status} CC200 ROI files: {n_roi} .1D files found (expected ~1035)')
" 2>/dev/null || echo "  Python check failed — ensure data files are in place."

echo ""
echo "STEP 5: Build connectivity matrix"
echo "──────────────────────────────────"
echo "  python scripts/compute_connectivity.py"
echo ""
echo "  This generates:"
echo "    data/processed/connectivity_matrix.npy  (989, 19900)"
echo "    data/processed/connectivity_metadata.csv"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete. Run: bash scripts/run_all_experiments.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
