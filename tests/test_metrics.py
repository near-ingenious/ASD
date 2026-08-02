"""
tests/test_metrics.py — Unit tests for metrics and statistical utilities.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np, pytest
from asd_multimodal.utils.metrics import (
    compute_metrics, compute_sex_metrics, bootstrap_ci,
    bootstrap_all_ci, bootstrap_permutation_auc, mcnemar_test,
    mann_whitney_test, expected_calibration_error, format_ci, sig_stars)

@pytest.fixture
def rnd_preds():
    rng=np.random.default_rng(42)
    y=rng.integers(0,2,200); s=rng.uniform(0,1,200)
    return y,s

@pytest.fixture
def imb_preds():
    rng=np.random.default_rng(42)
    y=np.array([1]*100+[0]*110)
    s=np.where(y==1,rng.uniform(0.4,0.9,210),rng.uniform(0.1,0.7,210))
    return y,s

class TestComputeMetrics:
    def test_returns_all_keys(self,rnd_preds):
        y,s=rnd_preds; m=compute_metrics(y,s)
        assert {"AUC","BAC","F1","Sens","Spec","PPV","NPV","FNR","FPR","Brier","NLL"}.issubset(m)
    def test_auc_range(self,rnd_preds):
        y,s=rnd_preds; m=compute_metrics(y,s)
        assert 0<=m["AUC"]<=1
    def test_fnr_plus_sens_eq_one(self,rnd_preds):
        y,s=rnd_preds; m=compute_metrics(y,s)
        assert abs(m["FNR"]+m["Sens"]-1)<1e-6
    def test_single_class_nan(self):
        y=np.zeros(50,int); s=np.random.uniform(0,1,50)
        m=compute_metrics(y,s); assert np.isnan(m["AUC"])
    def test_threshold_lowers_sens(self,imb_preds):
        y,s=imb_preds
        m_hi=compute_metrics(y,s,threshold=0.7)
        m_lo=compute_metrics(y,s,threshold=0.3)
        assert m_lo["Sens"]>=m_hi["Sens"]

class TestSexMetrics:
    def test_has_male_female(self,rnd_preds):
        y,s=rnd_preds; sex=np.array([1]*100+[2]*100)
        m=compute_sex_metrics(y,s,sex)
        assert "Male" in m and "Female" in m
    def test_exploratory_flag(self,rnd_preds):
        y,s=rnd_preds; sex=np.array([1]*190+[2]*10)
        m=compute_sex_metrics(y,s,sex)
        assert m.get("Female",{}).get("status")=="EXPLORATORY"
    def test_gap_keys_present(self,rnd_preds):
        y,s=rnd_preds; sex=np.array([1]*100+[2]*100)
        m=compute_sex_metrics(y,s,sex)
        assert "Gap_AUC" in m and "Gap_Sens" in m

class TestBootstrapCI:
    def test_ci_contains_estimate(self,imb_preds):
        y,s=imb_preds; m=compute_metrics(y,s)
        lo,hi=bootstrap_ci(y,s,n_boot=200,seed=42)
        assert lo<=m["AUC"]<=hi
    def test_ci_ordered(self,rnd_preds):
        y,s=rnd_preds; lo,hi=bootstrap_ci(y,s,n_boot=200,seed=42)
        assert lo<hi
    def test_all_keys(self,rnd_preds):
        y,s=rnd_preds; ci=bootstrap_all_ci(y,s,n_boot=100,seed=42)
        for k in ["AUC","BAC","Sens","Spec"]:
            assert k in ci and len(ci[k])==2

class TestStatTests:
    def test_permutation_same_model(self,rnd_preds):
        y,s=rnd_preds; r=bootstrap_permutation_auc(y,s,y,s,n_boot=100,seed=42)
        assert abs(r["delta"])<0.02 and r["p_value"]>0.05
    def test_mcnemar_no_disagreements(self):
        y=np.array([0,1,0,1]); p=np.array([0,1,0,1])
        r=mcnemar_test(y,p,p); assert r["p_value"]==1.0
    def test_mann_whitney_different(self):
        x=np.array([1.,2.,3.,4.]); y2=np.array([5.,6.,7.,8.])
        r=mann_whitney_test(x,y2); assert r["p_value"]<0.05

class TestCalibration:
    def test_ece_range(self,rnd_preds):
        y,s=rnd_preds; r=expected_calibration_error(y,s,n_bins=10)
        assert 0<=r["ECE"]<=1
    def test_brier_better_model(self,imb_preds):
        y,s=imb_preds
        r_good=expected_calibration_error(y,s)
        r_rand=expected_calibration_error(y,np.full(len(y),0.5))
        assert r_good["Brier"]<=r_rand["Brier"]

class TestFormatting:
    def test_format_ci(self):
        s=format_ci(0.763,0.734,0.792)
        assert "0.763" in s and "0.734" in s
    def test_sig_stars(self):
        assert sig_stars(0.0001)=="***"
        assert sig_stars(0.03)=="*"
        assert sig_stars(0.10)=="ns"
