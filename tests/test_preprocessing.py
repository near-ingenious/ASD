"""
tests/test_preprocessing.py — Tests for data preprocessing.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np, pytest
from asd_multimodal.data.preprocessing import (
    ModalityPreprocessor, MultimodalPreprocessor, make_imputer)

@pytest.fixture
def fmri():
    rng=np.random.default_rng(42); X=rng.normal(0,1,(100,200)).astype(np.float32)
    X[::5,::20]=np.nan; return X

@pytest.fixture
def pheno():
    rng=np.random.default_rng(42); X=rng.normal(100,15,(100,3)).astype(np.float32)
    X[::7,1]=np.nan; return X

@pytest.fixture
def demo():
    rng=np.random.default_rng(42)
    return np.column_stack([rng.uniform(7,50,100),rng.choice([1,2],100)]).astype(np.float32)

class TestMakeImputer:
    def test_zero(self):
        imp=make_imputer("zero"); X=np.array([[1.,np.nan],[np.nan,2.]])
        out=imp.fit_transform(X); assert out[0,1]==0. and out[1,0]==0.
    def test_mean(self):
        imp=make_imputer("mean"); X=np.array([[1.,2.],[3.,np.nan],[5.,4.]])
        out=imp.fit_transform(X); assert abs(out[1,1]-3.)<1e-5
    def test_knn(self):
        imp=make_imputer("knn",n_neighbors=2)
        X=np.array([[1.,2.],[1.1,np.nan],[0.9,1.9]])
        out=imp.fit_transform(X); assert not np.isnan(out).any()

class TestModalityPreprocessor:
    def test_no_leak(self,fmri):
        prep=ModalityPreprocessor(n_pca=10,scale=True,seed=42)
        Xtr=prep.fit_transform(fmri[:80]); Xte=prep.transform(fmri[80:])
        assert Xtr.shape==(80,10) and Xte.shape==(20,10)
        assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
    def test_raises_before_fit(self,fmri):
        prep=ModalityPreprocessor(n_pca=10)
        with pytest.raises(RuntimeError): prep.transform(fmri)
    def test_no_pca_preserves_features(self,pheno):
        prep=ModalityPreprocessor(n_pca=None,scale=True,impute_strategy="median")
        out=prep.fit_transform(pheno[:80])
        assert out.shape[1]==3 and not np.isnan(out).any()
    def test_variance_explained(self,fmri):
        prep=ModalityPreprocessor(n_pca=20); prep.fit_transform(fmri[:80])
        var=prep.pca_variance_explained
        assert var is not None and len(var)==20 and var.sum()<=1.+1e-6
    def test_reproducible(self,pheno):
        p1=ModalityPreprocessor(n_pca=None,scale=True,seed=42)
        p2=ModalityPreprocessor(n_pca=None,scale=True,seed=42)
        np.testing.assert_array_almost_equal(p1.fit_transform(pheno[:80]),
                                              p2.fit_transform(pheno[:80]))

class TestMultimodalPreprocessor:
    def test_concatenates(self,fmri,pheno,demo):
        prep=MultimodalPreprocessor(fmri_pca=20,seed=42)
        out=prep.fit_transform(fmri[:80],pheno[:80],demo[:80])
        assert out.shape==(80,25)  # 20+3+2
    def test_no_nan_after_transform(self,fmri,pheno,demo):
        prep=MultimodalPreprocessor(fmri_pca=20,seed=42)
        prep.fit_transform(fmri[:80],pheno[:80],demo[:80])
        out=prep.transform(fmri[80:],pheno[80:],demo[80:])
        assert not np.isnan(out).any()
    def test_slice_map(self,fmri,pheno,demo):
        prep=MultimodalPreprocessor(fmri_pca=50,seed=42)
        prep.fit_transform(fmri[:80],pheno[:80],demo[:80])
        assert prep.get_slice("fMRI")==(0,50)
        assert prep.get_slice("Pheno")==(50,53)
        assert prep.get_slice("Demo")==(53,55)
    def test_raises_before_fit(self,fmri,pheno,demo):
        prep=MultimodalPreprocessor(fmri_pca=10)
        with pytest.raises(RuntimeError): prep.transform(fmri[80:],pheno[80:],demo[80:])
