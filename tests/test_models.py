"""
tests/test_models.py — Tests for model architectures.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np, pytest, torch
from asd_multimodal.models.unimodal import (
    make_random_forest, make_lightgbm, make_mlp,
    TabTransformer, BrainGNN, GraphTransformer, SklearnWrapper)
from asd_multimodal.models.fusion import (
    LateFusionStacked, LateFusionAverage, EarlyFusion, CrossModalAttention)
from asd_multimodal.models.reconstruction import (
    BetaVAE, MaskedAutoencoder, train_vae, vae_reconstruct)
from asd_multimodal.training.missing_modality import apply_missingness, impute_missing, SCENARIOS

@pytest.fixture
def xy():
    rng=np.random.default_rng(42)
    return rng.normal(0,1,(120,25)).astype(np.float32),rng.integers(0,2,120)

@pytest.fixture
def fmri_pheno():
    rng=np.random.default_rng(42)
    return (rng.normal(0,1,(120,20)).astype(np.float32),
            rng.normal(100,15,(120,3)).astype(np.float32),
            rng.integers(0,2,120))

@pytest.fixture
def nan_data():
    rng=np.random.default_rng(42)
    X=rng.normal(0,1,(80,25)).astype(np.float32)
    X[::4,:5]=np.nan
    return X,rng.integers(0,2,80)

class TestClassical:
    @pytest.mark.parametrize("fn",[make_random_forest,make_lightgbm,make_mlp])
    def test_fit_predict(self,fn,xy):
        X,y=xy; clf=fn(seed=42); clf.fit(X[:80],y[:80])
        p=clf.predict_proba(X[80:])
        assert p.shape==(40,2) and np.allclose(p.sum(1),1,atol=1e-5)

class TestNeuralModels:
    @pytest.mark.parametrize("cls,kw",[
        (TabTransformer,{"d_model":16,"n_heads":2,"dropout":0.1}),
        (BrainGNN,{"hidden":32,"dropout":0.1}),
        (GraphTransformer,{"d_model":16,"n_heads":2,"dropout":0.1})])
    def test_forward(self,cls,kw):
        x=torch.randn(16,25); m=cls(in_dim=25,**kw)
        assert m(x).shape==(16,)

    def test_sklearn_wrapper(self,xy):
        X,y=xy
        w=SklearnWrapper(TabTransformer,{"d_model":16,"n_heads":2,"dropout":0.1},epochs=2,seed=42)
        w.fit(X[:80],y[:80])
        p=w.predict_proba(X[80:]); assert p.shape==(40,2)

class TestFusion:
    def test_lf_stacked(self,fmri_pheno):
        Xf,Xp,y=fmri_pheno
        clf=LateFusionStacked(n_estimators_fmri=50,n_estimators_pheno=50,seed=42)
        clf.fit(Xf[:80],Xp[:80],y[:80])
        p=clf.predict_proba(Xf[80:],Xp[80:])
        assert p.shape==(40,2) and np.allclose(p.sum(1),1,atol=1e-5)

    def test_lf_meta_weights(self,fmri_pheno):
        Xf,Xp,y=fmri_pheno
        clf=LateFusionStacked(seed=42); clf.fit(Xf[:80],Xp[:80],y[:80])
        assert clf.meta_weights is not None and len(clf.meta_weights)==2

    def test_early_fusion(self,fmri_pheno):
        Xf,Xp,y=fmri_pheno
        clf=EarlyFusion(seed=42); clf.fit([Xf[:80],Xp[:80]],y[:80])
        p=clf.predict_proba([Xf[80:],Xp[80:]]); assert p.shape==(40,2)

    def test_cma_forward(self):
        B=8; m=CrossModalAttention(fmri_dim=20,pheno_dim=3,demo_dim=2,d_model=32,n_heads=4,n_modalities=3)
        inputs=[torch.randn(B,20),torch.randn(B,3),torch.randn(B,2)]
        av=torch.ones(B,3); assert m(inputs,av).shape==(B,)

class TestReconstruction:
    def test_vae_forward(self):
        m=BetaVAE(obs_dim=25,latent_dim=8); x=torch.randn(16,25)
        r,mu,lv=m(x)
        assert r.shape==(16,25) and mu.shape==(16,8)

    def test_vae_reconstruct_fills_nan(self):
        X=np.random.randn(60,25).astype(np.float32)
        m=train_vae(BetaVAE(obs_dim=25,latent_dim=8),X,epochs=3,seed=42)
        Xm=X[:10].copy(); Xm[:,:5]=np.nan
        Xr=vae_reconstruct(m,Xm,X.mean(0))
        assert not np.isnan(Xr).any()
        np.testing.assert_array_almost_equal(Xr[:,5:],Xm[:,5:])

    def test_mae_forward(self):
        m=MaskedAutoencoder(obs_dim=25,hidden=64,latent_dim=16)
        x=torch.randn(8,25); r,mask=m(x)
        assert r.shape==(8,25) and mask.dtype==torch.bool

class TestMissingModality:
    def test_s1_unchanged(self,xy):
        X,_=xy; rng=np.random.default_rng(0)
        Xm,av=apply_missingness(X,"S1_All",rng)
        np.testing.assert_array_equal(Xm,X); assert av.all()

    def test_s4_zeros_fmri(self,xy):
        X,_=xy; X2=np.hstack([X,X[:,:5]]);  # 30-dim: slice 0:100 would be out of range
        # Use smaller test: check first modality masked
        rng=np.random.default_rng(0)
        from asd_multimodal.training.missing_modality import MODALITY_SLICES
        X3=np.zeros((10,105)); rng2=np.random.default_rng(0)
        Xm,av=apply_missingness(X3,"S4_fMRI",rng2)
        assert np.isnan(Xm[:,0:100]).all()
        assert not np.isnan(Xm[:,100:]).any()
        assert av[:,0].sum()==0

    def test_s7_random_fraction(self,xy):
        X,_=xy; X2=np.zeros((50,105)); rng=np.random.default_rng(0)
        Xm,_=apply_missingness(X2,"S7_Rand30",rng)
        frac=np.isnan(Xm).mean()
        assert 0.20<=frac<=0.40

    def test_impute_zero(self,nan_data):
        X,_=nan_data; Xi=impute_missing(X[20:],X[:20],"Zero")
        assert not np.isnan(Xi).any() and (Xi[np.isnan(X[20:])]==0).all()

    def test_impute_mean_no_nan(self,nan_data):
        X,_=nan_data; Xi=impute_missing(X[20:],X[:20],"Mean")
        assert not np.isnan(Xi).any()

    def test_all_scenarios(self):
        rng=np.random.default_rng(0); X=np.zeros((10,105))
        for sc in SCENARIOS:
            Xm,av=apply_missingness(X,sc,rng)
            assert Xm.shape==X.shape and av.shape==(10,3)
