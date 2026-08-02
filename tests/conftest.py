"""
tests/conftest.py — Shared pytest fixtures.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np, pandas as pd, pytest


@pytest.fixture(scope="session")
def synthetic_abide():
    rng=np.random.default_rng(42); n=200; n_asd=97
    y=np.array([1]*n_asd+[0]*(n-n_asd))
    sex=rng.choice([1,2],n,p=[0.85,0.15])
    X_fmri=rng.normal(0,1,(n,100)).astype(np.float32)
    X_fmri[y==1,:10]+=0.3
    X_pheno=np.column_stack([rng.normal(108,15,n),rng.normal(105,18,n),
                              rng.normal(110,15,n)]).astype(np.float32)
    X_pheno[y==1,1]-=8
    ages=rng.uniform(7,50,n)
    X_demo=np.column_stack([ages,sex]).astype(np.float32)
    X_full=np.hstack([X_fmri,X_pheno,X_demo]).astype(np.float32)
    return dict(X_fmri=X_fmri,X_pheno=X_pheno,X_demo=X_demo,X_full=X_full,
                y=y,sex=sex,ages=ages,n_asd=n_asd,n_tdc=n-n_asd,
                n_female=int((sex==2).sum()),n_female_asd=int(((sex==2)&(y==1)).sum()))

@pytest.fixture
def small_xy():
    rng=np.random.default_rng(0)
    return rng.normal(0,1,(100,20)).astype(np.float32),rng.integers(0,2,100)
