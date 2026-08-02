"""
data/preprocessing.py — Within-fold preprocessing (no leakage).

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
from __future__ import annotations
import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def make_imputer(strategy: str = "mean", **kwargs):
    if strategy == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    elif strategy == "knn":
        return KNNImputer(n_neighbors=kwargs.get("n_neighbors", 5))
    elif strategy == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        return IterativeImputer(random_state=kwargs.get("seed", 42),
                                max_iter=kwargs.get("max_iter", 5), tol=1e-2)
    else:
        return SimpleImputer(strategy=strategy)


class ModalityPreprocessor:
    """Within-fold preprocessing: impute → scale → PCA (all fit on training only)."""
    def __init__(self, impute_strategy="mean", n_pca=None, scale=True, seed=42):
        self.impute_strategy = impute_strategy
        self.n_pca  = n_pca
        self.scale  = scale
        self.seed   = seed
        self._imputer = make_imputer(impute_strategy, seed=seed)
        self._scaler  = StandardScaler() if scale else None
        self._pca     = (PCA(n_components=n_pca, random_state=seed)
                         if n_pca is not None else None)
        self._fitted  = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._imputer.fit_transform(X)
        if self._scaler:  X = self._scaler.fit_transform(X)
        if self._pca:
            n_comp = min(self.n_pca, X.shape[0]-1, X.shape[1])
            self._pca.set_params(n_components=n_comp)
            X = self._pca.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        X = self._imputer.transform(X)
        if self._scaler: X = self._scaler.transform(X)
        if self._pca:    X = self._pca.transform(X)
        return X

    @property
    def pca_variance_explained(self):
        if self._pca and hasattr(self._pca, "explained_variance_ratio_"):
            return self._pca.explained_variance_ratio_
        return None


class MultimodalPreprocessor:
    """
    Coordinates preprocessing across fMRI, Pheno, Demo modalities.
    Slice map after fit_transform:
      fMRI  → [0 : fmri_pca]
      Pheno → [fmri_pca : fmri_pca+3]
      Demo  → [fmri_pca+3 : fmri_pca+5]
    """
    def __init__(self, fmri_pca=100, impute_pheno="median",
                 include_behav=False, seed=42):
        self.fmri_pca      = fmri_pca
        self.include_behav = include_behav
        self.seed          = seed
        self._fmri_prep  = ModalityPreprocessor("mean",   fmri_pca, True, seed)
        self._pheno_prep = ModalityPreprocessor(impute_pheno, None, True, seed)
        self._demo_prep  = ModalityPreprocessor("median", None,     True, seed)
        self.slice_map: dict = {}
        self._fitted = False

    def fit_transform(self, X_fmri, X_pheno, X_demo, X_behav=None):
        Xf = self._fmri_prep.fit_transform(X_fmri)
        Xp = self._pheno_prep.fit_transform(X_pheno)
        Xd = self._demo_prep.fit_transform(X_demo)
        parts, ptr = [Xf, Xp, Xd], 0
        for name, arr in [("fMRI",Xf),("Pheno",Xp),("Demo",Xd)]:
            self.slice_map[name] = (ptr, ptr+arr.shape[1])
            ptr += arr.shape[1]
        if self.include_behav and X_behav is not None:
            from sklearn.impute import SimpleImputer
            Xb = StandardScaler().fit_transform(
                SimpleImputer(strategy="constant",fill_value=0.).fit_transform(X_behav))
            parts.append(Xb)
            self.slice_map["Behav"] = (ptr, ptr+Xb.shape[1])
        self._fitted = True
        return np.hstack(parts).astype(np.float32)

    def transform(self, X_fmri, X_pheno, X_demo, X_behav=None):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        parts = [self._fmri_prep.transform(X_fmri),
                 self._pheno_prep.transform(X_pheno),
                 self._demo_prep.transform(X_demo)]
        if self.include_behav and X_behav is not None:
            parts.append(X_behav)
        return np.hstack(parts).astype(np.float32)

    def get_slice(self, modality: str) -> Tuple[int, int]:
        return self.slice_map.get(modality, (None, None))
