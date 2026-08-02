"""
models/fusion.py — Multimodal fusion strategies (Track B).

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh

Best result: LF Stacked AUC=0.763 [0.734–0.792], p<0.001 vs BrainGNN.
H3 rejected: CMA (0.711) < LF Stacked (p=0.003) — honest negative.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import List, Optional


class LateFusionStacked:
    """Best system: AUC=0.763. Meta-learner over modality-specific RF experts."""
    def __init__(self,n_estimators_fmri=200,n_estimators_pheno=100,
                 meta_C=1.0,cv_meta=3,seed=42):
        self.seed=seed; self.cv_meta=cv_meta
        self._rf_fmri  = RandomForestClassifier(n_estimators_fmri,class_weight="balanced",n_jobs=-1,random_state=seed)
        self._rf_pheno = RandomForestClassifier(n_estimators_pheno,class_weight="balanced",n_jobs=-1,random_state=seed)
        self._meta     = LogisticRegression(C=meta_C,random_state=seed,max_iter=300)
        self._fitted   = False

    def fit(self,X_fmri,X_pheno,y):
        oof_f = cross_val_predict(RandomForestClassifier(self._rf_fmri.n_estimators,class_weight="balanced",n_jobs=-1,random_state=self.seed),
                                   X_fmri,y,cv=self.cv_meta,method="predict_proba")[:,1]
        oof_p = cross_val_predict(RandomForestClassifier(self._rf_pheno.n_estimators,class_weight="balanced",n_jobs=-1,random_state=self.seed),
                                   X_pheno,y,cv=self.cv_meta,method="predict_proba")[:,1]
        self._meta.fit(np.column_stack([oof_f,oof_p]),y)
        self._rf_fmri.fit(X_fmri,y); self._rf_pheno.fit(X_pheno,y)
        self._fitted=True; return self

    def predict_proba(self,X_fmri,X_pheno):
        if not self._fitted: raise RuntimeError("Call fit() first.")
        ps=np.column_stack([self._rf_fmri.predict_proba(X_fmri)[:,1],
                             self._rf_pheno.predict_proba(X_pheno)[:,1]])
        return self._meta.predict_proba(ps)

    def predict(self,X_fmri,X_pheno):
        return (self.predict_proba(X_fmri,X_pheno)[:,1]>=0.5).astype(int)

    @property
    def meta_weights(self):
        return self._meta.coef_[0] if self._fitted else None


class LateFusionAverage:
    def __init__(self,weights=None,seed=42):
        self.weights=weights; self.seed=seed; self._clfs=[]
    def fit(self,Xs,y):
        self._clfs=[RandomForestClassifier(200,class_weight="balanced",n_jobs=-1,random_state=self.seed).fit(X,y) for X in Xs]
        return self
    def predict_proba(self,Xs):
        P=np.array([c.predict_proba(X)[:,1] for c,X in zip(self._clfs,Xs)])
        w=np.array(self.weights) if self.weights else np.ones(len(Xs))/len(Xs)
        s=P.T@w; return np.column_stack([1-s,s])


class EarlyFusion:
    def __init__(self,base_clf=None,seed=42):
        self._clf=base_clf or RandomForestClassifier(200,class_weight="balanced",n_jobs=-1,random_state=seed)
    def fit(self,Xs,y):   self._clf.fit(np.hstack(Xs),y); return self
    def predict_proba(self,Xs): return self._clf.predict_proba(np.hstack(Xs))
    def predict(self,Xs): return self._clf.predict(np.hstack(Xs))


class CrossModalAttention(nn.Module):
    """
    CMA with availability-aware gating.
    H3 result: REJECTED — inferior to LF Stacked (p=0.003).
    Root cause: 3-dim Pheno insufficient for meaningful attention.
    """
    def __init__(self,fmri_dim=100,pheno_dim=3,demo_dim=2,
                 d_model=64,n_heads=4,n_modalities=3,dropout=0.2):
        super().__init__()
        dims=[fmri_dim,pheno_dim,demo_dim][:n_modalities]
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(d,d_model),nn.LayerNorm(d_model)) for d in dims])
        self.cross_attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(d_model)
        self.gate=nn.Sequential(nn.Linear(n_modalities,16),nn.ReLU(),nn.Linear(16,n_modalities),nn.Sigmoid())
        self.cls=nn.Sequential(nn.Linear(d_model,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
        self.n_modalities=n_modalities

    def forward(self,modality_inputs:List[torch.Tensor],availability:torch.Tensor):
        tokens=torch.stack([p(x) for p,x in zip(self.projs,modality_inputs)],dim=1)
        gates=self.gate(availability).unsqueeze(-1); tokens=tokens*gates
        out,_=self.cross_attn(tokens,tokens,tokens); tokens=self.norm(tokens+out)
        w=(availability/(availability.sum(dim=1,keepdim=True)+1e-9)).unsqueeze(-1)
        return self.cls((tokens*w).sum(dim=1)).squeeze(-1)


class CMAWrapper:
    """sklearn-style wrapper for CrossModalAttention."""
    def __init__(self,fmri_dim=100,pheno_dim=3,demo_dim=2,d_model=64,
                 n_heads=4,dropout=0.2,epochs=60,lr=3e-4,batch_size=64,
                 device="cpu",seed=42):
        self.mkw=dict(fmri_dim=fmri_dim,pheno_dim=pheno_dim,demo_dim=demo_dim,
                      d_model=d_model,n_heads=n_heads,dropout=dropout,n_modalities=3)
        self.epochs=epochs; self.lr=lr; self.bs=batch_size
        self.dev=torch.device(device); self.seed=seed; self._model=None

    def _split(self,X):
        f=torch.tensor(X[:,  :100],  dtype=torch.float32).to(self.dev)
        p=torch.tensor(X[:,100:103],dtype=torch.float32).to(self.dev)
        d=torch.tensor(X[:,103:105],dtype=torch.float32).to(self.dev)
        return [f,p,d], torch.ones(len(X),3).to(self.dev)

    def fit(self,X,y):
        torch.manual_seed(self.seed)
        self._model=CrossModalAttention(**self.mkw).to(self.dev)
        pw=torch.tensor([(1-y).sum()/max(y.sum(),1)],dtype=torch.float32).to(self.dev)
        crit=nn.BCEWithLogitsLoss(pos_weight=pw)
        opt=torch.optim.AdamW(self._model.parameters(),lr=self.lr,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.epochs)
        inputs,av=self._split(X); yt=torch.tensor(y,dtype=torch.float32).to(self.dev)
        self._model.train()
        for _ in range(self.epochs):
            perm=torch.randperm(len(yt))
            for s in range(0,len(yt),self.bs):
                idx=perm[s:s+self.bs]; opt.zero_grad()
                crit(self._model([m[idx] for m in inputs],av[idx]),yt[idx]).backward(); opt.step()
            sch.step()
        return self

    def predict_proba(self,X):
        self._model.eval()
        inputs,av=self._split(X)
        with torch.no_grad():
            logits=self._model(inputs,av).cpu().numpy()
        p=1/(1+np.exp(-logits)); return np.column_stack([1-p,p])

    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(int)
