"""
models/unimodal.py — Unimodal baseline models (Track A).

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Optional


def make_random_forest(seed=42, **kw):
    d = dict(n_estimators=500,max_features="sqrt",class_weight="balanced",
             n_jobs=-1,random_state=seed)
    d.update(kw); return RandomForestClassifier(**d)

def make_xgboost(y_train=None, seed=42, **kw):
    pw = float((1-y_train).sum()/y_train.sum()) if y_train is not None and y_train.sum()>0 else 1.
    d  = dict(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
              colsample_bytree=0.8,gamma=1,scale_pos_weight=pw,
              eval_metric="auc",verbosity=0,random_state=seed,n_jobs=-1)
    d.update(kw); return xgb.XGBClassifier(**d)

def make_lightgbm(seed=42, **kw):
    d = dict(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
             colsample_bytree=0.8,min_child_samples=10,class_weight="balanced",
             verbose=-1,random_state=seed,n_jobs=-1)
    d.update(kw); return lgb.LGBMClassifier(**d)

def make_mlp(seed=42, **kw):
    d = dict(hidden_layer_sizes=(256,128,64),activation="relu",
             learning_rate_init=1e-3,max_iter=300,early_stopping=True,
             validation_fraction=0.1,n_iter_no_change=15,alpha=1e-4,random_state=seed)
    d.update(kw); return MLPClassifier(**d)


class TabTransformer(nn.Module):
    def __init__(self,in_dim,d_model=64,n_heads=4,n_layers=2,dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim,d_model)
        enc = nn.TransformerEncoderLayer(d_model,n_heads,d_model*2,dropout,batch_first=True)
        self.encoder = nn.TransformerEncoder(enc,num_layers=n_layers)
        self.cls = nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,32),
                                  nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x):
        x=self.proj(x).unsqueeze(1); x=self.encoder(x).squeeze(1)
        return self.cls(x).squeeze(-1)


class BrainGNN(nn.Module):
    def __init__(self,in_dim,hidden=128,dropout=0.35):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden); self.fc2=nn.Linear(hidden,hidden)
        self.fc3=nn.Linear(hidden,64);    self.out=nn.Linear(64,1)
        self.bn1=nn.BatchNorm1d(hidden);  self.bn2=nn.BatchNorm1d(hidden)
        self.drop=nn.Dropout(dropout);    self.res=nn.Linear(in_dim,hidden)
    def forward(self,x):
        h=self.drop(torch.relu(self.bn1(self.fc1(x))))+self.res(x)
        h=self.drop(torch.relu(self.bn2(self.fc2(h))))
        return self.out(torch.relu(self.fc3(h))).squeeze(-1)


class GraphTransformer(nn.Module):
    def __init__(self,in_dim,d_model=64,n_heads=4,dropout=0.2):
        super().__init__()
        self.proj=nn.Linear(in_dim,d_model)
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(d_model)
        self.cls=nn.Sequential(nn.Linear(d_model,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x):
        x=self.proj(x).unsqueeze(1); x2,_=self.attn(x,x,x)
        x=self.norm(x+x2).squeeze(1); return self.cls(x).squeeze(-1)


class SklearnWrapper:
    """sklearn-compatible wrapper for PyTorch modules."""
    def __init__(self,module_cls,module_kwargs=None,epochs=60,lr=3e-4,
                 batch_size=64,device="cpu",seed=42):
        self.module_cls=module_cls; self.module_kwargs=module_kwargs or {}
        self.epochs=epochs; self.lr=lr; self.batch_size=batch_size
        self.device=torch.device(device); self.seed=seed; self.model_=None

    def fit(self,X,y):
        torch.manual_seed(self.seed)
        self.model_=self.module_cls(in_dim=X.shape[1],**self.module_kwargs).to(self.device)
        pos_w=torch.tensor([(1-y).sum()/max(y.sum(),1)],dtype=torch.float32).to(self.device)
        crit=nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt=torch.optim.AdamW(self.model_.parameters(),lr=self.lr,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.epochs)
        Xt=torch.tensor(X,dtype=torch.float32).to(self.device)
        yt=torch.tensor(y,dtype=torch.float32).to(self.device)
        self.model_.train()
        for _ in range(self.epochs):
            perm=torch.randperm(len(Xt))
            for s in range(0,len(Xt),self.batch_size):
                idx=perm[s:s+self.batch_size]; opt.zero_grad()
                crit(self.model_(Xt[idx]),yt[idx]).backward(); opt.step()
            sch.step()
        return self

    def predict_proba(self,X):
        self.model_.eval()
        with torch.no_grad():
            logits=self.model_(torch.tensor(X,dtype=torch.float32).to(self.device)).cpu().numpy()
        p=1/(1+np.exp(-logits)); return np.column_stack([1-p,p])

    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(int)
