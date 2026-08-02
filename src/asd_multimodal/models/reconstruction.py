"""
models/reconstruction.py — VAE and MAE for missing-modality reconstruction.

Authors:    Jarin Alam Prity (222-115-005) · Popy Rani Boidya (007)
Institution: Metropolitan University, Sylhet-3104, Bangladesh

H1 finding: VAE advantage is context-specific:
  S7 (30% random dropout): ΔAUC=+0.081 vs zero-fill (p<0.001)  ✓ SUPPORTED
  S4 (structural absence):  ΔAUC=+0.000 vs zero-fill (p=1.000) ✗ NOT SUPPORTED
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple


class BetaVAE(nn.Module):
    def __init__(self, obs_dim=105, latent_dim=48, beta=0.5, hidden=128):
        super().__init__()
        self.latent_dim = latent_dim; self.beta = beta
        self.encoder  = nn.Sequential(nn.Linear(obs_dim,hidden),nn.ReLU(),
                                       nn.Linear(hidden,hidden),nn.ReLU())
        self.fc_mu    = nn.Linear(hidden, latent_dim)
        self.fc_lv    = nn.Linear(hidden, latent_dim)
        self.decoder  = nn.Sequential(nn.Linear(latent_dim,hidden),nn.ReLU(),
                                       nn.Linear(hidden,hidden),nn.ReLU(),
                                       nn.Linear(hidden,obs_dim))

    def encode(self,x): h=self.encoder(x); return self.fc_mu(h),self.fc_lv(h)
    def reparametrise(self,mu,lv): return mu+torch.exp(0.5*lv)*torch.randn_like(mu)
    def decode(self,z): return self.decoder(z)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparametrise(mu,lv)
        return self.decode(z),mu,lv
    def vae_loss(self,recon,x,mu,lv):
        return (F.mse_loss(recon,x,reduction="sum") +
                self.beta*(-0.5*(1+lv-mu**2-lv.exp()).sum())) / len(x)


def train_vae(model, X_train, epochs=80, lr=1e-3, batch=64, seed=42, device="cpu"):
    torch.manual_seed(seed)
    dev=torch.device(device); model=model.to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    Xt=torch.tensor(X_train,dtype=torch.float32).to(dev)
    model.train()
    for _ in range(epochs):
        perm=torch.randperm(len(Xt))
        for s in range(0,len(Xt),batch):
            idx=perm[s:s+batch]; opt.zero_grad()
            r,mu,lv=model(Xt[idx]); model.vae_loss(r,Xt[idx],mu,lv).backward(); opt.step()
        sch.step()
    model.eval(); return model


def vae_reconstruct(model, X_masked, train_mean, device="cpu"):
    dev=torch.device(device); model=model.to(dev); model.eval()
    X_imp=X_masked.copy().astype(np.float64)
    nan_m=np.isnan(X_imp)
    for j in range(X_imp.shape[1]):
        rows=np.where(nan_m[:,j])[0]
        if len(rows): X_imp[rows,j]=train_mean[j]
    with torch.no_grad():
        Xt=torch.tensor(X_imp,dtype=torch.float32).to(dev)
        recon,_,_=model(Xt); recon=recon.cpu().numpy()
    X_result=X_masked.copy().astype(np.float64)
    X_result[nan_m]=recon[nan_m]
    return X_result.astype(np.float32)


class MaskedAutoencoder(nn.Module):
    def __init__(self, obs_dim=105, hidden=256, latent_dim=64,
                 dropout=0.1, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio=mask_ratio
        self.encoder=nn.Sequential(nn.Linear(obs_dim,hidden),nn.GELU(),nn.Dropout(dropout),
                                    nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,hidden//2),nn.GELU(),
                                    nn.Linear(hidden//2,hidden),nn.GELU(),nn.Dropout(dropout),
                                    nn.Linear(hidden,obs_dim))

    def forward(self,x,mask=None):
        if mask is None: mask=torch.rand_like(x)<self.mask_ratio
        x_m=x.clone(); x_m[mask]=0.
        z=self.encoder(x_m); return self.decoder(z),mask

    def reconstruction_loss(self,recon,x,mask):
        return F.mse_loss(recon[mask],x[mask])


def train_mae(model, X_train, epochs=80, lr=1e-3, batch=64, seed=42, device="cpu"):
    torch.manual_seed(seed)
    dev=torch.device(device); model=model.to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    Xt=torch.tensor(X_train,dtype=torch.float32).to(dev)
    model.train()
    for _ in range(epochs):
        perm=torch.randperm(len(Xt))
        for s in range(0,len(Xt),batch):
            idx=perm[s:s+batch]; opt.zero_grad()
            recon,mask=model(Xt[idx])
            model.reconstruction_loss(recon,Xt[idx],mask).backward(); opt.step()
        sch.step()
    model.eval(); return model


def mae_reconstruct(model, X_masked, train_mean, device="cpu"):
    dev=torch.device(device); model=model.to(dev); model.eval()
    X_imp=X_masked.copy().astype(np.float64); nan_m=np.isnan(X_imp)
    for j in range(X_imp.shape[1]):
        rows=np.where(nan_m[:,j])[0]
        if len(rows): X_imp[rows,j]=train_mean[j]
    nan_t=torch.tensor(nan_m,dtype=torch.bool).to(dev)
    Xt=torch.tensor(X_imp,dtype=torch.float32).to(dev)
    with torch.no_grad():
        recon,_=model(Xt,mask=nan_t); recon=recon.cpu().numpy()
    X_result=X_masked.copy().astype(np.float64)
    X_result[nan_m]=recon[nan_m]
    return X_result.astype(np.float32)
