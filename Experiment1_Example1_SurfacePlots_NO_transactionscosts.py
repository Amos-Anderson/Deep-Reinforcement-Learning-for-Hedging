#!/usr/bin/env python3
"""
plot_delta_surfaces.py

Compute and render three 3D‐surface plots over the 30‐day horizon:
  1. Model hedge Δ(t,s) (analytic Black‐Scholes)
  2. Deep‐hedge Δ(t,s)  (learned policy under CVaR@50%)
  3. Difference Δ_DH − Δ_Model

Usage:
    python -m deephedging.plot_delta_surfaces
"""
import os
# ── 0) Force CPU-only, disable XLA/JIT ─────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"]       = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"]          = "--xla_force_host_platform_device_count=1"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.optimizer.set_jit(False)
# ── 1) DYNAPLOT → CANVAS MODE ───────────────────────────────────────────
from cdxbasics.dynaplot import DynamicFig
DynamicFig.MODE = "canvas"
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

from cdxbasics.config          import Config
from deephedging.world         import SimpleWorld_Spot_ATM
from deephedging.gym           import VanillaDeepHedgingGym
from deephedging.trainer       import train

# ── 1) first train/load your deep‐hedge under CVaR@50% ───────────────────
cfg = Config()
# … your Heston/no‐TC setup exactly as in train_cvar_prices.py …
cfg.update({
    "r":0.0, "rvol":0.2, "volvol_rvol":0.3, "meanrev_rvol":2.0,
    "corr_vs":-0.7, "cost_s":0.0, "cost_v":0.0, "cost_p":0.0,
    "dt":1.0/250, "steps":60, "samples":10000, "no_stoch_drift":True
})
cfg.train.epochs = 100
cfg.train.batch_size = 2048
cfg.train.caching_mode = "off"
cfg.update({"monetary_utility":"cvar@0.5"})

# train (will reuse cache if present)
world   = SimpleWorld_Spot_ATM(cfg)
val_w   = world.clone()
gym     = VanillaDeepHedgingGym(cfg)
train(gym, world, val_w, config=cfg)

# simulate once to grab spots & actions
out     = gym.predict(world.tf_data)
# spots: [N, steps+1], actions: [N, steps, action_dims]
spots   = world.details.spot_all
#actions = out["actions"][:,:,0]   # delta‐component
a = out["actions"]
if isinstance(a, tf.Tensor):
    actions = a.numpy()[:,:,0]
else:
    actions = a[:,:,0]


# ── 2) build time & spot‐bins grids ─────────────────────────────────────
steps      = cfg["steps"]
times      = np.arange(steps)              # 0 … 59
T_total    = steps * cfg["dt"]
T_grid     = T_total - times * cfg["dt"]   # time‐to‐maturity in years

# choose 50 spot bins over the simulated range
s_min, s_max = spots.min(), spots.max()
bins         = np.linspace(s_min, s_max, 50)
centers      = 0.5*(bins[:-1] + bins[1:])

# analytic BS delta
def bs_delta(S, K, T, r, sigma):
    T = np.clip(T, 1e-6, None)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1)

# prepare surfaces
model_surf = np.zeros((steps, len(centers)))
dh_surf    = np.zeros_like(model_surf)

# fill in
for t in range(steps):
    model_surf[t] = bs_delta(centers, 1.0, T_grid[t], cfg["r"], cfg["rvol"])
    # bin the spot paths at time t
    idx = np.digitize(spots[:,t], bins)-1
    for i in range(len(centers)):
        mask = (idx==i)
        dh_surf[t,i] = actions[mask,t].mean() if mask.any() else np.nan

# mesh for plotting
T_mesh, S_mesh = np.meshgrid(times, centers, indexing='ij')

# ── 3) plot each surface ────────────────────────────────────────────────
def make_3d(fig, surf, zlabel, title):
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_mesh, S_mesh, surf, cmap="viridis", edgecolor='none')
    ax.set_xlabel("Time step")
    ax.set_ylabel("Spot price")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

# a) Model hedge Δ
fig = plt.figure(figsize=(8,6))
make_3d(fig, model_surf, "Δ_model", "Model-Hedge Delta, t= 30days")

# b) Deep‐hedge Δ
fig = plt.figure(figsize=(8,6))
make_3d(fig, dh_surf,    "Δ_deep",  "Deep-Hedge Delta α=0.5, t= 30days")

# c) Difference
fig = plt.figure(figsize=(8,6))
make_3d(fig, dh_surf-model_surf, "Δ_diff", "Δ Difference Surface (Deep – Model)")

plt.tight_layout()
plt.show()
