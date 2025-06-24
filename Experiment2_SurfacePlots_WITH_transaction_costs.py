#!/usr/bin/env python3
"""
plot_delta_surfaces_tc.py  ░  Deep‑Hedging with proportional transaction costs
--------------------------------------------------------------------------
This is a drop‑in variant of *plot_delta_surfaces.py* that replicates
**Experiment 2** (Section 5.3) of the Deep‑Hedging paper by adding a
non‑zero proportional cost ε on the stock leg.

It produces the same three 3‑D surface plots but after re‑training the
policy with the new market frictions:
  1. Analytic Black‑Scholes Δ(t,s)
  2. Deep‑hedge Δ(t,s) learned under CVaR@50 % **with costs**
  3. Difference Δ_Deep − Δ_Model

Usage
──────
    python -m deephedging.plot_delta_surfaces_tc

--------------------------------------------------------------------------
If a cached checkpoint for the exact configuration (same ε, seeds, network
layout, etc.) is found, the training step is skipped and the policy is
loaded automatically.
"""

import os
# ── 0)  Force CPU‑only, disable XLA/JIT  ─────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"]        = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"]           = "--xla_force_host_platform_device_count=1"

import tensorflow as tf
# fully disable GPUs (safety for multi‑user servers)
tf.config.set_visible_devices([], "GPU")
tf.config.optimizer.set_jit(False)

# ── 1)  DYNAPLOT → CANVAS MODE (nice inline 3‑D)  ───────────────────────
from cdxbasics.dynaplot import DynamicFig
DynamicFig.MODE = "canvas"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – keeps 3‑D backend active
from scipy.stats import norm

from cdxbasics.config   import Config
from deephedging.world  import SimpleWorld_Spot_ATM
from deephedging.gym    import VanillaDeepHedgingGym
from deephedging.trainer import train

# ════════════════════════════════════════════════════════════════════════
# 2)  CONFIGURATION –  ε > 0   (transaction cost experiment)             ║
# ════════════════════════════════════════════════════════════════════════

cfg               = Config()

#  ----  Market / model parameters  ----
cfg.update({
    "r"      : 0.00,   # risk‑free rate
    "rvol"   : 0.20,   # spot vol (Black‑Scholes reference sigma)
    "volvol_rvol": 0.30,   # vol‑of‑vol for Heston‑like dyn.
    "meanrev_rvol": 2.0,
    "corr_vs": -0.7,   # corr(S, v)

    # ----  ⧉  **PROPORTIONAL TRANSACTION COSTS**  ----
    # ε is the proportional cost coefficient  (e.g. 50 bp per round‑turn)
    #   cost = ε · |ΔN| · S_t
    # Set ε anywhere in [0.0, 0.01].  Paper uses a variety.
    "cost_s": 0.005,   # ← 0.5 % proportional cost on the stock
    "cost_v": 0.0,     # no cost on variance swap leg
    "cost_p": 0.0,     # no cost on cash (redundant)

    # ----  Simulation grid  ----
    "dt"     : 1.0/250,  # 1 trading day ≈ 1/250 yr
    "steps"  : 60,       # 60 trading days ≈ 3 months
    "samples": 10000,    # should train fine on a laptop

    "no_stoch_drift": True  # keep μ = r for comparability
})

#  ----  Training hyper‑parameters  ----
cfg.train.epochs        = 200        # a bit more since costs hamper learning
cfg.train.batch_size    = 2048
cfg.train.caching_mode  = "off"      # change to "full" to store datasets on disk

#  ----  Objective  ----
cfg.update({"monetary_utility": "cvar@0.5"})

# ════════════════════════════════════════════════════════════════════════
# 3)  Train / load the deep‑hedge network                                  ║
# ════════════════════════════════════════════════════════════════════════

world  = SimpleWorld_Spot_ATM(cfg)
val_w  = world.clone()
gym    = VanillaDeepHedgingGym(cfg)

#  •  If an identical checkpoint exists -> will be reused automatically
train(gym, world, val_w, config=cfg)

# ════════════════════════════════════════════════════════════════════════
# 4)    Simulation → grab spot paths & delta component                     ║
# ════════════════════════════════════════════════════════════════════════

out       = gym.predict(world.tf_data)  # one forward pass of N paths
spots     = world.details.spot_all      # shape: [N, steps+1]
_actions  = out["actions"]
if isinstance(_actions, tf.Tensor):
    actions = _actions.numpy()[:, :, 0]   # delta component only
else:
    actions = _actions[:, :, 0]

# ════════════════════════════════════════════════════════════════════════
# 5)     Build (t,s) grid + surfaces                                      ║
# ════════════════════════════════════════════════════════════════════════

steps      = cfg["steps"]
times      = np.arange(steps)                      # 0 … 59
total_T    = steps * cfg["dt"]                    # full maturity in years
T_grid     = total_T - times * cfg["dt"]          # time‑to‑maturity per hedge step

# choose 50 equi‑distant spot bins over realised range
s_min, s_max = spots.min(), spots.max()
spot_bins    = np.linspace(s_min, s_max, 50)
centers      = 0.5 * (spot_bins[:-1] + spot_bins[1:])

# ----  Analytic Black‑Scholes delta  ----

def bs_delta(S, K, T, r, sigma):
    """European call delta under Black‑Scholes."""
    T = np.clip(T, 1e-6, None)  # avoid division by zero at expiry
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

model_surf = np.zeros((steps, len(centers)))
dh_surf    = np.zeros_like(model_surf)

for t in range(steps):
    model_surf[t] = bs_delta(centers, 1.0, T_grid[t], cfg["r"], cfg["rvol"])

    # bin the simulated spot paths at hedge step t
    idx = np.digitize(spots[:, t], spot_bins) - 1
    for i in range(len(centers)):
        mask = idx == i
        dh_surf[t, i] = actions[mask, t].mean() if mask.any() else np.nan

# Meshgrid for 3‑D plotting
T_mesh, S_mesh = np.meshgrid(times, centers, indexing="ij")

# ════════════════════════════════════════════════════════════════════════
# 6)     Plotting helper                                                  ║
# ════════════════════════════════════════════════════════════════════════

def add_surface(fig, zdata, zlabel, title):
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T_mesh, S_mesh, zdata, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Spot price")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

# a) Model hedge Δ
fig = plt.figure(figsize=(8, 6))
add_surface(fig, model_surf, "Δ_model", "Model‑Hedge Δ – analytic, ε=0")

# b) Deep‑hedge Δ with costs
fig = plt.figure(figsize=(8, 6))
add_surface(fig, dh_surf, "Δ_deep", "Deep‑Hedge Δ (ε=0.5 %, CVaR@50 %)")

# c) Difference (Deep − Model)
fig = plt.figure(figsize=(8, 6))
add_surface(fig, dh_surf - model_surf, "Δ_diff", "Δ Difference Surface – with costs")

plt.tight_layout()
plt.show()
