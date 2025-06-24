import os
# (your CPU-only flags; you can remove these if you actually want to use GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from cdxbasics.dynaplot import DynamicFig
DynamicFig.MODE = "canvas"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.optimizer.set_jit(False)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from cdxbasics.config import Config
from deephedging.world import SimpleWorld_Spot_ATM
from deephedging.gym   import VanillaDeepHedgingGym
from deephedging.trainer import train

def bs_delta(S, K, T, r, sigma):
    T = np.clip(T, 1e-6, None)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def train_deephedge(cfg, alpha):
    cfg.update({"monetary_utility": f"cvar@{alpha}"})
    gym   = VanillaDeepHedgingGym(cfg)
    world = SimpleWorld_Spot_ATM(cfg)
    train(gym, world, world.clone(), config=cfg)
    out = gym.predict(world.tf_data)
    pnl     = out["pnl"].numpy().flatten() if isinstance(out["pnl"], tf.Tensor) else out["pnl"].flatten()
    actions = out["actions"].numpy() if isinstance(out["actions"], tf.Tensor) else out["actions"]
    return pnl, actions[:,:,0], world

def main():
    cfg = Config()
    cfg.update({
        "r":0.0, "rvol":0.2, "volvol_rvol":0.3, "meanrev_rvol":2.0,
        "corr_vs":-0.7, "cost_s":0.0, "cost_v":0.0, "cost_p":0.0,
        "dt":1/250, "steps":60, "samples":10000, "no_stoch_drift":True
    })
    cfg.train.epochs       = 100
    cfg.train.batch_size   = 2048
    cfg.train.caching_mode = "off"

    # train CVaR@50% and @99% policies
    pnl50, delta50, world = train_deephedge(cfg.copy(), 0.5)
    pnl99, delta99, _     = train_deephedge(cfg.copy(), 0.99)

    # simulate payoffs & compute hedging errors
    spots  = world.details.spot_all
    payoff = np.maximum(spots[:,-1] - 1.0, 0.0)
    dS     = spots[:,1:] - spots[:,:-1]

    # model‐hedge (BS) errors
    time_grid = np.linspace(cfg["steps"]*cfg["dt"], 0, cfg["steps"])[None,:]
    delta_bs  = bs_delta(spots[:,:-1], 1.0, time_grid, cfg["r"], cfg["rvol"])
    e_bs      = -payoff + (delta_bs * dS).sum(axis=1) + 0.038939

    # deep‐hedge errors
    e_dh50 = -payoff + (delta50 * dS).sum(axis=1) + np.mean(pnl50)
    e_dh99 = -payoff + (delta99 * dS).sum(axis=1) + np.mean(pnl99)

    # --- SIDE-BY-SIDE HISTOGRAMS ---
    bins = 25
    counts_bs, edges = np.histogram(e_bs,   bins=bins, density=True)
    counts_50, _     = np.histogram(e_dh50, bins=edges, density=True)
    counts_99, _     = np.histogram(e_dh99, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    width   = (edges[1] - edges[0]) / 3  # bar width

    plt.figure(figsize=(10,6))
    plt.bar(centers - width, counts_bs, width=width, alpha=0.7,
            label="Model Hedge (BS)", color="grey")
    plt.bar(centers,          counts_50, width=width, alpha=0.7,
            label="Deep Hedge CVaR@99%", color="C0")
    plt.bar(centers + width,  counts_99, width=width, alpha=0.7,
            label="Deep Hedge CVaR@50%", color="C1")

    plt.title("Hedging Error Distributions")
    plt.xlabel("Hedging error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("side_by_side_hedge_errors.png", dpi=300)
    print("✅ Saved as side_by_side_hedge_errors.png")
    plt.show()

if __name__ == "__main__":
    main()