"""End-to-end demo: noisy synthetic equity-index surface -> Heston calibration
-> fit diagnostics -> independent Monte Carlo check -> plots in output/.

    python demo.py            # writes output/smile_fit.png, output/mc_check.png
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from heston.calibrate import calibrate, model_ivs, synthetic_market
from heston.montecarlo import mc_call_prices
from heston.pricing import HestonParams, heston_call

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

TRUTH = HestonParams(v0=0.035, kappa=1.8, theta=0.055, sigma=0.65, rho=-0.72)
NOISE = 0.0025          # 25 bp of implied vol, i.i.d. per quote
S, R, Q = 100.0, 0.035, 0.013


def main():
    os.makedirs(OUT, exist_ok=True)
    mkt = synthetic_market(TRUTH, S=S, r=R, q=Q,
                           maturities=(1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0),
                           iv_noise=NOISE, seed=2026)
    print(f"Synthetic surface: {mkt.n_quotes} quotes over {len(mkt.slices)} maturities, "
          f"{NOISE * 1e4:.0f} bp vol noise, S={S}, r={R}, q={Q}")

    t0 = time.perf_counter()
    res = calibrate(mkt)
    dt = time.perf_counter() - t0
    print(f"Levenberg-Marquardt converged in {res.iterations} iterations ({dt:.2f} s)\n")

    names = ["v0", "kappa", "theta", "sigma", "rho"]
    print(f"{'param':>6} {'truth':>9} {'fitted':>9} {'std err':>9} {'z':>6}")
    for n, t, f, s in zip(names, TRUTH.as_array(), res.params.as_array(), res.std_errors.as_array()):
        print(f"{n:>6} {t:9.4f} {f:9.4f} {s:9.4f} {(f - t) / s:6.2f}")
    print(f"\nFeller ratio 2*kappa*theta/sigma^2: truth {TRUTH.feller_ratio:.3f}, "
          f"fitted {res.params.feller_ratio:.3f}")
    print(f"IV fit: RMSE {res.rmse_iv * 1e4:.1f} bp, max |err| {res.max_abs_iv_err * 1e4:.1f} bp "
          f"(noise injected: {NOISE * 1e4:.0f} bp)")
    print(f"cond(J^T J) = {res.cond:.2e}")
    print("parameter correlation matrix:")
    print("        " + " ".join(f"{n:>7}" for n in names))
    for n, row in zip(names, res.corr):
        print(f"{n:>7} " + " ".join(f"{x:7.2f}" for x in row))

    fits = model_ivs(mkt, res.params)
    print(f"\n{'T':>6} {'ATM mkt':>8} {'ATM fit':>8} {'skew(±0.1√T) mkt':>17} {'fit':>7} {'RMSE bp':>8}")
    for s, f in zip(mkt.slices, fits):
        F = mkt.forward(s.T)
        k = np.log(s.strikes / F)
        atm_m, atm_f = np.interp(0, k, s.ivs), np.interp(0, k, f)
        w = 0.1 * math.sqrt(s.T)
        sk_m = np.interp(-w, k, s.ivs) - np.interp(w, k, s.ivs)
        sk_f = np.interp(-w, k, f) - np.interp(w, k, f)
        print(f"{s.T:6.3f} {atm_m:8.4f} {atm_f:8.4f} {sk_m:17.4f} {sk_f:7.4f} "
              f"{np.sqrt(np.mean((f - s.ivs) ** 2)) * 1e4:8.1f}")

    # Independent check: price the 1y slice by Andersen QE Monte Carlo under
    # the fitted parameters and compare to the Fourier price.
    T = 1.0
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    four = heston_call(S, K, T, R, Q, res.params)
    mc, se, _ = mc_call_prices(S, K, T, R, Q, res.params, n_steps=50,
                               n_paths=400_000, scheme="qe", seed=1)
    print(f"\nMonte Carlo (Andersen QE, 50 steps, 400k antithetic paths) vs Fourier, T={T}:")
    print(f"{'K':>6} {'Fourier':>10} {'MC':>10} {'MC s.e.':>9} {'diff/se':>8}")
    for k_, a, b, e in zip(K, four, mc, se):
        print(f"{k_:6.0f} {a:10.4f} {b:10.4f} {e:9.4f} {(b - a) / e:8.2f}")

    _plot(mkt, res, fits)
    _plot_mc(res.params)
    print(f"\nwrote {os.path.relpath(OUT)}/smile_fit.png and {os.path.relpath(OUT)}/mc_check.png")


def _plot(mkt, res, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(mkt.slices)
    cols = 3
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(12, 6.5), sharey=True)
    for ax, s, f in zip(axes.flat, mkt.slices, fits):
        F = mkt.forward(s.T)
        k = np.log(s.strikes / F)
        kk = np.linspace(k.min(), k.max(), 80)
        from heston.black import implied_vols
        dense = implied_vols(heston_call(mkt.S, F * np.exp(kk), s.T, mkt.r, mkt.q, res.params),
                             F, F * np.exp(kk), s.T, mkt.df(s.T))
        ax.plot(kk, dense * 100, color="#2f6fdf", lw=1.8, label="Heston fit")
        ax.scatter(k, s.ivs * 100, color="#d94a3a", s=18, zorder=3, label="market (noisy)")
        ax.set_title(f"T = {s.T:.3g}y")
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("implied vol (%)")
    axes.flat[0].legend(fontsize=8)
    p = res.params
    fig.suptitle(f"Heston calibration: v0={p.v0:.4f} kappa={p.kappa:.2f} theta={p.theta:.4f} "
                 f"sigma={p.sigma:.2f} rho={p.rho:.2f}   RMSE {res.rmse_iv * 1e4:.1f} bp")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "smile_fit.png"), dpi=110)
    plt.close(fig)


def _plot_mc(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, K = 1.0, 100.0
    ref = float(heston_call(S, K, T, R, Q, p)[0])
    steps = [2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for scheme, c in (("euler", "#d94a3a"), ("qe", "#2f6fdf")):
        errs, ses = [], []
        for n in steps:
            mc, se, _ = mc_call_prices(S, [K], T, R, Q, p, n_steps=n, n_paths=400_000,
                                       scheme=scheme, seed=5)
            errs.append(mc[0] - ref)
            ses.append(se[0])
        ax.errorbar(steps, errs, yerr=2 * np.array(ses), marker="o", color=c, capsize=3,
                    label=f"{scheme.upper()}  (bias ±2 s.e.)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("time steps per year")
    ax.set_ylabel("MC price − Fourier price")
    ax.set_title(f"ATM 1y call, Fourier = {ref:.4f}: discretisation bias by scheme")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mc_check.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
