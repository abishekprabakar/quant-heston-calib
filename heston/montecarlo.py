"""Monte Carlo simulation of the Heston model.

Two discretisations of the variance process:

* "qe"    -- Andersen (2008) Quadratic-Exponential scheme. Moment-matches the
             non-central chi-square transition of v: a squared Gaussian when
             the variance is far from zero (psi <= 1.5), an exponential with a
             point mass at zero otherwise. Log-price uses Andersen's central
             (gamma1 = gamma2 = 1/2) integration of the variance path, with the
             Brownian increment of v eliminated through rho.
* "euler" -- full-truncation Euler (Lord, Koekkoek & van Dijk 2010).

Both simulate X_t = ln(S_t / F_t), which is driftless up to the -v/2
convexity term, so the same code serves any r, q.
"""
from __future__ import annotations

import math

import numpy as np

from .pricing import HestonParams

PSI_C = 1.5


def simulate_terminal(p: HestonParams, T: float, n_steps: int, n_paths: int,
                      scheme: str = "qe", seed: int = 0, antithetic: bool = True):
    """Return terminal X_T = ln(S_T/F) for n_paths paths."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    dt = T / n_steps
    kappa, theta, sigma, rho = p.kappa, p.theta, p.sigma, p.rho

    x = np.zeros(half * (2 if antithetic else 1))
    v = np.full_like(x, p.v0)

    def normals():
        z = rng.standard_normal(half)
        return np.concatenate([z, -z]) if antithetic else z

    def uniforms():
        u = rng.random(half)
        return np.concatenate([u, 1.0 - u]) if antithetic else u

    if scheme == "euler":
        sq = math.sqrt(dt)
        for _ in range(n_steps):
            z1, z2 = normals(), normals()
            zv = rho * z1 + math.sqrt(1 - rho * rho) * z2
            vp = np.maximum(v, 0.0)
            x += -0.5 * vp * dt + np.sqrt(vp) * sq * z1
            v += kappa * (theta - vp) * dt + sigma * np.sqrt(vp) * sq * zv
        return x

    if scheme != "qe":
        raise ValueError(f"unknown scheme {scheme!r}")

    e = math.exp(-kappa * dt)
    c1 = sigma * sigma * e * (1 - e) / kappa
    c2 = theta * sigma * sigma * (1 - e) ** 2 / (2 * kappa)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1 - rho * rho)
    K4 = g2 * dt * (1 - rho * rho)

    for _ in range(n_steps):
        m = theta + (v - theta) * e
        s2 = v * c1 + c2
        psi = s2 / (m * m)
        zv, uv, zx = normals(), uniforms(), normals()
        vn = np.empty_like(v)

        quad = psi <= PSI_C
        if quad.any():
            ip = 2.0 / psi[quad]
            b2 = ip - 1 + np.sqrt(ip) * np.sqrt(ip - 1)
            a = m[quad] / (1 + b2)
            vn[quad] = a * (np.sqrt(b2) + zv[quad]) ** 2
        ex = ~quad
        if ex.any():
            pp = (psi[ex] - 1) / (psi[ex] + 1)
            beta = (1 - pp) / m[ex]
            u = uv[ex]
            vn[ex] = np.where(u <= pp, 0.0,
                              np.log(np.maximum((1 - pp) / np.maximum(1 - u, 1e-300), 1.0)) / beta)

        x += K0 + K1 * v + K2 * vn + np.sqrt(np.maximum(K3 * v + K4 * vn, 0.0)) * zx
        v = vn
    return x


def mc_call_prices(S, strikes, T, r, q, p: HestonParams, n_steps=100, n_paths=200_000,
                   scheme="qe", seed=0):
    """Discounted MC call prices and standard errors (antithetic pairs are
    averaged before the variance is taken, so the SE is honest)."""
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    X = simulate_terminal(p, T, n_steps, n_paths, scheme, seed, antithetic=True)
    ST = F * np.exp(X)
    half = len(ST) // 2
    prices, ses = [], []
    for K in np.atleast_1d(strikes):
        pay = df * np.maximum(ST - K, 0.0)
        pair = 0.5 * (pay[:half] + pay[half:])
        prices.append(pair.mean())
        ses.append(pair.std(ddof=1) / math.sqrt(half))
    return np.array(prices), np.array(ses), F * np.exp(X).mean()
