"""Black (forward) pricing, vega and a robust implied-volatility solver.

Calibration fits in implied-vol space, so the inversion has to be reliable
across deep ITM/OTM strikes: Newton steps are safeguarded by a bracket that
shrinks on every iteration, falling back to bisection when Newton leaves it.
"""
from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _ncdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT2))


def _npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT2PI


def black_call(F, K, T, df, vol):
    F, K, vol = (np.asarray(a, dtype=float) for a in (F, K, vol))
    sd = vol * math.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    price = df * (F * _ncdf(d1) - K * _ncdf(d2))
    intrinsic = df * np.maximum(F - K, 0.0)
    return np.where(sd > 0, price, intrinsic)


def black_vega(F, K, T, df, vol):
    F, K, vol = (np.asarray(a, dtype=float) for a in (F, K, vol))
    sd = vol * math.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    return df * F * _npdf(d1) * math.sqrt(T)


def bs_call(S, K, T, r, q, vol):
    F = S * math.exp((r - q) * T)
    return black_call(F, K, T, math.exp(-r * T), vol)


def implied_vol(price, F, K, T, df, lo=1e-4, hi=5.0, tol=1e-12, max_iter=100) -> float:
    """Black implied vol of a call price. Raises if the price is outside the
    no-arbitrage band (intrinsic, DF*F)."""
    intrinsic = df * max(F - K, 0.0)
    upper = df * F
    if not (intrinsic - 1e-14 <= price < upper):
        raise ValueError(f"call price {price} outside arbitrage bounds [{intrinsic}, {upper})")
    if price - intrinsic < 1e-14 * max(1.0, F):
        return 0.0
    a, b = lo, hi
    while float(black_call(F, K, T, df, b)) < price and b < 50:
        b *= 2.0
    x = 0.5 * (a + b)
    # Start Newton at the inflection-point vol, where it converges monotonically.
    x0 = math.sqrt(2.0 * abs(math.log(F / K)) / T) if K != F else 0.2
    if a < x0 < b:
        x = x0
    for _ in range(max_iter):
        f = float(black_call(F, K, T, df, x)) - price
        if abs(f) < tol:
            return x
        if f > 0:
            b = x
        else:
            a = x
        v = float(black_vega(F, K, T, df, x))
        step_ok = False
        if v > 1e-14:
            xn = x - f / v
            if a < xn < b:
                x, step_ok = xn, True
        if not step_ok:
            x = 0.5 * (a + b)
        if b - a < 1e-15:
            return x
    return x


def implied_vols(prices, F, strikes, T, df) -> np.ndarray:
    return np.array([implied_vol(float(p), F, float(k), T, df) for p, k in zip(prices, strikes)])
