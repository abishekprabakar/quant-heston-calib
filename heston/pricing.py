"""Heston (1993) stochastic-volatility model: characteristic function and
Fourier pricing of European options.

Dynamics under the pricing measure:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2,   d<W1, W2> = rho dt

Everything is expressed in forward terms. With F = S exp((r - q) T) and
X_T = ln(S_T / F), the characteristic function phi(u) = E[exp(i u X_T)]
satisfies phi(-i) = 1 (the forward is a martingale).

Two independent inversion formulas are implemented:

* Lewis (2001) single integral, contour shifted to Im(u) = -1/2:
      C = DF * ( F - sqrt(F K)/pi * int_0^inf Re[e^{i u k} phi(u - i/2)] / (u^2 + 1/4) du ),
  k = ln(F / K). Smooth, non-singular integrand; used for calibration.

* Gil-Pelaez / original Heston two-probability form:
      C = DF * ( F P1 - K P2 ),
      P_j = 1/2 + 1/pi int_0^inf Re[ e^{-i u ln(K/F)} psi_j(u) / (i u) ] du,
  with psi_2(u) = phi(u) and psi_1(u) = phi(u - i) (share-measure change).
  Used as a cross-check.

The characteristic function uses the Albrecher et al. (2007) "little trap"
branch, which avoids the complex-log discontinuity of Heston's original form
for long maturities.
"""
from __future__ import annotations

from dataclasses import dataclass, astuple
import math

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    sigma: float   # vol of variance
    rho: float     # spot/variance correlation

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @staticmethod
    def from_array(a) -> "HestonParams":
        return HestonParams(*map(float, a))

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2; > 1 means v_t never hits zero."""
        return 2.0 * self.kappa * self.theta / self.sigma ** 2

    def mean_integrated_variance(self, T: float) -> float:
        """E[(1/T) int_0^T v_t dt] = theta + (v0 - theta)(1 - e^{-kappa T})/(kappa T)."""
        k = self.kappa
        if k * T < 1e-10:
            return self.v0
        return self.theta + (self.v0 - self.theta) * (1.0 - math.exp(-k * T)) / (k * T)


def _log1p(w):
    """Complex log(1 + w), accurate for small |w| (numpy's complex log1p is
    not guaranteed to be)."""
    w = np.asarray(w, dtype=complex)
    shape = w.shape
    w = w.reshape(-1)
    small = np.abs(w) < 1e-3
    out = np.log(1.0 + w)
    ws = w[small]
    out[small] = ws * (1 - ws * (0.5 - ws * (1 / 3 - ws * (0.25 - ws * 0.2))))
    return out.reshape(shape)


def heston_cf(u, T: float, p: HestonParams) -> np.ndarray:
    """phi(u) = E[exp(i u ln(S_T/F))] for complex u (array-friendly)."""
    u = np.asarray(u, dtype=complex)
    kappa, theta, sigma, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    s2 = sigma * sigma
    b = kappa - 1j * rho * sigma * u
    a = 1j * u + u * u
    d = np.sqrt(b * b + s2 * a)
    # b - d evaluated as (b^2 - d^2)/(b + d): no cancellation when sigma -> 0,
    # so the Black-Scholes limit stays accurate instead of dissolving in noise.
    bpd = b + d
    bmd = -s2 * a / bpd
    g = bmd / bpd
    edt = np.exp(-d * T)
    one_minus_gedt = 1.0 - g * edt
    C = (kappa * theta / s2) * (bmd * T - 2.0 * _log1p(g * (1.0 - edt) / (1.0 - g)))
    D = -a * (1.0 - edt) / (bpd * one_minus_gedt)
    return np.exp(C + D * v0)


def black_scholes_cf(u, T: float, vol: float) -> np.ndarray:
    """Characteristic function of ln(S_T/F) under Black-Scholes."""
    u = np.asarray(u, dtype=complex)
    return np.exp(-0.5 * vol * vol * T * (u * u + 1j * u))


# ---------------------------------------------------------------------------
# Quadrature: composite Gauss-Legendre on geometrically growing panels
# [0, 0.5], [0.5, 1], [1, 2], ... [U/2, U]. Dense near the origin where the
# integrand has curvature, coarse in the exponentially decaying tail.
# ---------------------------------------------------------------------------
def _make_grid(upper: float = 2048.0, nodes_per_panel: int = 48):
    x, w = np.polynomial.legendre.leggauss(nodes_per_panel)
    edges = [0.0, 0.5]
    while edges[-1] < upper:
        edges.append(edges[-1] * 2.0)
    us, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        us.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(us), np.concatenate(ws)


_U, _W = _make_grid()


def lewis_call_prices(F: float, strikes, T: float, df: float, cf) -> np.ndarray:
    """Undiscounted-forward Lewis formula, vectorised over strikes.

    `cf` is a callable u -> phi(u) for ln(S_T/F)."""
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    k = np.log(F / K)                                     # (m,)
    phi = cf(_U - 0.5j)                                   # (n,)
    integrand = np.real(np.exp(1j * np.outer(k, _U)) * phi) / (_U * _U + 0.25)
    integral = integrand @ _W                             # (m,)
    return df * (F - np.sqrt(F * K) / math.pi * integral)


def gil_pelaez_call_prices(F: float, strikes, T: float, df: float, cf) -> np.ndarray:
    """Heston's original two-probability inversion (independent check)."""
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    lk = np.log(K / F)
    u = _U
    e = np.exp(-1j * np.outer(lk, u))
    P1 = 0.5 + (np.real(e * cf(u - 1j) / (1j * u)) @ _W) / math.pi
    P2 = 0.5 + (np.real(e * cf(u) / (1j * u)) @ _W) / math.pi
    return df * (F * P1 - K * P2)


def heston_call(S, K, T, r, q, p: HestonParams, method: str = "lewis") -> np.ndarray:
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    cf = lambda u: heston_cf(u, T, p)
    if method == "lewis":
        return lewis_call_prices(F, K, T, df, cf)
    if method == "gil-pelaez":
        return gil_pelaez_call_prices(F, K, T, df, cf)
    raise ValueError(f"unknown method {method!r}")


def heston_put(S, K, T, r, q, p: HestonParams, method: str = "lewis") -> np.ndarray:
    """Put via put-call parity: P = C - DF (F - K)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    return heston_call(S, K, T, r, q, p, method) - df * (F - K)
