"""Calibrate Heston parameters to an implied-volatility surface.

Objective: vega-scaled price residuals,

    r_j = (C_model(K_j, T_j) - C_mkt(K_j, T_j)) / Vega_BS(K_j, T_j; sigma_mkt),

which to first order equals sigma_model - sigma_mkt, so least squares on r is
least squares in implied-vol space without re-inverting Black every
iteration. The final fit is reported in true implied-vol terms.

Optimiser: Levenberg-Marquardt (Nielsen damping update) in an unconstrained
parametrisation -- log for v0, kappa, theta, sigma and atanh for rho -- so
every trial point is admissible. Jacobian by central differences; one
characteristic-function evaluation per maturity is shared across strikes.

Parameter standard errors come from s^2 (J^T J)^-1 with J taken w.r.t. the
natural parameters at the optimum, s^2 = RSS / (n - 5). The correlation
matrix is reported too: kappa and sigma are notoriously collinear on short
surfaces, and the condition number makes that visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .black import black_call, black_vega, implied_vols
from .pricing import HestonParams, heston_cf, lewis_call_prices


@dataclass
class Slice:
    T: float
    strikes: np.ndarray
    ivs: np.ndarray          # market implied vols


@dataclass
class Market:
    S: float
    r: float
    q: float
    slices: list

    def forward(self, T):
        return self.S * math.exp((self.r - self.q) * T)

    def df(self, T):
        return math.exp(-self.r * T)

    @property
    def n_quotes(self):
        return sum(len(s.strikes) for s in self.slices)


@dataclass
class CalibrationResult:
    params: HestonParams
    iterations: int
    rmse_iv: float                 # in vol units (0.01 = 1 vol point)
    max_abs_iv_err: float
    std_errors: HestonParams
    corr: np.ndarray
    cond: float
    history: list = field(default_factory=list)


def to_z(p: HestonParams) -> np.ndarray:
    return np.array([math.log(p.v0), math.log(p.kappa), math.log(p.theta),
                     math.log(p.sigma), math.atanh(p.rho)])


def from_z(z) -> HestonParams:
    return HestonParams(math.exp(z[0]), math.exp(z[1]), math.exp(z[2]),
                        math.exp(z[3]), math.tanh(z[4]))


def model_prices(mkt: Market, p: HestonParams) -> list:
    out = []
    for s in mkt.slices:
        cf = lambda u, T=s.T: heston_cf(u, T, p)
        out.append(lewis_call_prices(mkt.forward(s.T), s.strikes, s.T, mkt.df(s.T), cf))
    return out


def model_ivs(mkt: Market, p: HestonParams) -> list:
    return [implied_vols(c, mkt.forward(s.T), s.strikes, s.T, mkt.df(s.T))
            for c, s in zip(model_prices(mkt, p), mkt.slices)]


class _Objective:
    def __init__(self, mkt: Market):
        self.mkt = mkt
        self.mkt_px, self.vega = [], []
        for s in mkt.slices:
            F, df = mkt.forward(s.T), mkt.df(s.T)
            self.mkt_px.append(black_call(F, s.strikes, s.T, df, s.ivs))
            self.vega.append(black_vega(F, s.strikes, s.T, df, s.ivs))

    def residuals(self, p: HestonParams) -> np.ndarray:
        mp = model_prices(self.mkt, p)
        return np.concatenate([(m - c) / v for m, c, v in zip(mp, self.mkt_px, self.vega)])


def _jac(fun, x, h):
    f0 = fun(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h[i]
        J[:, i] = (fun(x + e) - fun(x - e)) / (2 * h[i])
    return f0, J


def initial_guess(mkt: Market) -> HestonParams:
    """ATM short/long implied variances seed v0/theta; the rest are generic."""
    def atm_iv(s):
        F = mkt.forward(s.T)
        return float(np.interp(0.0, np.log(s.strikes / F), s.ivs))
    short, long_ = mkt.slices[0], mkt.slices[-1]
    return HestonParams(v0=atm_iv(short) ** 2, kappa=1.0, theta=atm_iv(long_) ** 2,
                        sigma=0.5, rho=-0.5)


def calibrate(mkt: Market, x0: HestonParams | None = None, max_iter: int = 200,
              tol: float = 1e-12) -> CalibrationResult:
    obj = _Objective(mkt)
    p0 = x0 or initial_guess(mkt)
    z = to_z(p0)
    fz = lambda zz: obj.residuals(from_z(zz))
    hz = np.full(5, 1e-5)

    r, J = _jac(fz, z, hz)
    cost = 0.5 * r @ r
    A, g = J.T @ J, J.T @ r
    mu = 1e-3 * np.max(np.diag(A))
    nu = 2.0
    history = [cost]
    it = 0
    for it in range(1, max_iter + 1):
        if np.max(np.abs(g)) < tol:
            break
        try:
            step = np.linalg.solve(A + mu * np.eye(5), -g)
        except np.linalg.LinAlgError:
            mu *= nu; nu *= 2; continue
        if np.linalg.norm(step) < 1e-12 * (np.linalg.norm(z) + 1e-12):
            break
        z_new = z + step
        z_new[4] = np.clip(z_new[4], -6.0, 6.0)   # |rho| < 0.99999
        r_new = fz(z_new)
        cost_new = 0.5 * r_new @ r_new
        pred = -(step @ g) - 0.5 * step @ A @ step
        rho_gain = (cost - cost_new) / pred if pred > 0 else -1.0
        if rho_gain > 0:
            z = z_new
            r, J = _jac(fz, z, hz)
            A, g = J.T @ J, J.T @ r
            converged = (cost - cost_new) < tol * max(cost, 1e-30)
            cost = cost_new
            history.append(cost)
            mu *= max(1.0 / 3.0, 1.0 - (2 * rho_gain - 1) ** 3)
            nu = 2.0
            if converged:
                break
        else:
            mu *= nu
            nu *= 2.0

    p = from_z(z)

    # Uncertainty in natural parameters.
    x = p.as_array()
    hx = np.maximum(np.abs(x) * 1e-5, 1e-7)
    fx = lambda xx: obj.residuals(HestonParams.from_array(xx))
    rr, Jx = _jac(fx, x, hx)
    dof = max(rr.size - 5, 1)
    s2 = (rr @ rr) / dof
    JTJ = Jx.T @ Jx
    cov = s2 * np.linalg.pinv(JTJ)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    dsd = np.where(se > 0, se, 1.0)
    corr = cov / np.outer(dsd, dsd)
    cond = float(np.linalg.cond(JTJ))

    errs = np.concatenate([m - s.ivs for m, s in zip(model_ivs(mkt, p), mkt.slices)])
    return CalibrationResult(params=p, iterations=it,
                             rmse_iv=float(np.sqrt(np.mean(errs ** 2))),
                             max_abs_iv_err=float(np.max(np.abs(errs))),
                             std_errors=HestonParams.from_array(se), corr=corr,
                             cond=cond, history=history)


def synthetic_market(p: HestonParams, S=100.0, r=0.03, q=0.01,
                     maturities=(0.1, 0.25, 0.5, 1.0, 2.0),
                     moneyness=(0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30),
                     iv_noise=0.0, seed=0) -> Market:
    """Quotes generated by the Heston model itself, optionally with i.i.d.
    Gaussian noise in vol space (a stand-in for bid/ask mid noise). Wings
    are narrowed by sqrt(T) so short maturities don't quote worthless options."""
    rng = np.random.default_rng(seed)
    m = np.asarray(moneyness)
    slices = []
    for T in maturities:
        F = S * math.exp((r - q) * T)
        width = min(1.0, math.sqrt(T / 0.5))
        K = F * np.exp(np.log(m) * width)
        mk = Market(S, r, q, [])
        cf = lambda u, T=T: heston_cf(u, T, p)
        px = lewis_call_prices(F, K, T, mk.df(T), cf)
        iv = implied_vols(px, F, K, T, mk.df(T))
        iv = iv + iv_noise * rng.standard_normal(iv.size)
        slices.append(Slice(T=T, strikes=K, ivs=iv))
    return Market(S, r, q, slices)
