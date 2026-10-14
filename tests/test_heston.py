"""Tests: every expected value is a closed form, a theorem, or an independent
numerical method -- never a number copied from this code's own output."""
import math

import numpy as np
import pytest

from heston.black import bs_call, black_call, implied_vol, implied_vols
from heston.calibrate import calibrate, synthetic_market, model_ivs, HestonParams as HP
from heston.montecarlo import mc_call_prices
from heston.pricing import (HestonParams, black_scholes_cf, gil_pelaez_call_prices,
                            heston_call, heston_cf, lewis_call_prices)

S, R, Q = 100.0, 0.03, 0.01
P = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.6, rho=-0.7)
STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])


def fwd(T):
    return S * math.exp((R - Q) * T)


# ---------------------------------------------------------------- Black ----
def test_black_scholes_textbook_value():
    # Hull, "Options, Futures & Other Derivatives": S=42, K=40, r=10%, sigma=20%,
    # T=0.5  ->  call = 4.76
    assert float(bs_call(42, 40, 0.5, 0.10, 0.0, 0.2)) == pytest.approx(4.7594, abs=1e-4)


@pytest.mark.parametrize("K", [50.0, 80.0, 100.0, 125.0, 200.0])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8])
def test_implied_vol_round_trip(K, vol):
    T, F, df = 0.75, fwd(0.75), math.exp(-R * 0.75)
    price = float(black_call(F, K, T, df, vol))
    if price - df * max(F - K, 0) < 1e-12:
        pytest.skip("time value below double precision")
    assert implied_vol(price, F, K, T, df) == pytest.approx(vol, abs=1e-8)


def test_implied_vol_rejects_arbitrage():
    with pytest.raises(ValueError):
        implied_vol(101.0, 100.0, 100.0, 1.0, 1.0)


# ------------------------------------------------- characteristic function --
def test_cf_normalisation_and_martingale():
    for T in (0.1, 1.0, 5.0):
        assert abs(heston_cf(0.0, T, P) - 1) < 1e-14          # E[1] = 1
        assert abs(heston_cf(-1j, T, P) - 1) < 1e-12          # E[S_T/F] = 1


def test_cf_first_moment_matches_closed_form():
    # E[ln(S_T/F)] = -1/2 E[int_0^T v dt] = -T/2 * mean integrated variance.
    T, h = 2.0, 1e-5
    dphi = (heston_cf(h, T, P) - heston_cf(-h, T, P)) / (2 * h)
    mean = (dphi / 1j).real
    assert mean == pytest.approx(-0.5 * T * P.mean_integrated_variance(T), rel=1e-8)


# ---------------------------------------------------------- Fourier pricing --
def test_lewis_reproduces_black_scholes_closed_form():
    T, vol = 0.5, 0.27
    F, df = fwd(T), math.exp(-R * T)
    got = lewis_call_prices(F, STRIKES, T, df, lambda u: black_scholes_cf(u, T, vol))
    np.testing.assert_allclose(got, bs_call(S, STRIKES, T, R, Q, vol), atol=1e-10)


@pytest.mark.parametrize("T", [0.05, 0.5, 2.0, 10.0])
def test_lewis_agrees_with_gil_pelaez(T):
    a = heston_call(S, STRIKES, T, R, Q, P, "lewis")
    b = heston_call(S, STRIKES, T, R, Q, P, "gil-pelaez")
    np.testing.assert_allclose(a, b, atol=1e-8)
    assert np.all(np.isfinite(a))


def test_vanishing_vol_of_vol_is_black_scholes_with_average_variance():
    # sigma -> 0: v_t is deterministic, v(t) = theta + (v0 - theta) e^{-kappa t},
    # so the price is Black-Scholes with the time-averaged variance. With
    # rho = 0 the first correction is O(sigma^2).
    T = 1.3
    p = HestonParams(v0=0.09, kappa=2.0, theta=0.03, sigma=1e-5, rho=0.0)
    vol = math.sqrt(p.mean_integrated_variance(T))
    np.testing.assert_allclose(heston_call(S, STRIKES, T, R, Q, p),
                               bs_call(S, STRIKES, T, R, Q, vol), atol=1e-8)


def test_small_vol_of_vol_correction_is_linear_in_rho_sigma():
    # With rho != 0 the leading correction to BS is O(rho * sigma): shrinking
    # sigma 100x must shrink the gap 100x. This also checks the CF stays
    # numerically stable as sigma^2 -> 1e-18 (no cancellation in b - d).
    T = 1.3
    gaps = []
    for sig in (1e-7, 1e-9):
        p = HestonParams(v0=0.09, kappa=2.0, theta=0.03, sigma=sig, rho=-0.5)
        vol = math.sqrt(p.mean_integrated_variance(T))
        gaps.append(heston_call(S, STRIKES, T, R, Q, p) - bs_call(S, STRIKES, T, R, Q, vol))
    np.testing.assert_allclose(gaps[0] / gaps[1], 100.0, rtol=1e-3)


def test_no_static_arbitrage_in_strike():
    T = 0.5
    K = np.linspace(60, 160, 41)
    c = heston_call(S, K, T, R, Q, P)
    df, F = math.exp(-R * T), fwd(T)
    assert np.all(c >= df * np.maximum(F - K, 0) - 1e-10)
    assert np.all(c <= df * F)
    assert np.all(np.diff(c) < 0)                    # decreasing in K
    assert np.all(np.diff(c, 2) > 0)                 # convex in K (density > 0)


def test_zero_correlation_smile_is_symmetric_in_log_moneyness():
    # Renault & Touzi (1996): with rho = 0 the implied vol is a symmetric
    # function of ln(K/F).
    p = HestonParams(v0=0.04, kappa=1.0, theta=0.05, sigma=0.8, rho=0.0)
    T = 0.75
    F, df = fwd(T), math.exp(-R * T)
    k = np.array([0.05, 0.15, 0.3])
    K = np.concatenate([F * np.exp(-k), F * np.exp(k)])
    iv = implied_vols(heston_call(S, K, T, R, Q, p), F, K, T, df)
    np.testing.assert_allclose(iv[:3], iv[3:], atol=1e-8)
    assert iv[2] > iv[0]                             # but it is a smile, not flat


def test_negative_rho_gives_downward_skew():
    T = 0.5
    F, df = fwd(T), math.exp(-R * T)
    K = np.array([0.9 * F, F, 1.1 * F])
    iv = implied_vols(heston_call(S, K, T, R, Q, P), F, K, T, df)
    assert iv[0] > iv[1] > iv[2]


# -------------------------------------------------------------- Monte Carlo --
@pytest.mark.parametrize("scheme,steps", [("qe", 32), ("euler", 400)])
def test_monte_carlo_converges_to_fourier(scheme, steps):
    T = 1.0
    ref = heston_call(S, STRIKES, T, R, Q, P)
    mc, se, mean_S = mc_call_prices(S, STRIKES, T, R, Q, P, n_steps=steps,
                                    n_paths=200_000, scheme=scheme, seed=11)
    # 4 standard errors plus a small allowance for time-discretisation bias.
    assert np.all(np.abs(mc - ref) < 4 * se + 0.02), (mc - ref, se)


def test_qe_preserves_forward():
    T = 2.0
    _, _, mean_S = mc_call_prices(S, [100.0], T, R, Q, P, n_steps=20,
                                  n_paths=400_000, scheme="qe", seed=3)
    assert mean_S == pytest.approx(fwd(T), rel=2e-3)


# -------------------------------------------------------------- Calibration --
def test_calibration_recovers_ground_truth_from_clean_surface():
    truth = HestonParams(v0=0.03, kappa=2.5, theta=0.05, sigma=0.7, rho=-0.6)
    mkt = synthetic_market(truth)
    res = calibrate(mkt)                            # starts from a generic guess
    np.testing.assert_allclose(res.params.as_array(), truth.as_array(), rtol=1e-5)
    assert res.rmse_iv < 1e-7


def test_calibration_with_noise_is_statistically_consistent():
    noise = 0.003                                    # 30 bp of vol
    mkt = synthetic_market(P, iv_noise=noise, seed=7)
    res = calibrate(mkt)
    # Residual RMSE should be the noise level (5 params soak up a few dof).
    assert 0.6 * noise < res.rmse_iv < 1.3 * noise
    # Ground truth within 4 reported standard errors for every parameter.
    z = np.abs(res.params.as_array() - P.as_array()) / res.std_errors.as_array()
    assert np.all(z < 4), z


def test_fitted_ivs_match_market_on_clean_surface():
    mkt = synthetic_market(P)
    res = calibrate(mkt)
    for fit, s in zip(model_ivs(mkt, res.params), mkt.slices):
        np.testing.assert_allclose(fit, s.ivs, atol=1e-7)


def test_feller_and_average_variance_limits():
    p = HestonParams(v0=0.09, kappa=2.0, theta=0.04, sigma=0.3, rho=0.0)
    assert p.feller_ratio == pytest.approx(2 * 2.0 * 0.04 / 0.09)
    assert p.mean_integrated_variance(1e-12) == pytest.approx(0.09)
    assert p.mean_integrated_variance(1e6) == pytest.approx(0.04, rel=1e-5)
