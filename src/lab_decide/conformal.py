"""Finite-sample Mondrian & adaptive conformal calibration, stacked surrogates, and kNN/Tanimoto gating."""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_ZERO_VAR_TOL = 1e-12
_MIN_NONZERO_STD_FLOOR = 1e-3
_CONST_INVARIANT_TOL = 1e-6
_CONST_OOD_SCALE = 0.10
_MIN_ACTIVE_FEATURE_WEIGHT = 0.05
_MAHALANOBIS_RIDGE = 0.10
_DEFAULT_MORGAN_TAIL_BITS = 1024
_DEFAULT_CORE_PHYS_COLS = 30

VALID_SURROGATES: tuple[str, ...] = (
    "gbdt",
    "hist_gbdt",
    "lightgbm",
    "frontier_ensemble",
)

VALID_DISTANCE_METRICS: tuple[str, ...] = (
    "euclidean",
    "mahalanobis",
    "ensemble_mahalanobis",
    "tanimoto_mahalanobis",
)


@lru_cache(maxsize=4096)
def _float_to_fraction(fval: float) -> Fraction:
    return Fraction(str(fval))


def _to_exact_fraction(value: float | int | Fraction | Decimal) -> Fraction:
    """Convert numeric level/weight to exact Fraction via shortest decimal representation."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Expected finite numeric value, got {value}")
        return Fraction(value)
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Expected finite numeric value, got {value}")
    return _float_to_fraction(fval)


def spearman_rank_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """Compute Spearman rank correlation between `x` and `y` (`nan` if degenerate or `< 3` finite rows)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and y must be 1D arrays of matching shape")
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xm = xa[mask]
    ym = ya[mask]
    if np.ptp(xm) <= _ZERO_VAR_TOL or np.ptp(ym) <= _ZERO_VAR_TOL:
        return float("nan")
    res = spearmanr(xm, ym)
    stat = float(res.statistic)
    return stat if math.isfinite(stat) else float("nan")


def conformal_rank(
    n_calib: int,
    alpha: float | Fraction | Decimal = 0.10,
    *,
    alpha_permille: int | None = None,
) -> int:
    """Exact finite-sample order-statistic rank `ceil((n + 1) * (1 - alpha))` matching `LabDecideLean.conformalRank`.

    Uses exact rational/integer arithmetic so binary64 representation drift (e.g. `1 - 0.18`
    at `n = 149`) never shifts order statistics away from `((n + 1) * (1000 - alphaPermille) + 999) // 1000`.
    """
    n_int = int(n_calib)
    if n_int < 0:
        raise ValueError(f"n_calib must be non-negative, got {n_calib}")
    if alpha_permille is not None:
        ap = int(alpha_permille)
        if not (0 < ap < 1000):
            raise ValueError(f"alpha_permille must lie strictly in (0, 1000), got {alpha_permille}")
        return ((n_int + 1) * (1000 - ap) + 999) // 1000

    alpha_frac = _to_exact_fraction(alpha)
    if not (Fraction(0, 1) < alpha_frac < Fraction(1, 1)):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    return int(math.ceil((n_int + 1) * (Fraction(1, 1) - alpha_frac)))


def conformal_quantile(
    scores: Sequence[float] | np.ndarray,
    alpha: float | Fraction | Decimal = 0.10,
) -> float:
    """Exact finite-sample split conformal quantile `ceil((n + 1) * (1 - alpha)) / n`.

    Never clamps `level` to `1.0`: when `n < (1 - alpha) / alpha`, achieving `1 - alpha`
    distribution-free coverage from `n` calibration samples is mathematically impossible,
    so a `UserWarning` is emitted and `math.inf` is returned.
    """
    alpha_frac = _to_exact_fraction(alpha)
    if not (Fraction(0, 1) < alpha_frac < Fraction(1, 1)):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Calibration scores must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Calibration scores contain NaN or Inf")

    n = int(arr.size)
    k_ord = conformal_rank(n, alpha_frac)
    raw_level = k_ord / float(n)
    if k_ord > n:
        warnings.warn(
            f"Calibration size n={n} is too small for finite (1 - alpha)={float(1 - alpha_frac):.3f} "
            f"conformal coverage (ceil((n+1)*(1-alpha))/n = {raw_level:.3f} > 1.0); "
            "returning q = inf (abstain).",
            UserWarning,
            stacklevel=2,
        )
        return float("inf")
    return float(np.partition(arr, k_ord - 1)[k_ord - 1])


def weighted_conformal_quantile(
    scores: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    alpha: float | Fraction | Decimal = 0.10,
    *,
    test_weight: float = 1.0,
) -> float:
    """Importance-weighted conformal quantile under covariate shift (Tibshirani et al., 2019).

    Aggregates weights over tied nonconformity scores and accumulates exact rational
    unnormalized weights against `(1 - alpha) * (sum(weights) + test_weight)` so unit
    weights match `conformal_quantile` identically without IEEE-754 ULP drift.
    """
    alpha_frac = _to_exact_fraction(alpha)
    if not (Fraction(0, 1) < alpha_frac < Fraction(1, 1)):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or s.shape != w.shape:
        raise ValueError("scores and weights must be non-empty 1D arrays of equal length")
    if not np.all(np.isfinite(s)):
        raise ValueError("Calibration scores contain NaN or Inf")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)) or not math.isfinite(float(test_weight)) or float(test_weight) <= 0.0:
        raise ValueError("weights must be finite non-negative and test_weight > 0")

    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    w_sorted = w[order]

    one_minus_alpha = Fraction(1, 1) - alpha_frac
    test_w_frac = _to_exact_fraction(float(test_weight))

    unique_scores: list[float] = []
    grouped_weights: list[Fraction] = []
    total_cal_w = Fraction(0, 1)
    for sc_val, wt_val in zip(s_sorted.tolist(), w_sorted.tolist()):
        if wt_val <= 0.0:
            continue
        wt_frac = _to_exact_fraction(wt_val)
        total_cal_w += wt_frac
        if unique_scores and sc_val == unique_scores[-1]:
            grouped_weights[-1] += wt_frac
        else:
            unique_scores.append(sc_val)
            grouped_weights.append(wt_frac)

    total_w = total_cal_w + test_w_frac
    if total_w <= 0:
        return float("inf")
    target_w = one_minus_alpha * total_w
    if total_cal_w < target_w:
        warnings.warn(
            "Total calibration weight insufficient for weighted conformal level; returning inf.",
            UserWarning,
            stacklevel=2,
        )
        return float("inf")

    cum_w = Fraction(0, 1)
    for sc_val, grp_w in zip(unique_scores, grouped_weights):
        cum_w += grp_w
        if cum_w >= target_w:
            return float(sc_val)

    warnings.warn(
        "Total calibration weight insufficient for weighted conformal level; returning inf.",
        UserWarning,
        stacklevel=2,
    )
    return float("inf")


def conformal_threshold(
    cal_distances: Sequence[float] | np.ndarray,
    trust_pct: float = 80.0,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    test_weight: float = 1.0,
) -> float:
    """Finite-sample conformal outlier cutoff `ceil((n + 1) * (trust_pct / 100)) / n`."""
    if not (0.0 < float(trust_pct) <= 100.0):
        raise ValueError(f"trust_pct must lie in (0, 100], got {trust_pct}")
    d = np.asarray(cal_distances, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("Calibration distances must be a non-empty 1D sequence")
    if not np.all(np.isfinite(d)):
        raise ValueError("Calibration distances contain NaN or Inf")

    pct_frac = _to_exact_fraction(float(trust_pct)) / Fraction(100, 1)
    if weights is not None:
        if pct_frac >= Fraction(1, 1):
            return float("inf")
        alpha_thr = Fraction(1, 1) - pct_frac
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return weighted_conformal_quantile(d, weights, alpha=alpha_thr, test_weight=test_weight)

    n = int(d.size)
    k_ord = int(math.ceil((n + 1) * pct_frac))
    if k_ord > n:
        return float("inf")
    return float(np.partition(d, k_ord - 1)[k_ord - 1])


def compute_covariate_shift_weights(
    X_ref: np.ndarray | Sequence[Sequence[float]],
    X_active: np.ndarray | Sequence[Sequence[float]],
    *,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute defensive-mixture Radon-Nikodym weights `w(x) = dP_0 / dP_cal` in `(0, 1]`."""
    X_ref_2d, _ = _ensure_2d(X_ref)
    n_ref = int(X_ref_2d.shape[0])
    if n_ref == 0:
        raise ValueError("X_ref must contain at least 1 reference sample")

    X_act_arr = np.asarray(X_active, dtype=np.float64)
    if X_act_arr.size == 0:
        return np.ones(n_ref, dtype=np.float64)
    X_act_2d, _ = _ensure_2d(X_act_arr, expected_dim=X_ref_2d.shape[1])
    n_act = int(X_act_2d.shape[0])

    if scaler_mean is not None and scaler_scale is not None:
        g_dim = int(np.asarray(scaler_mean).size)
        X_ref_2d = X_ref_2d[:, :g_dim]
        X_act_2d = X_act_2d[:, :g_dim]
        mean_v = np.asarray(scaler_mean, dtype=np.float64)
        scale_v = np.where(np.asarray(scaler_scale, dtype=np.float64) > _ZERO_VAR_TOL, scaler_scale, 1.0)
    else:
        mean_v = np.mean(X_ref_2d, axis=0)
        raw_s = np.std(X_ref_2d, axis=0)
        scale_v = np.where(raw_s > _ZERO_VAR_TOL, np.maximum(raw_s, _MIN_NONZERO_STD_FLOOR), 1.0)

    fw = np.ones(X_ref_2d.shape[1], dtype=np.float64) if feature_weights is None else np.asarray(feature_weights, dtype=np.float64)[: X_ref_2d.shape[1]]
    Z_ref = ((X_ref_2d - mean_v) / scale_v) * fw
    Z_act = ((X_act_2d - mean_v) / scale_v) * fw
    Z_all = np.vstack([Z_ref, Z_act])

    if n_ref >= 2:
        diff_rr = Z_ref[:, None, :] - Z_ref[None, :, :]
        d2_rr = np.sum(diff_rr ** 2, axis=2)
        tri = d2_rr[np.triu_indices(n_ref, k=1)]
        pos_tri = tri[tri > _ZERO_VAR_TOL]
        med_d2 = float(np.median(pos_tri)) if pos_tri.size > 0 else 1.0
    else:
        med_d2 = 1.0
    sigma2 = max(0.5 * med_d2, 0.25)

    d2_to_ref = np.sum((Z_all[:, None, :] - Z_ref[None, :, :]) ** 2, axis=2)
    d2_to_act = np.sum((Z_all[:, None, :] - Z_act[None, :, :]) ** 2, axis=2)

    p_ref = np.mean(np.exp(-0.5 * d2_to_ref / sigma2), axis=1)
    q_act = np.mean(np.exp(-0.5 * d2_to_act / sigma2), axis=1)

    p_floor = max(float(np.percentile(p_ref[:n_ref], 5)) * 0.10, 1e-6)
    r_hat = q_act / np.maximum(p_ref, p_floor)

    n_total = float(n_ref + n_act)
    pi_act = float(n_act) / n_total
    u = 1.0 / ((1.0 - pi_act) + pi_act * r_hat)
    u_max = float(np.max(u))
    return np.clip(u / u_max if u_max > 0.0 else np.ones_like(u), 1e-6, 1.0)


def winkler_interval_score(
    y: Sequence[float] | np.ndarray,
    lo: Sequence[float] | np.ndarray,
    hi: Sequence[float] | np.ndarray,
    alpha: float = 0.10,
    *,
    mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Mean Winkler interval efficiency score penalizing both width and miscoverage."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    y_arr = np.asarray(y, dtype=np.float64)
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if int(np.sum(m)) == 0:
            return float("nan")
        y_arr = y_arr[m]
        lo_arr = lo_arr[m]
        hi_arr = hi_arr[m]
    elif y_arr.size == 0:
        return float("nan")

    width = hi_arr - lo_arr
    below = np.maximum(lo_arr - y_arr, 0.0)
    above = np.maximum(y_arr - hi_arr, 0.0)
    scores = width + (2.0 / float(alpha)) * (below + above)
    return float(np.mean(scores))


def _ensure_2d(X: np.ndarray | Sequence[float], expected_dim: int | None = None) -> tuple[np.ndarray, bool]:
    arr = np.asarray(X, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D feature array, got {arr.ndim}D array")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} features, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input features contain NaN or Inf")
    return arr, is_1d


# ---------------------------------------------------------------------------
# Frontier Semiparametric Stacked Molecular Surrogate & Centered Residual CQR
# ---------------------------------------------------------------------------


class _AnalyticSVDRidge:
    """Closed-form SVD Ridge selecting optimal L2 regularization via exact Leave-One-Out CV."""

    def __init__(self, n_cols: int, min_binary_scale: float = 0.35) -> None:
        self.n_cols = int(n_cols)
        self.min_binary_scale = float(min_binary_scale)
        self.mean_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.scale_: np.ndarray = np.ones(0, dtype=np.float64)
        self.active_: np.ndarray = np.zeros(0, dtype=bool)
        self.coef_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.y_mean_: float = 0.0
        self.alpha_: float = 100.0

    def fit_with_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = min(self.n_cols, X.shape[1])
        Xl = X[:, :c]
        self.mean_ = np.mean(Xl, axis=0)
        raw_std = np.std(Xl, axis=0)
        ptp = np.ptp(Xl, axis=0)
        self.active_ = raw_std > 1e-8
        self.y_mean_ = float(np.mean(y))

        if len(y) < 3 or not np.any(self.active_) or np.ptp(y) <= _ZERO_VAR_TOL:
            self.scale_ = np.ones(c, dtype=np.float64)
            self.coef_ = np.zeros(int(np.sum(self.active_)), dtype=np.float64)
            return np.full_like(y, self.y_mean_, dtype=np.float64)

        safe_std = np.where(ptp <= 1.0 + 1e-6, np.maximum(raw_std, self.min_binary_scale), raw_std)
        self.scale_ = np.where(self.active_, safe_std, 1.0)
        Z = np.clip(
            (Xl[:, self.active_] - self.mean_[self.active_]) / self.scale_[self.active_],
            -8.0,
            8.0,
        )

        yc = y - self.y_mean_
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s2 = s ** 2
        uty = U.T @ yc
        n = float(len(y))

        best_mse = float("inf")
        best_oof = np.full_like(y, self.y_mean_, dtype=np.float64)
        best_alpha = 100.0
        for alpha in np.geomspace(max(0.5, 0.01 * n), 80.0 * n, 28):
            filt = s2 / (s2 + alpha)
            y_hat_c = U @ (filt * uty)
            h_diag = np.clip((1.0 / n) + (U ** 2) @ filt, 0.0, 0.995)
            oof_res = (yc - y_hat_c) / (1.0 - h_diag)
            mse = float(np.mean(oof_res ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_oof = y - oof_res

        self.alpha_ = best_alpha
        self.coef_ = Vt.T @ ((s / (s2 + best_alpha)) * uty)
        return best_oof

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_.size == 0 or not np.any(self.active_):
            return np.full(X.shape[0], self.y_mean_, dtype=np.float64)
        c = min(self.n_cols, X.shape[1])
        Xl = X[:, :c]
        Z = np.clip(
            (Xl[:, self.active_] - self.mean_[self.active_]) / self.scale_[self.active_],
            -8.0,
            8.0,
        )
        return Z @ self.coef_ + self.y_mean_


class _DualSpaceKernelRidge:
    """Dual-space (2D descriptor RBF/RQ + Morgan Tanimoto/RBF) Kernel Ridge with exact Cholesky LOOCV."""

    def __init__(
        self,
        n_dm: int,
        alpha: float = 0.04,
        gamma_scale: float = 1.0,
        desc_weight: float = 0.50,
        min_binary_scale: float = 0.10,
    ) -> None:
        self.n_dm = int(n_dm)
        self.alpha = float(alpha)
        self.gamma_scale = float(gamma_scale)
        self.desc_weight = float(desc_weight)
        self.min_binary_scale = float(min_binary_scale)
        self.mean_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.scale_: np.ndarray = np.ones(0, dtype=np.float64)
        self.Zd_ref_: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.Zf_ref_: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.dual_coef_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.gamma_d_: float = 0.01
        self.gamma_f_: float = 2.0
        self.y_mean_: float = 0.0
        self.has_fp_: bool = False

    def _prep(self, X: np.ndarray, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        dm = min(self.n_dm, X.shape[1])
        Xd = X[:, :dm]
        Xf = X[:, dm:]
        if fit:
            self.has_fp_ = Xf.shape[1] > 0
            self.mean_ = np.mean(Xd, axis=0)
            raw_std = np.std(Xd, axis=0)
            ptp = np.ptp(Xd, axis=0)
            safe_std = np.where(ptp <= 1.0 + 1e-6, np.maximum(raw_std, self.min_binary_scale), raw_std)
            self.scale_ = np.where(raw_std > 1e-8, safe_std, 1.0)
        Zd = np.clip((Xd - self.mean_) / self.scale_, -10.0, 10.0)
        return Zd, Xf

    def _kernel(
        self,
        Zd1: np.ndarray,
        Xf1: np.ndarray,
        Zd2: np.ndarray,
        Xf2: np.ndarray,
    ) -> np.ndarray:
        sq1 = np.sum(Zd1 ** 2, axis=1, keepdims=True)
        sq2 = np.sum(Zd2 ** 2, axis=1, keepdims=True).T
        d2_d = np.maximum(sq1 + sq2 - 2.0 * (Zd1 @ Zd2.T), 0.0)
        K_d = 0.65 * np.exp(-self.gamma_d_ * d2_d) + 0.35 / (1.0 + 0.5 * self.gamma_d_ * d2_d)

        if not self.has_fp_ or Xf1.shape[1] == 0:
            return K_d

        dot = Xf1 @ Xf2.T
        n1 = np.sum(Xf1 ** 2, axis=1, keepdims=True)
        n2 = np.sum(Xf2 ** 2, axis=1, keepdims=True).T
        tan = dot / np.maximum(n1 + n2 - dot, 1e-8)
        K_f = 0.60 * tan + 0.40 * np.exp(-self.gamma_f_ * np.maximum(1.0 - tan, 0.0))
        return self.desc_weight * K_d + (1.0 - self.desc_weight) * K_f

    def fit_with_oof(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
        Zd, Xf = self._prep(X, fit=True)
        n = Zd.shape[0]
        self.y_mean_ = float(np.mean(y))
        if n < 2 or np.ptp(y) <= _ZERO_VAR_TOL:
            self.Zd_ref_ = Zd
            self.Zf_ref_ = Xf
            self.dual_coef_ = np.zeros(n, dtype=np.float64)
            return np.full(n, self.y_mean_, dtype=np.float64)

        sub = Zd[: min(n, 512)]
        s1 = np.sum(sub ** 2, axis=1, keepdims=True)
        d2 = np.maximum(s1 + s1.T - 2.0 * (sub @ sub.T), 0.0)
        tri = d2[np.triu_indices(len(sub), k=1)]
        med_d2 = float(np.median(tri[tri > 1e-8])) if np.any(tri > 1e-8) else float(max(Zd.shape[1], 1))
        self.gamma_d_ = self.gamma_scale / max(med_d2, 1.0)
        self.gamma_f_ = 2.2 * self.gamma_scale

        yc = y - self.y_mean_

        if n <= 3000:
            self.Zd_ref_ = Zd
            self.Zf_ref_ = Xf
            K = self._kernel(Zd, Xf, Zd, Xf)
            K[np.diag_indices_from(K)] += self.alpha
            c_fac = cho_factor(K, lower=True, check_finite=False)
            self.dual_coef_ = cho_solve(c_fac, yc, check_finite=False)
            L_inv = solve_triangular(c_fac[0], np.eye(n), lower=True, check_finite=False)
            diag_inv = np.sum(L_inv ** 2, axis=0)
            return y - self.dual_coef_ / np.maximum(diag_inv, 1e-8)

        rng = np.random.default_rng(seed + 42)
        m_sub = min(1536, n)
        idx = np.sort(rng.choice(n, size=m_sub, replace=False))
        self.Zd_ref_ = Zd[idx]
        self.Zf_ref_ = Xf[idx]
        K_mm = self._kernel(self.Zd_ref_, self.Zf_ref_, self.Zd_ref_, self.Zf_ref_)
        K_mm[np.diag_indices_from(K_mm)] += 1e-6
        K_nm = self._kernel(Zd, Xf, self.Zd_ref_, self.Zf_ref_)
        A = K_nm.T @ K_nm + self.alpha * K_mm
        c_fac = cho_factor(A, lower=True, check_finite=False)
        self.dual_coef_ = cho_solve(c_fac, K_nm.T @ yc, check_finite=False)
        A_inv_K = cho_solve(c_fac, K_nm.T, check_finite=False)
        h_diag = np.clip(np.sum(K_nm.T * A_inv_K, axis=0), 0.0, 0.995)
        fit_pred = K_nm @ self.dual_coef_ + self.y_mean_
        return y - (y - fit_pred) / (1.0 - h_diag)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.dual_coef_.size == 0:
            return np.full(X.shape[0], self.y_mean_, dtype=np.float64)
        Zd, Xf = self._prep(X, fit=False)
        return self._kernel(Zd, Xf, self.Zd_ref_, self.Zf_ref_) @ self.dual_coef_ + self.y_mean_


def _infer_molecular_block_layout(n_features: int) -> tuple[int, int]:
    """Infer `(n_core_phys, n_desc_maccs)` layout from `X.shape[1]`."""
    if n_features > _DEFAULT_MORGAN_TAIL_BITS + 32:
        n_dm = n_features - _DEFAULT_MORGAN_TAIL_BITS
        n_core = min(_DEFAULT_CORE_PHYS_COLS, n_dm)
        return n_core, n_dm
    return min(_DEFAULT_CORE_PHYS_COLS, n_features), n_features


def _make_tree_regressor(n_samples: int, seed: int) -> Any:
    min_child = max(2, min(15, n_samples // 25))
    try:
        import lightgbm as lgb

        n_threads = max(1, min(4, os.cpu_count() or 1))
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=260,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.35,
            max_bin=63,
            min_child_samples=min_child,
            reg_alpha=0.05,
            reg_lambda=1.2,
            random_state=int(seed),
            n_jobs=n_threads,
            verbosity=-1,
        )
    except ImportError:
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=min_child,
            l2_regularization=1.0,
            random_state=int(seed),
        )


def _make_tree_quantile_regressor(n_samples: int, quantile_alpha: float, seed: int) -> Any:
    min_child = max(3, min(20, max(1, n_samples // 15)))
    try:
        import lightgbm as lgb

        n_threads = max(1, min(4, os.cpu_count() or 1))
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(quantile_alpha),
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.35,
            max_bin=63,
            min_child_samples=min_child,
            random_state=int(seed),
            n_jobs=n_threads,
            verbosity=-1,
        )
    except ImportError:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile_alpha),
            max_iter=140,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=min_child,
            random_state=int(seed),
        )


class FrontierEnsembleRegressor:
    """Semiparametric out-of-fold stacked blend for molecular & tabular conformal regression.

    Combines 7 complementary learners strictly on `(X_tr, y_tr)`:
    1. Core physicochemical analytic SVD Ridge (stabilizes scaffold-shift linear extrapolation),
    2. Full 2D+MACCS analytic SVD Ridge,
    3. Sharp Universal-Kriging Dual-Space Kernel Ridge (RBF/RQ 2D + Tanimoto/RBF Morgan on linear residuals),
    4. Smooth Universal-Kriging Dual-Space Kernel Ridge,
    5. Direct Dual-Space Kernel Ridge (local support anchor whose divergence flags OOD scaffolds),
    6. Residual Gradient-Boosted Trees (LightGBM / HistGBDT on linear-trend residuals),
    7. Direct Gradient-Boosted Trees (LightGBM / HistGBDT).
    """

    def __init__(self, *, seed: int = 0, lin_damp: float = 0.88) -> None:
        self.seed = int(seed)
        self.lin_damp = float(lin_damp)
        self.n_core_: int = 0
        self.n_dm_: int = 0
        self.active_cols_: np.ndarray = np.zeros(0, dtype=int)
        self.y_mean_: float = 0.0
        self.weights_: np.ndarray = np.ones(1, dtype=np.float64)
        self.oof_pred_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.core_lin_: _AnalyticSVDRidge | None = None
        self.full_lin_: _AnalyticSVDRidge | None = None
        self.uk_sharp_: _DualSpaceKernelRidge | None = None
        self.uk_smooth_: _DualSpaceKernelRidge | None = None
        self.dk_direct_: _DualSpaceKernelRidge | None = None
        self.res_tree_: Any | None = None
        self.dir_tree_: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrontierEnsembleRegressor":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n, d = X_arr.shape
        self.n_core_, self.n_dm_ = _infer_molecular_block_layout(d)
        self.y_mean_ = float(np.mean(y_arr))

        std_all = np.std(X_arr, axis=0)
        act = np.where(std_all > _ZERO_VAR_TOL)[0]
        self.active_cols_ = act if act.size > 0 else np.arange(d, dtype=int)
        X_act = X_arr[:, self.active_cols_]

        self.core_lin_ = _AnalyticSVDRidge(n_cols=self.n_core_, min_binary_scale=0.50)
        oof_core = self.core_lin_.fit_with_oof(X_arr, y_arr)
        fit_core = self.core_lin_.predict(X_arr)

        self.full_lin_ = _AnalyticSVDRidge(n_cols=self.n_dm_, min_binary_scale=0.15)
        oof_full = self.full_lin_.fit_with_oof(X_arr, y_arr)
        fit_full = self.full_lin_.predict(X_arr)

        blend_fit_lin = 0.35 * fit_core + 0.65 * fit_full
        blend_oof_lin = 0.35 * oof_core + 0.65 * oof_full
        base_tr = self.lin_damp * blend_fit_lin + (1.0 - self.lin_damp) * self.y_mean_
        base_oof = self.lin_damp * blend_oof_lin + (1.0 - self.lin_damp) * self.y_mean_
        res_y = y_arr - base_tr

        self.uk_sharp_ = _DualSpaceKernelRidge(
            self.n_dm_, alpha=0.035, gamma_scale=1.15, desc_weight=0.50, min_binary_scale=0.05
        )
        oof_uk1 = base_oof + self.uk_sharp_.fit_with_oof(X_arr, res_y, seed=self.seed)

        self.uk_smooth_ = _DualSpaceKernelRidge(
            self.n_dm_, alpha=0.18, gamma_scale=0.50, desc_weight=0.65, min_binary_scale=0.30
        )
        oof_uk2 = base_oof + self.uk_smooth_.fit_with_oof(X_arr, res_y, seed=self.seed + 5)

        self.dk_direct_ = _DualSpaceKernelRidge(
            self.n_dm_, alpha=0.045, gamma_scale=1.0, desc_weight=0.50, min_binary_scale=0.05
        )
        oof_dk = self.dk_direct_.fit_with_oof(X_arr, y_arr, seed=self.seed + 11)

        if n >= 6 and np.ptp(y_arr) > _ZERO_VAR_TOL:
            oof_rt = np.zeros(n, dtype=np.float64)
            oof_dt = np.zeros(n, dtype=np.float64)
            n_splits = min(3, max(2, n // 3))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            for fold_i, (ti, vi) in enumerate(kf.split(X_act)):
                cf = _AnalyticSVDRidge(n_cols=self.n_core_, min_binary_scale=0.50)
                cf.fit_with_oof(X_arr[ti], y_arr[ti])
                ff = _AnalyticSVDRidge(n_cols=self.n_dm_, min_binary_scale=0.15)
                ff.fit_with_oof(X_arr[ti], y_arr[ti])
                ym_f = float(np.mean(y_arr[ti]))
                bl_tr = 0.35 * cf.predict(X_arr[ti]) + 0.65 * ff.predict(X_arr[ti])
                bl_va = 0.35 * cf.predict(X_arr[vi]) + 0.65 * ff.predict(X_arr[vi])
                bt_f = self.lin_damp * bl_tr + (1.0 - self.lin_damp) * ym_f
                bv_f = self.lin_damp * bl_va + (1.0 - self.lin_damp) * ym_f

                m_r = _make_tree_regressor(len(ti), self.seed * 19 + fold_i).fit(X_act[ti], y_arr[ti] - bt_f)
                oof_rt[vi] = bv_f + np.asarray(m_r.predict(X_act[vi]), dtype=np.float64)

                m_d = _make_tree_regressor(len(ti), self.seed * 29 + fold_i).fit(X_act[ti], y_arr[ti])
                oof_dt[vi] = np.asarray(m_d.predict(X_act[vi]), dtype=np.float64)

            self.res_tree_ = _make_tree_regressor(n, self.seed).fit(X_act, res_y)
            self.dir_tree_ = _make_tree_regressor(n, self.seed + 101).fit(X_act, y_arr)
            oof_mat = np.column_stack([oof_core, oof_full, oof_uk1, oof_uk2, oof_dk, oof_rt, oof_dt])
        else:
            self.res_tree_ = None
            self.dir_tree_ = None
            oof_mat = np.column_stack([oof_core, oof_full, oof_uk1, oof_uk2, oof_dk])

        m_cnt = oof_mat.shape[1]

        def _stack_loss(w: np.ndarray) -> float:
            err = y_arr - oof_mat @ w
            return float(np.mean(np.abs(err)) + 0.30 * np.sqrt(np.mean(err ** 2)))

        opt = minimize(
            _stack_loss,
            x0=np.full(m_cnt, 1.0 / float(m_cnt)),
            bounds=[(0.0, 1.0)] * m_cnt,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        )
        w_raw = np.maximum(np.asarray(opt.x, dtype=np.float64), 0.0)
        w_sum = float(np.sum(w_raw))
        self.weights_ = w_raw / w_sum if w_sum > _ZERO_VAR_TOL else np.full(m_cnt, 1.0 / float(m_cnt))
        self.oof_pred_ = oof_mat @ self.weights_
        return self

    def predict_components(self, X: np.ndarray) -> np.ndarray:
        assert (
            self.core_lin_ is not None
            and self.full_lin_ is not None
            and self.uk_sharp_ is not None
            and self.uk_smooth_ is not None
            and self.dk_direct_ is not None
        )
        X_arr = np.asarray(X, dtype=np.float64)
        p_core = self.core_lin_.predict(X_arr)
        p_full = self.full_lin_.predict(X_arr)
        b = self.lin_damp * (0.35 * p_core + 0.65 * p_full) + (1.0 - self.lin_damp) * self.y_mean_
        rows = [
            p_core,
            p_full,
            b + self.uk_sharp_.predict(X_arr),
            b + self.uk_smooth_.predict(X_arr),
            self.dk_direct_.predict(X_arr),
        ]
        if self.res_tree_ is not None and self.dir_tree_ is not None:
            X_act = X_arr[:, self.active_cols_]
            rows.append(b + np.asarray(self.res_tree_.predict(X_act), dtype=np.float64))
            rows.append(np.asarray(self.dir_tree_.predict(X_act), dtype=np.float64))
        return np.vstack(rows)

    def epistemic_spread(self, X: np.ndarray) -> np.ndarray:
        """Compute cross-family epistemic disagreement (`std` across linear/kernel/tree families)."""
        comps = self.predict_components(X)
        if comps.shape[0] <= 2:
            return np.std(comps, axis=0)
        return np.std(comps[1:], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        comps = self.predict_components(X)
        return np.asarray((self.weights_[:, None] * comps).sum(axis=0), dtype=np.float64)


class ResidualQuantileBandModel:
    """Picklable centered heteroscedastic band predictor `pred(X) +- half_band(X)` for `FrontierEnsembleRegressor`."""

    def __init__(
        self,
        *,
        base_model: FrontierEnsembleRegressor,
        q_lo_res_model: Any,
        q_hi_res_model: Any,
        sign: float,
        band_floor: float,
        alpha: float,
    ) -> None:
        self.base_model = base_model
        self.q_lo_res_model = q_lo_res_model
        self.q_hi_res_model = q_hi_res_model
        self.sign = float(sign)
        self.band_floor = float(band_floor)
        self.alpha = float(alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        comps = self.base_model.predict_components(X_arr)
        pred = np.asarray((self.base_model.weights_[:, None] * comps).sum(axis=0), dtype=np.float64)
        ens_std = np.std(comps[1:], axis=0) if comps.shape[0] > 2 else np.std(comps, axis=0)
        X_act = X_arr[:, self.base_model.active_cols_]
        q_lo = np.asarray(self.q_lo_res_model.predict(X_act), dtype=np.float64)
        q_hi = np.asarray(self.q_hi_res_model.predict(X_act), dtype=np.float64)
        q_span = np.maximum(q_hi - q_lo, 0.0)
        half_band = np.maximum(0.5 * q_span + 0.50 * ens_std, self.band_floor)
        return pred + self.sign * half_band


def _continuous_tanimoto_knn(
    Xf_query: np.ndarray,
    Xf_ref: np.ndarray,
    *,
    k_eff: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Compute mean continuous Tanimoto (Ruzicka/Jaccard) kNN distance in `[0, 1]`."""
    n_q = Xf_query.shape[0]
    n_r = Xf_ref.shape[0]
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)
    if Xf_query.shape[1] == 0 or n_r == 0 or (exclude_self and n_r <= 1):
        return np.zeros(n_q, dtype=np.float64)

    dot = Xf_query @ Xf_ref.T
    n1 = np.sum(Xf_query ** 2, axis=1, keepdims=True)
    n2 = np.sum(Xf_ref ** 2, axis=1, keepdims=True).T
    denom = n1 + n2 - dot
    # Two all-zero fingerprint vectors have Tanimoto distance 0.0
    both_zero = (n1 <= _ZERO_VAR_TOL) & (n2 <= _ZERO_VAR_TOL)
    tan_sim = np.where(both_zero, 1.0, dot / np.maximum(denom, 1e-8))
    tan_dist = np.clip(1.0 - tan_sim, 0.0, 1.0)

    if exclude_self:
        k_loo = min(k_eff + 1, n_r)
        part = np.partition(tan_dist, k_loo - 1, axis=1)[:, :k_loo]
        # Drop the smallest (self-distance = 0) entry per row
        row_max_drop = np.sort(part, axis=1)[:, 1:]
        return np.asarray(row_max_drop.mean(axis=1), dtype=np.float64)

    k_use = min(k_eff, n_r)
    part = np.partition(tan_dist, k_use - 1, axis=1)[:, :k_use]
    return np.asarray(part.mean(axis=1), dtype=np.float64)


@dataclass
class ConformalTrustModel:
    """Picklable conformal regression & kNN / Tanimoto-ensemble applicability-domain advisor."""

    model: Any
    q_marginal: float
    q_in: float
    q_out: float
    threshold: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    feature_weights: np.ndarray
    nn: NearestNeighbors
    feature_names: list[str]
    alpha: float = 0.10
    trust_pct: float = 80.0
    k_eff: int = 5
    nonconformity: str = "normalized"
    gamma: float = 0.50
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75)
    lower_model: Any | None = None
    upper_model: Any | None = None
    cqr_scale_floor: float = 1e-3
    constant_feature_mask: np.ndarray | None = None
    mondrian_fallback_used: bool = False
    distance_metric: str = "euclidean"
    whitening_matrix: np.ndarray | None = None
    ensemble_models: list[Any] | None = None
    knn_norm_scale: float = 1.0
    spread_norm_scale: float = 1.0
    tanimoto_ref_fp: np.ndarray | None = None
    tanimoto_norm_scale: float = 1.0
    n_input_features: int | None = None
    gate_n_features: int | None = None

    @property
    def expected_input_dim(self) -> int:
        if self.n_input_features is not None:
            return int(self.n_input_features)
        return int(self.scaler_mean.size)

    @property
    def q(self) -> float:
        """Selective inside-trust-region conformal quantile."""
        return self.q_in

    def __iter__(self) -> Iterator[Any]:
        """Support legacy tuple unpacking `model, q, dist, thr = fit_and_calibrate(...)`."""
        yield self.model
        yield self.q_in
        yield self.distance
        yield self.threshold

    def _slice_gate_features(self, arr2d: np.ndarray) -> np.ndarray:
        g_dim = int(self.gate_n_features) if self.gate_n_features is not None else int(self.scaler_mean.size)
        return arr2d[:, :g_dim]

    def _constant_invariant_violated(self, arr2d: np.ndarray) -> np.ndarray:
        if self.constant_feature_mask is None or not np.any(self.constant_feature_mask):
            return np.zeros(arr2d.shape[0], dtype=bool)
        g_arr = self._slice_gate_features(arr2d)
        cmask = np.asarray(self.constant_feature_mask, dtype=bool)
        diffs = np.abs(g_arr[:, cmask] - self.scaler_mean[cmask])
        return np.any(diffs > _CONST_INVARIANT_TOL, axis=1)

    def transform_features(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        raw_arr = np.asarray(X, dtype=np.float64)
        g_dim = int(self.gate_n_features) if self.gate_n_features is not None else int(self.scaler_mean.size)
        if (raw_arr.ndim == 1 and raw_arr.size == g_dim) or (raw_arr.ndim == 2 and raw_arr.shape[1] == g_dim):
            arr2d, _ = _ensure_2d(raw_arr, expected_dim=g_dim)
            g_arr = arr2d
        else:
            arr2d, _ = _ensure_2d(raw_arr, expected_dim=self.expected_input_dim)
            g_arr = self._slice_gate_features(arr2d)
        standardized = (g_arr - self.scaler_mean) / self.scaler_scale
        if self.whitening_matrix is not None:
            return standardized @ self.whitening_matrix
        if self.distance_metric == "tanimoto_mahalanobis":
            return np.clip(standardized, -8.0, 8.0) * self.feature_weights
        return standardized * self.feature_weights

    def _epistemic_spread_vec(self, arr2d: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "epistemic_spread"):
            return np.asarray(self.model.epistemic_spread(arr2d), dtype=np.float64)
        if self.ensemble_models:
            ens_preds = np.vstack(
                [np.asarray(m.predict(arr2d), dtype=np.float64) for m in self.ensemble_models]
            )
            return np.std(ens_preds, axis=0)
        return np.zeros(arr2d.shape[0], dtype=np.float64)

    def distance(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        """Compute applicability-domain distance (Euclidean, Mahalanobis, ensemble, or Tanimoto-ensemble)."""
        raw_arr = np.asarray(X, dtype=np.float64)
        g_dim = int(self.gate_n_features) if self.gate_n_features is not None else int(self.scaler_mean.size)
        if (
            self.gate_n_features is not None
            and self.distance_metric in ("euclidean", "mahalanobis")
            and ((raw_arr.ndim == 1 and raw_arr.size == g_dim) or (raw_arr.ndim == 2 and raw_arr.shape[1] == g_dim))
        ):
            arr2d, _ = _ensure_2d(raw_arr, expected_dim=g_dim)
        else:
            arr2d, _ = _ensure_2d(raw_arr, expected_dim=self.expected_input_dim)
        if arr2d.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)

        d_knn, _ = self.nn.kneighbors(self.transform_features(arr2d))
        knn_mean = np.asarray(d_knn.mean(axis=1), dtype=np.float64)

        if self.distance_metric == "ensemble_mahalanobis":
            spread = self._epistemic_spread_vec(arr2d)
            return knn_mean / max(float(self.knn_norm_scale), 1e-8) + spread / max(
                float(self.spread_norm_scale), 1e-8
            )

        if self.distance_metric == "tanimoto_mahalanobis":
            spread = self._epistemic_spread_vec(arr2d)
            kd_part = knn_mean / max(float(self.knn_norm_scale), 1e-8)
            sp_part = spread / max(float(self.spread_norm_scale), 1e-8)
            if self.tanimoto_ref_fp is not None and self.tanimoto_ref_fp.shape[1] > 0:
                fp_start = arr2d.shape[1] - self.tanimoto_ref_fp.shape[1]
                kt = _continuous_tanimoto_knn(
                    arr2d[:, fp_start:],
                    self.tanimoto_ref_fp,
                    k_eff=self.k_eff,
                    exclude_self=False,
                )
                kt_part = kt / max(float(self.tanimoto_norm_scale), 1e-8)
                return 0.12 * kd_part + 0.23 * kt_part + 0.65 * sp_part
            return 0.50 * kd_part + 0.50 * sp_part

        return knn_mean

    def _raw_intervals(
        self,
        arr2d: np.ndarray,
        dists: np.ndarray,
        *,
        use_mondrian: bool = True,
        return_unclipped: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        pred = np.asarray(self.model.predict(arr2d), dtype=np.float64)
        n = arr2d.shape[0]
        if use_mondrian:
            const_viol = self._constant_invariant_violated(arr2d)
            inside = (dists <= self.threshold) & (~const_viol)
            q_vec = np.where(inside, self.q_in, self.q_out)
        else:
            q_vec = np.full(n, self.q_marginal, dtype=np.float64)

        if self.nonconformity == "absolute":
            half = np.maximum(q_vec, 0.0)
            lo = pred - half
            hi = pred + half
        elif self.nonconformity == "normalized":
            scale = 1.0 + self.gamma * dists
            half = np.maximum(q_vec, 0.0) * scale
            lo = pred - half
            hi = pred + half
        elif self.nonconformity in ("cqr", "cqr_r"):
            if self.lower_model is None or self.upper_model is None:
                raise RuntimeError("CQR models are missing on ConformalTrustModel")
            q_lo = np.asarray(self.lower_model.predict(arr2d), dtype=np.float64)
            q_hi = np.asarray(self.upper_model.predict(arr2d), dtype=np.float64)
            lo_raw = np.minimum(q_lo, q_hi)
            hi_raw = np.maximum(q_lo, q_hi)
            if self.nonconformity == "cqr_r":
                spread = np.maximum(hi_raw - lo_raw, float(self.cqr_scale_floor))
                adj = q_vec * spread
            else:
                adj = q_vec

            lo_cand = lo_raw - adj
            hi_cand = hi_raw + adj
            mid_raw = 0.5 * (lo_raw + hi_raw)
            crossed = np.isfinite(lo_cand) & np.isfinite(hi_cand) & (lo_cand > hi_cand)
            lo = np.where(crossed, mid_raw, lo_cand)
            hi = np.where(crossed, mid_raw, hi_cand)
            finite_cqr = np.isfinite(lo) & np.isfinite(hi)
            pred = np.where(finite_cqr, np.clip(pred, lo, hi), pred)
        else:
            raise ValueError(f"Unsupported nonconformity mode: {self.nonconformity!r}")

        pred_unclipped = pred.copy()
        lo_unclipped = lo.copy()
        hi_unclipped = hi.copy()

        # Clip ONLY where intervals are finite; infinite intervals indicate uncertifiable small-n calibration
        if self.domain_bounds is not None:
            y_min, y_max = float(self.domain_bounds[0]), float(self.domain_bounds[1])
            finite_mask = np.isfinite(lo) & np.isfinite(hi)
            lo = np.where(finite_mask, np.clip(lo, y_min, y_max), lo)
            hi = np.where(finite_mask, np.clip(hi, y_min, y_max), hi)
            hi = np.where(finite_mask, np.maximum(lo, hi), hi)
            pred = np.clip(pred, y_min, y_max)
            pred = np.where(finite_mask, np.clip(pred, lo, hi), pred)

        if return_unclipped:
            return pred, lo, hi, pred_unclipped, lo_unclipped, hi_unclipped
        return pred, lo, hi

    def predict(
        self,
        X: np.ndarray | Sequence[float],
        *,
        use_mondrian: bool = True,
        include_unclipped: bool = False,
    ) -> dict[str, Any]:
        """Predict point estimates, conformal intervals, distances, and trust decisions on 1D or 2D X."""
        arr2d, is_1d = _ensure_2d(X, expected_dim=self.expected_input_dim)
        if arr2d.shape[0] == 0:
            empty = np.zeros(0, dtype=np.float64)
            out_empty: dict[str, Any] = {
                "pred": empty,
                "lo": empty,
                "hi": empty,
                "width": empty,
                "distance": empty,
                "trusted": np.zeros(0, dtype=bool),
                "decision": [],
            }
            if include_unclipped:
                out_empty["pred_unclipped"] = empty
                out_empty["lo_unclipped"] = empty
                out_empty["hi_unclipped"] = empty
            return out_empty

        dists = self.distance(arr2d)
        const_viol = self._constant_invariant_violated(arr2d)
        raw_iv = self._raw_intervals(
            arr2d, dists, use_mondrian=use_mondrian, return_unclipped=include_unclipped
        )
        if include_unclipped:
            pred, lo, hi, pred_u, lo_u, hi_u = raw_iv  # type: ignore[misc]
        else:
            pred, lo, hi = raw_iv  # type: ignore[misc]
            pred_u = lo_u = hi_u = pred

        finite_iv = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        trusted = (dists <= self.threshold) & finite_iv & (~const_viol)
        width = hi - lo
        decisions = ["ANSWER" if bool(t) else "ABSTAIN_ACQUIRE" for t in trusted]

        if is_1d:
            out_1d: dict[str, Any] = {
                "pred": float(pred[0]),
                "lo": float(lo[0]),
                "hi": float(hi[0]),
                "width": float(width[0]),
                "distance": float(dists[0]),
                "trusted": bool(trusted[0]),
                "decision": decisions[0],
            }
            if include_unclipped:
                out_1d["pred_unclipped"] = float(pred_u[0])
                out_1d["lo_unclipped"] = float(lo_u[0])
                out_1d["hi_unclipped"] = float(hi_u[0])
            return out_1d

        out_2d: dict[str, Any] = {
            "pred": pred,
            "lo": lo,
            "hi": hi,
            "width": width,
            "distance": dists,
            "trusted": trusted,
            "decision": decisions,
        }
        if include_unclipped:
            out_2d["pred_unclipped"] = pred_u
            out_2d["lo_unclipped"] = lo_u
            out_2d["hi_unclipped"] = hi_u
        return out_2d


def _compute_feature_weights(
    X_tr: np.ndarray,
    model: Any,
    mode: str,
    *,
    raw_std: np.ndarray | None = None,
) -> np.ndarray:
    d = X_tr.shape[1]
    std = np.std(X_tr, axis=0) if raw_std is None else np.asarray(raw_std, dtype=np.float64)
    const_mask = std <= _ZERO_VAR_TOL

    if mode == "uniform":
        return np.ones(d, dtype=np.float64)
    if mode == "importance":
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and len(imp) == d and float(np.sum(imp)) > _ZERO_VAR_TOL:
            w = np.sqrt(np.maximum(np.asarray(imp, dtype=np.float64), 0.0))
            norm = math.sqrt(float(np.mean(w ** 2)))
            scaled = w / norm if norm > _ZERO_VAR_TOL else np.ones(d, dtype=np.float64)
            scaled = np.maximum(scaled, _MIN_ACTIVE_FEATURE_WEIGHT)
            return np.where(const_mask, 1.0, scaled)
        return np.ones(d, dtype=np.float64)
    if mode == "iqr":
        q75 = np.percentile(X_tr, 75, axis=0)
        q25 = np.percentile(X_tr, 25, axis=0)
        iqr = q75 - q25
        return np.where(const_mask, 1.0, np.where(iqr > _ZERO_VAR_TOL, 1.0, 0.20))
    raise ValueError(f"Unknown distance_weighting {mode!r}; expected 'importance', 'uniform', or 'iqr'")


def _fit_point_surrogate(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    surrogate: str,
    loss: str,
    seed: int,
) -> Any:
    eff_loss = loss
    if eff_loss == "huber" and (X_tr.shape[0] < 3 or np.ptp(y_tr) < _ZERO_VAR_TOL):
        eff_loss = "squared_error"

    if surrogate == "gbdt":
        return GradientBoostingRegressor(loss=eff_loss, random_state=seed).fit(X_tr, y_tr)
    if surrogate == "hist_gbdt":
        h_loss = "squared_error" if eff_loss == "squared_error" else "absolute_error"
        return HistGradientBoostingRegressor(loss=h_loss, random_state=seed).fit(X_tr, y_tr)
    if surrogate == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for surrogate='lightgbm'; install `pip install lightgbm`."
            ) from exc
        obj = "huber" if eff_loss == "huber" else "regression"
        min_child = max(1, min(10, X_tr.shape[0] // 5))
        return lgb.LGBMRegressor(
            objective=obj,
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        ).fit(X_tr, y_tr)
    if surrogate == "frontier_ensemble":
        return FrontierEnsembleRegressor(seed=seed).fit(X_tr, y_tr)
    raise ValueError(
        f"Unknown surrogate {surrogate!r}; expected 'gbdt', 'hist_gbdt', 'lightgbm', or 'frontier_ensemble'"
    )


def _fit_quantile_surrogates(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    surrogate: str,
    lo_alpha: float,
    hi_alpha: float,
    seed: int,
    point_model: Any | None = None,
    cqr_scale_floor: float = 1e-3,
) -> tuple[Any, Any]:
    if surrogate == "gbdt":
        lo_m = GradientBoostingRegressor(loss="quantile", alpha=lo_alpha, random_state=seed).fit(X_tr, y_tr)
        hi_m = GradientBoostingRegressor(loss="quantile", alpha=hi_alpha, random_state=seed).fit(X_tr, y_tr)
        return lo_m, hi_m
    if surrogate == "hist_gbdt":
        lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=lo_alpha, random_state=seed).fit(X_tr, y_tr)
        hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=hi_alpha, random_state=seed).fit(X_tr, y_tr)
        return lo_m, hi_m
    if surrogate == "lightgbm":
        import lightgbm as lgb

        min_child = max(1, min(10, X_tr.shape[0] // 5))
        lo_m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=lo_alpha,
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        ).fit(X_tr, y_tr)
        hi_m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=hi_alpha,
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=min_child,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        ).fit(X_tr, y_tr)
        return lo_m, hi_m
    if surrogate == "frontier_ensemble":
        ens = (
            point_model
            if isinstance(point_model, FrontierEnsembleRegressor)
            else FrontierEnsembleRegressor(seed=seed).fit(X_tr, y_tr)
        )
        oof_res = y_tr - ens.oof_pred_
        X_act = X_tr[:, ens.active_cols_]
        n_tr = X_tr.shape[0]
        q_lo_m = _make_tree_quantile_regressor(n_tr, lo_alpha, seed).fit(X_act, oof_res)
        q_hi_m = _make_tree_quantile_regressor(n_tr, hi_alpha, seed + 13).fit(X_act, oof_res)
        y_sd = float(np.std(y_tr))
        band_floor = max(float(cqr_scale_floor), 0.06 * y_sd if y_sd > _ZERO_VAR_TOL else float(cqr_scale_floor))
        lo_band = ResidualQuantileBandModel(
            base_model=ens,
            q_lo_res_model=q_lo_m,
            q_hi_res_model=q_hi_m,
            sign=-1.0,
            band_floor=band_floor,
            alpha=lo_alpha,
        )
        hi_band = ResidualQuantileBandModel(
            base_model=ens,
            q_lo_res_model=q_lo_m,
            q_hi_res_model=q_hi_m,
            sign=+1.0,
            band_floor=band_floor,
            alpha=hi_alpha,
        )
        return lo_band, hi_band
    raise ValueError(f"Unknown surrogate {surrogate!r}")


def _compute_mahalanobis_whitener(Z_tr: np.ndarray, const_mask: np.ndarray) -> np.ndarray:
    """Compute regularized Mahalanobis whitening matrix strictly from standardized `Z_tr`."""
    n_tr, d = Z_tr.shape
    if n_tr < 2:
        return np.eye(d, dtype=np.float64)
    active = ~const_mask
    W_full = np.zeros((d, d), dtype=np.float64)
    # Preserve unit weight on constant training columns so unseen OOD shifts trigger large distances
    for j in np.where(const_mask)[0]:
        W_full[int(j), int(j)] = 1.0
    if np.any(active):
        act_idx = np.where(active)[0]
        Z_act = Z_tr[:, act_idx]
        _, s_vals, vt = np.linalg.svd(Z_act, full_matrices=False)
        eig = (s_vals ** 2) / float(max(n_tr - 1, 1))
        inv_sqrt = 1.0 / np.sqrt(eig + _MAHALANOBIS_RIDGE)
        W_act = (vt.T * inv_sqrt) @ vt
        if vt.shape[0] < len(act_idx):
            # Complete orthogonal complement with 1/sqrt(ridge) so unseen nullspace OOD shifts are never zeroed!
            ortho_scale = 1.0 / math.sqrt(_MAHALANOBIS_RIDGE)
            W_act = ortho_scale * np.eye(len(act_idx), dtype=np.float64) + (
                vt.T * (inv_sqrt - ortho_scale)
            ) @ vt
        W_full[np.ix_(act_idx, act_idx)] = W_act
    return W_full


def fit_and_calibrate(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    alpha: float = 0.10,
    k: int = 5,
    trust_pct: float = 80.0,
    seed: int = 0,
    loss: str = "huber",
    nonconformity: str = "normalized",
    distance_weighting: str = "importance",
    distance_metric: str = "euclidean",
    surrogate: str = "gbdt",
    gamma: float = 0.50,
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75),
    feature_names: Sequence[str] | None = None,
    cal_weights: Sequence[float] | np.ndarray | None = None,
    test_weight: float = 1.0,
    mondrian_fallback: str = "global_pool",
    cqr_scale_floor: float = 1e-3,
    gate_n_features: int | None = None,
    gate_reference_model: Any | None = None,
) -> ConformalTrustModel:
    """Fit surrogate, kNN / Tanimoto-ensemble applicability gate, and Mondrian conformal quantiles."""
    X_tr_2d, _ = _ensure_2d(X_tr)
    X_cal_2d, _ = _ensure_2d(X_cal, expected_dim=X_tr_2d.shape[1])
    y_tr_1d = np.asarray(y_tr, dtype=np.float64)
    y_cal_1d = np.asarray(y_cal, dtype=np.float64)

    if X_tr_2d.shape[0] == 0:
        raise ValueError("Training set X_tr must contain at least 1 sample")
    if X_cal_2d.shape[0] == 0:
        raise ValueError("Calibration set X_cal must contain at least 1 sample")
    if y_tr_1d.shape != (X_tr_2d.shape[0],) or y_cal_1d.shape != (X_cal_2d.shape[0],):
        raise ValueError("Target vector length does not match feature matrix row count")
    if not (np.all(np.isfinite(y_tr_1d)) and np.all(np.isfinite(y_cal_1d))):
        raise ValueError("Target vectors contain NaN or Inf")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if distance_metric not in VALID_DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance_metric {distance_metric!r}; expected one of {VALID_DISTANCE_METRICS}"
        )
    if mondrian_fallback not in ("global_pool", "none", "abstain"):
        raise ValueError(
            f"Unknown mondrian_fallback {mondrian_fallback!r}; expected 'global_pool', 'none', or 'abstain'"
        )
    if cqr_scale_floor <= 0.0 or not math.isfinite(float(cqr_scale_floor)):
        raise ValueError(f"cqr_scale_floor must be finite and > 0, got {cqr_scale_floor}")

    cal_w: np.ndarray | None = None
    if cal_weights is not None:
        cal_w = np.asarray(cal_weights, dtype=np.float64)
        if cal_w.shape != (X_cal_2d.shape[0],) or not np.all(np.isfinite(cal_w)) or np.any(cal_w < 0.0):
            raise ValueError("cal_weights must be a finite non-negative 1D array matching X_cal length")

    model = _fit_point_surrogate(X_tr_2d, y_tr_1d, surrogate=surrogate, loss=loss, seed=seed)

    total_dim = int(X_tr_2d.shape[1])
    is_rdkit_morgan_full = total_dim > _DEFAULT_MORGAN_TAIL_BITS + 32
    if gate_n_features is not None:
        eff_gate_dim = int(gate_n_features)
        if not (1 <= eff_gate_dim <= total_dim):
            raise ValueError(f"gate_n_features={gate_n_features} must lie in [1, {total_dim}]")
    elif distance_metric == "tanimoto_mahalanobis" and is_rdkit_morgan_full:
        _, eff_gate_dim = _infer_molecular_block_layout(total_dim)
    else:
        eff_gate_dim = total_dim

    X_tr_gate = X_tr_2d[:, :eff_gate_dim]
    X_cal_gate = X_cal_2d[:, :eff_gate_dim]

    raw_scaler = StandardScaler().fit(X_tr_gate)
    mean_vec = np.asarray(raw_scaler.mean_, dtype=np.float64)
    raw_std = np.sqrt(np.asarray(raw_scaler.var_, dtype=np.float64))
    raw_const = raw_std <= _ZERO_VAR_TOL
    if distance_metric == "tanimoto_mahalanobis" and is_rdkit_morgan_full:
        # Prevent rare binary MACCS/fragment bits from getting variance-inflated; enforce hard constant
        # invariant mask ONLY on core physical topology columns [0:6] while preserving distance OOD scaling.
        scale_vec = np.where(~raw_const, np.maximum(raw_std, 0.25), _CONST_OOD_SCALE)
        const_mask = np.zeros_like(raw_const, dtype=bool)
        const_mask[: min(6, eff_gate_dim)] = raw_const[: min(6, eff_gate_dim)]
    elif is_rdkit_morgan_full and gate_n_features is None:
        scale_vec = np.where(~raw_const, np.maximum(raw_std, _MIN_NONZERO_STD_FLOOR), _CONST_OOD_SCALE)
        const_mask = np.zeros_like(raw_const, dtype=bool)
        const_mask[: min(6, eff_gate_dim)] = raw_const[: min(6, eff_gate_dim)]
    else:
        const_mask = raw_const
        scale_vec = np.where(
            ~const_mask,
            np.maximum(raw_std, _MIN_NONZERO_STD_FLOOR),
            _CONST_OOD_SCALE,
        )

    imp_model = gate_reference_model if gate_reference_model is not None else model
    f_weights = _compute_feature_weights(X_tr_gate, imp_model, distance_weighting, raw_std=raw_std)

    k_eff = max(1, min(int(k), int(X_tr_2d.shape[0])))
    Z_tr_std = (X_tr_gate - mean_vec) / scale_vec
    Z_cal_std = (X_cal_gate - mean_vec) / scale_vec

    whitening_mat: np.ndarray | None = None
    ens_models: list[Any] | None = None
    tanimoto_ref_fp: np.ndarray | None = None
    knn_norm_scale = 1.0
    spread_norm_scale = 1.0
    tanimoto_norm_scale = 1.0

    if distance_metric in ("mahalanobis", "ensemble_mahalanobis"):
        whitening_mat = _compute_mahalanobis_whitener(Z_tr_std, const_mask)
        X_tr_trans = Z_tr_std @ whitening_mat
        X_cal_trans = Z_cal_std @ whitening_mat
    elif distance_metric == "tanimoto_mahalanobis":
        X_tr_trans = np.clip(Z_tr_std, -8.0, 8.0) * f_weights
        X_cal_trans = np.clip(Z_cal_std, -8.0, 8.0) * f_weights
    else:
        X_tr_trans = Z_tr_std * f_weights
        X_cal_trans = Z_cal_std * f_weights

    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_tr_trans)
    d_cal_knn = np.asarray(nn.kneighbors(X_cal_trans)[0].mean(axis=1), dtype=np.float64)

    if distance_metric in ("ensemble_mahalanobis", "tanimoto_mahalanobis"):
        n_tr = int(X_tr_2d.shape[0])
        if n_tr > 1:
            k_loo = min(k_eff + 1, n_tr)
            d_tr_loo = np.asarray(nn.kneighbors(X_tr_trans, n_neighbors=k_loo)[0][:, 1:].mean(axis=1))
            pos_knn = d_tr_loo[d_tr_loo > _ZERO_VAR_TOL]
            knn_norm_scale = float(np.median(pos_knn)) if pos_knn.size > 0 else 1.0

        if hasattr(model, "epistemic_spread"):
            tr_spread = np.asarray(model.epistemic_spread(X_tr_2d), dtype=np.float64)
            cal_spread = np.asarray(model.epistemic_spread(X_cal_2d), dtype=np.float64)
        else:
            rng = np.random.default_rng(seed + 104729)
            sub_n = max(2, int(round(0.80 * n_tr)))
            ens_models = []
            for m_i in range(4):
                if n_tr >= 5 and sub_n < n_tr:
                    sub_idx = rng.choice(n_tr, size=sub_n, replace=False)
                else:
                    sub_idx = np.arange(n_tr)
                ens_m = HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=100,
                    min_samples_leaf=max(1, min(10, len(sub_idx) // 4)),
                    random_state=seed * 31 + m_i,
                ).fit(X_tr_2d[sub_idx], y_tr_1d[sub_idx])
                ens_models.append(ens_m)
            tr_ens_preds = np.vstack([np.asarray(m.predict(X_tr_2d), dtype=np.float64) for m in ens_models])
            cal_ens_preds = np.vstack([np.asarray(m.predict(X_cal_2d), dtype=np.float64) for m in ens_models])
            tr_spread = np.std(tr_ens_preds, axis=0)
            cal_spread = np.std(cal_ens_preds, axis=0)

        pos_sp = tr_spread[tr_spread > _ZERO_VAR_TOL]
        spread_norm_scale = float(np.median(pos_sp)) if pos_sp.size > 0 else 1.0

        if distance_metric == "ensemble_mahalanobis":
            d_cal = d_cal_knn / max(knn_norm_scale, 1e-8) + cal_spread / max(spread_norm_scale, 1e-8)
        else:
            kd_part = d_cal_knn / max(knn_norm_scale, 1e-8)
            sp_part = cal_spread / max(spread_norm_scale, 1e-8)
            if is_rdkit_morgan_full and total_dim > eff_gate_dim:
                tanimoto_ref_fp = np.array(X_tr_2d[:, eff_gate_dim:], dtype=np.float64, copy=True)
                if n_tr > 1:
                    kt_tr_loo = _continuous_tanimoto_knn(
                        tanimoto_ref_fp,
                        tanimoto_ref_fp,
                        k_eff=k_eff,
                        exclude_self=True,
                    )
                    pos_kt = kt_tr_loo[kt_tr_loo > _ZERO_VAR_TOL]
                    tanimoto_norm_scale = float(np.median(pos_kt)) if pos_kt.size > 0 else 1.0
                kt_cal = _continuous_tanimoto_knn(
                    X_cal_2d[:, eff_gate_dim:],
                    tanimoto_ref_fp,
                    k_eff=k_eff,
                    exclude_self=False,
                )
                kt_part = kt_cal / max(tanimoto_norm_scale, 1e-8)
                d_cal = 0.12 * kd_part + 0.23 * kt_part + 0.65 * sp_part
            else:
                d_cal = 0.50 * kd_part + 0.50 * sp_part
    else:
        d_cal = d_cal_knn

    cal_const_viol = (
        np.any(np.abs(X_cal_gate[:, const_mask] - mean_vec[const_mask]) > _CONST_INVARIANT_TOL, axis=1)
        if np.any(const_mask)
        else np.zeros(X_cal_2d.shape[0], dtype=bool)
    )
    threshold = conformal_threshold(d_cal, trust_pct=trust_pct, weights=cal_w, test_weight=test_weight)
    if np.any(cal_const_viol) and np.any(~cal_const_viol) and float(trust_pct) < 100.0:
        max_in_subspace = float(np.max(d_cal[~cal_const_viol]))
        threshold = min(threshold, max_in_subspace)

    lower_model = None
    upper_model = None
    pred_cal = np.asarray(model.predict(X_cal_2d), dtype=np.float64)

    if nonconformity == "absolute":
        scores = np.abs(y_cal_1d - pred_cal)
    elif nonconformity == "normalized":
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0 for normalized residuals, got {gamma}")
        scores = np.abs(y_cal_1d - pred_cal) / (1.0 + float(gamma) * d_cal)
    elif nonconformity in ("cqr", "cqr_r"):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
        lo_alpha = float(alpha) / 2.0
        hi_alpha = 1.0 - float(alpha) / 2.0
        lower_model, upper_model = _fit_quantile_surrogates(
            X_tr_2d,
            y_tr_1d,
            surrogate=surrogate,
            lo_alpha=lo_alpha,
            hi_alpha=hi_alpha,
            seed=seed,
            point_model=model,
            cqr_scale_floor=cqr_scale_floor,
        )
        q_lo_cal = np.asarray(lower_model.predict(X_cal_2d), dtype=np.float64)
        q_hi_cal = np.asarray(upper_model.predict(X_cal_2d), dtype=np.float64)
        lo_c = np.minimum(q_lo_cal, q_hi_cal)
        hi_c = np.maximum(q_lo_cal, q_hi_cal)
        raw_cqr = np.maximum(lo_c - y_cal_1d, y_cal_1d - hi_c)
        if nonconformity == "cqr_r":
            spread_c = np.maximum(hi_c - lo_c, float(cqr_scale_floor))
            scores = raw_cqr / spread_c
        else:
            scores = raw_cqr
    else:
        raise ValueError(
            f"Unknown nonconformity {nonconformity!r}; expected 'absolute', 'normalized', 'cqr', or 'cqr_r'"
        )

    def _quant(sub_scores: np.ndarray, sub_w: np.ndarray | None) -> float:
        if sub_w is None:
            return conformal_quantile(sub_scores, alpha=alpha)
        return weighted_conformal_quantile(sub_scores, sub_w, alpha=alpha, test_weight=test_weight)

    q_marginal = _quant(scores, cal_w)

    in_mask = (d_cal <= threshold) & (~cal_const_viol)
    out_mask = ~in_mask

    mondrian_fallback_used = False
    n_in = int(np.sum(in_mask))
    if n_in > 0:
        with warnings.catch_warnings(record=True) as caught_in:
            warnings.simplefilter("always", UserWarning)
            q_in_raw = _quant(scores[in_mask], cal_w[in_mask] if cal_w is not None else None)
        if math.isfinite(q_in_raw):
            q_in = q_in_raw
        elif math.isfinite(q_marginal):
            if mondrian_fallback == "global_pool":
                mondrian_fallback_used = True
                q_in = float(max(q_marginal, float(np.max(scores[in_mask]))))
                warnings.warn(
                    f"Mondrian inside-domain bin starvation (n_in={n_in} insufficient for finite "
                    f"(1 - alpha)={1.0 - float(alpha):.3f} quantile while global n_cal={len(scores)} "
                    "is certifiable); falling back to conservative hierarchical global pool.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                q_in = float("inf")
                warnings.warn(
                    f"Mondrian inside-domain bin starvation (n_in={n_in} insufficient for finite "
                    f"(1 - alpha)={1.0 - float(alpha):.3f} quantile); setting q_in = inf.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            for w_rec in caught_in:
                warnings.warn(w_rec.message, UserWarning, stacklevel=2)
            q_in = float("inf")
    elif math.isfinite(q_marginal) and mondrian_fallback == "global_pool":
        mondrian_fallback_used = True
        q_in = q_marginal
        warnings.warn(
            "No calibration samples fell inside threshold; falling back to global marginal quantile.",
            UserWarning,
            stacklevel=2,
        )
    else:
        warnings.warn("No calibration samples fell inside threshold; setting q_in = inf.", UserWarning)
        q_in = float("inf")

    if int(np.sum(out_mask)) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            q_out_cand = _quant(scores[out_mask], cal_w[out_mask] if cal_w is not None else None)
        q_out = q_out_cand if math.isfinite(q_out_cand) else q_marginal
    else:
        q_out = q_marginal

    names = (
        list(feature_names)
        if feature_names is not None
        else [f"x{j}" for j in range(total_dim)]
    )
    stored_gate_n = eff_gate_dim if eff_gate_dim != total_dim else None
    return ConformalTrustModel(
        model=model,
        q_marginal=q_marginal,
        q_in=q_in,
        q_out=q_out,
        threshold=threshold,
        scaler_mean=mean_vec,
        scaler_scale=scale_vec,
        feature_weights=f_weights,
        nn=nn,
        feature_names=names,
        alpha=float(alpha),
        trust_pct=float(trust_pct),
        k_eff=k_eff,
        nonconformity=nonconformity,
        gamma=float(gamma),
        domain_bounds=domain_bounds,
        lower_model=lower_model,
        upper_model=upper_model,
        cqr_scale_floor=float(cqr_scale_floor),
        constant_feature_mask=const_mask,
        mondrian_fallback_used=mondrian_fallback_used,
        distance_metric=distance_metric,
        whitening_matrix=whitening_mat,
        ensemble_models=ens_models,
        knn_norm_scale=knn_norm_scale,
        spread_norm_scale=spread_norm_scale,
        tanimoto_ref_fp=tanimoto_ref_fp,
        tanimoto_norm_scale=tanimoto_norm_scale,
        n_input_features=total_dim,
        gate_n_features=stored_gate_n,
    )


def summarize(
    y: np.ndarray,
    lo_or_pred: np.ndarray,
    hi_or_q: np.ndarray | float,
    mask: np.ndarray,
    *,
    pred: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute `(coverage, mean_width, mae)` over `mask` using per-sample `(lo, hi)` bounds."""
    y_arr = np.asarray(y, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if int(np.sum(m)) == 0:
        return float("nan"), float("nan"), float("nan")

    if np.ndim(hi_or_q) == 0:
        p_arr = np.asarray(lo_or_pred, dtype=np.float64)
        q_val = float(hi_or_q)
        lo_arr = p_arr - q_val
        hi_arr = p_arr + q_val
        point = p_arr if pred is None else np.asarray(pred, dtype=np.float64)
    else:
        lo_arr = np.asarray(lo_or_pred, dtype=np.float64)
        hi_arr = np.asarray(hi_or_q, dtype=np.float64)
        point = (
            0.5 * (lo_arr + hi_arr)
            if pred is None
            else np.asarray(pred, dtype=np.float64)
        )

    ym = y_arr[m]
    lom = lo_arr[m]
    him = hi_arr[m]
    pm = point[m]

    covered = (ym >= lom) & (ym <= him)
    width = him - lom
    mae = np.abs(ym - pm)
    return float(np.mean(covered)), float(np.mean(width)), float(np.mean(mae))
