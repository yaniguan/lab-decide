"""Finite-sample Mondrian & adaptive conformal calibration and kNN applicability-domain gating."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_ZERO_VAR_TOL = 1e-12
_MIN_NONZERO_STD_FLOOR = 1e-3
_CONST_INVARIANT_TOL = 1e-6
_CONST_OOD_SCALE = 0.10
_MIN_ACTIVE_FEATURE_WEIGHT = 0.05
_MAHALANOBIS_RIDGE = 0.10


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
        mean_v = np.asarray(scaler_mean, dtype=np.float64)
        scale_v = np.where(np.asarray(scaler_scale, dtype=np.float64) > _ZERO_VAR_TOL, scaler_scale, 1.0)
    else:
        mean_v = np.mean(X_ref_2d, axis=0)
        raw_s = np.std(X_ref_2d, axis=0)
        scale_v = np.where(raw_s > _ZERO_VAR_TOL, np.maximum(raw_s, _MIN_NONZERO_STD_FLOOR), 1.0)

    fw = np.ones(X_ref_2d.shape[1], dtype=np.float64) if feature_weights is None else np.asarray(feature_weights, dtype=np.float64)
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


@dataclass
class ConformalTrustModel:
    """Picklable conformal regression & kNN applicability-domain advisor."""

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

    def _constant_invariant_violated(self, arr2d: np.ndarray) -> np.ndarray:
        if self.constant_feature_mask is None or not np.any(self.constant_feature_mask):
            return np.zeros(arr2d.shape[0], dtype=bool)
        cmask = np.asarray(self.constant_feature_mask, dtype=bool)
        diffs = np.abs(arr2d[:, cmask] - self.scaler_mean[cmask])
        return np.any(diffs > _CONST_INVARIANT_TOL, axis=1)

    def transform_features(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        arr2d, _ = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
        standardized = (arr2d - self.scaler_mean) / self.scaler_scale
        if self.whitening_matrix is not None:
            return standardized @ self.whitening_matrix
        return standardized * self.feature_weights

    def distance(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        """Compute applicability-domain distance (Euclidean, Mahalanobis, or ensemble-Mahalanobis)."""
        arr2d, _ = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
        if arr2d.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        d_knn, _ = self.nn.kneighbors(self.transform_features(arr2d))
        knn_mean = np.asarray(d_knn.mean(axis=1), dtype=np.float64)
        if self.distance_metric == "ensemble_mahalanobis" and self.ensemble_models:
            ens_preds = np.vstack(
                [np.asarray(m.predict(arr2d), dtype=np.float64) for m in self.ensemble_models]
            )
            spread = np.std(ens_preds, axis=0)
            return knn_mean / max(float(self.knn_norm_scale), 1e-8) + spread / max(
                float(self.spread_norm_scale), 1e-8
            )
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
        arr2d, is_1d = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
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
        if imp is not None and float(np.sum(imp)) > _ZERO_VAR_TOL:
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
    raise ValueError(f"Unknown surrogate {surrogate!r}; expected 'gbdt', 'hist_gbdt', or 'lightgbm'")


def _fit_quantile_surrogates(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    surrogate: str,
    lo_alpha: float,
    hi_alpha: float,
    seed: int,
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
) -> ConformalTrustModel:
    """Fit surrogate, kNN applicability gate, and Mondrian conformal quantiles."""
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
    if distance_metric not in ("euclidean", "mahalanobis", "ensemble_mahalanobis"):
        raise ValueError(
            f"Unknown distance_metric {distance_metric!r}; expected 'euclidean', 'mahalanobis', or 'ensemble_mahalanobis'"
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

    raw_scaler = StandardScaler().fit(X_tr_2d)
    mean_vec = np.asarray(raw_scaler.mean_, dtype=np.float64)
    raw_std = np.sqrt(np.asarray(raw_scaler.var_, dtype=np.float64))
    const_mask = raw_std <= _ZERO_VAR_TOL
    scale_vec = np.where(
        ~const_mask,
        np.maximum(raw_std, _MIN_NONZERO_STD_FLOOR),
        _CONST_OOD_SCALE,
    )
    f_weights = _compute_feature_weights(X_tr_2d, model, distance_weighting, raw_std=raw_std)

    k_eff = max(1, min(int(k), int(X_tr_2d.shape[0])))
    Z_tr_std = (X_tr_2d - mean_vec) / scale_vec
    Z_cal_std = (X_cal_2d - mean_vec) / scale_vec

    whitening_mat: np.ndarray | None = None
    ens_models: list[Any] | None = None
    knn_norm_scale = 1.0
    spread_norm_scale = 1.0

    if distance_metric in ("mahalanobis", "ensemble_mahalanobis"):
        whitening_mat = _compute_mahalanobis_whitener(Z_tr_std, const_mask)
        X_tr_trans = Z_tr_std @ whitening_mat
        X_cal_trans = Z_cal_std @ whitening_mat
    else:
        X_tr_trans = Z_tr_std * f_weights
        X_cal_trans = Z_cal_std * f_weights

    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_tr_trans)
    d_cal_knn = np.asarray(nn.kneighbors(X_cal_trans)[0].mean(axis=1), dtype=np.float64)

    if distance_metric == "ensemble_mahalanobis":
        rng = np.random.default_rng(seed + 104729)
        n_tr = int(X_tr_2d.shape[0])
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

        # Compute normalization scales strictly on X_tr (never touching X_cal -> exact exchangeability!)
        if n_tr > 1:
            k_loo = min(k_eff + 1, n_tr)
            d_tr_loo = np.asarray(nn.kneighbors(X_tr_trans, n_neighbors=k_loo)[0][:, 1:].mean(axis=1))
            pos_knn = d_tr_loo[d_tr_loo > _ZERO_VAR_TOL]
            knn_norm_scale = float(np.median(pos_knn)) if pos_knn.size > 0 else 1.0
        tr_ens_preds = np.vstack([np.asarray(m.predict(X_tr_2d), dtype=np.float64) for m in ens_models])
        tr_spread = np.std(tr_ens_preds, axis=0)
        pos_sp = tr_spread[tr_spread > _ZERO_VAR_TOL]
        spread_norm_scale = float(np.median(pos_sp)) if pos_sp.size > 0 else 1.0

        cal_ens_preds = np.vstack([np.asarray(m.predict(X_cal_2d), dtype=np.float64) for m in ens_models])
        cal_spread = np.std(cal_ens_preds, axis=0)
        d_cal = d_cal_knn / max(knn_norm_scale, 1e-8) + cal_spread / max(spread_norm_scale, 1e-8)
    else:
        d_cal = d_cal_knn

    cal_const_viol = (
        np.any(np.abs(X_cal_2d[:, const_mask] - mean_vec[const_mask]) > _CONST_INVARIANT_TOL, axis=1)
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
        else [f"x{j}" for j in range(X_tr_2d.shape[1])]
    )
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
