"""Finite-sample Mondrian & adaptive conformal calibration and kNN applicability-domain gating."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float = 0.10) -> float:
    """Exact finite-sample split conformal quantile `ceil((n + 1) * (1 - alpha)) / n`.

    Never clamps `level` to `1.0`: when `n < (1 - alpha) / alpha`, achieving `1 - alpha`
    distribution-free coverage from `n` calibration samples is mathematically impossible,
    so a `UserWarning` is emitted and `math.inf` is returned.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Calibration scores must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Calibration scores contain NaN or Inf")

    n = int(arr.size)
    k_ord = math.ceil((n + 1) * (1.0 - float(alpha)))
    raw_level = k_ord / float(n)
    if k_ord > n:
        warnings.warn(
            f"Calibration size n={n} is too small for finite (1 - alpha)={1.0 - alpha:.3f} "
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
    alpha: float = 0.10,
    *,
    test_weight: float = 1.0,
) -> float:
    """Importance-weighted conformal quantile under covariate shift (Tibshirani et al., 2019)."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.size == 0 or s.shape != w.shape:
        raise ValueError("scores and weights must be non-empty 1D arrays of equal length")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)) or test_weight <= 0.0:
        raise ValueError("weights must be finite non-negative and test_weight > 0")

    total = float(np.sum(w)) + float(test_weight)
    if total <= 0.0:
        return float("inf")
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    p_sorted = w[order] / total
    cum_p = np.cumsum(p_sorted)
    target = 1.0 - float(alpha)
    idx = int(np.searchsorted(cum_p, target, side="left"))
    if idx >= s_sorted.size:
        warnings.warn(
            "Total calibration weight insufficient for weighted conformal level; returning inf.",
            UserWarning,
            stacklevel=2,
        )
        return float("inf")
    return float(s_sorted[idx])


def conformal_threshold(cal_distances: Sequence[float] | np.ndarray, trust_pct: float = 80.0) -> float:
    """Finite-sample conformal outlier cutoff `ceil((n + 1) * (trust_pct / 100)) / n`."""
    if not (0.0 < float(trust_pct) <= 100.0):
        raise ValueError(f"trust_pct must lie in (0, 100], got {trust_pct}")
    d = np.asarray(cal_distances, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("Calibration distances must be a non-empty 1D sequence")
    n = int(d.size)
    k_ord = math.ceil((n + 1) * (float(trust_pct) / 100.0))
    if k_ord > n:
        return float("inf")
    return float(np.partition(d, k_ord - 1)[k_ord - 1])


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

    def transform_features(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        arr2d, _ = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
        standardized = (arr2d - self.scaler_mean) / self.scaler_scale
        return standardized * self.feature_weights

    def distance(self, X: np.ndarray | Sequence[float]) -> np.ndarray:
        """Compute importance-weighted standardized kNN distance (supports 1D or 2D inputs)."""
        arr2d, _ = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
        if arr2d.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        d, _ = self.nn.kneighbors(self.transform_features(arr2d))
        return np.asarray(d.mean(axis=1), dtype=np.float64)

    def _raw_intervals(
        self,
        arr2d: np.ndarray,
        dists: np.ndarray,
        *,
        use_mondrian: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = np.asarray(self.model.predict(arr2d), dtype=np.float64)
        n = arr2d.shape[0]
        if use_mondrian:
            inside = dists <= self.threshold
            q_vec = np.where(inside, self.q_in, self.q_out)
        else:
            q_vec = np.full(n, self.q_marginal, dtype=np.float64)

        if self.nonconformity == "absolute":
            half = q_vec
            lo = pred - half
            hi = pred + half
        elif self.nonconformity == "normalized":
            scale = 1.0 + self.gamma * dists
            half = q_vec * scale
            lo = pred - half
            hi = pred + half
        elif self.nonconformity == "cqr":
            if self.lower_model is None or self.upper_model is None:
                raise RuntimeError("CQR models are missing on ConformalTrustModel")
            q_lo = np.asarray(self.lower_model.predict(arr2d), dtype=np.float64)
            q_hi = np.asarray(self.upper_model.predict(arr2d), dtype=np.float64)
            lo_raw = np.minimum(q_lo, q_hi)
            hi_raw = np.maximum(q_lo, q_hi)
            lo = lo_raw - q_vec
            hi = hi_raw + q_vec
        else:
            raise ValueError(f"Unsupported nonconformity mode: {self.nonconformity!r}")

        # Clip ONLY where intervals are finite; infinite intervals indicate uncertifiable small-n calibration
        if self.domain_bounds is not None:
            y_min, y_max = float(self.domain_bounds[0]), float(self.domain_bounds[1])
            finite_mask = np.isfinite(lo) & np.isfinite(hi)
            lo = np.where(finite_mask, np.clip(lo, y_min, y_max), lo)
            hi = np.where(finite_mask, np.clip(hi, y_min, y_max), hi)
            pred = np.clip(pred, y_min, y_max)

        return pred, lo, hi

    def predict(
        self,
        X: np.ndarray | Sequence[float],
        *,
        use_mondrian: bool = True,
    ) -> dict[str, Any]:
        """Predict point estimates, conformal intervals, distances, and trust decisions on 1D or 2D X."""
        arr2d, is_1d = _ensure_2d(X, expected_dim=int(self.scaler_mean.size))
        if arr2d.shape[0] == 0:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "pred": empty,
                "lo": empty,
                "hi": empty,
                "width": empty,
                "distance": empty,
                "trusted": np.zeros(0, dtype=bool),
                "decision": [],
            }

        dists = self.distance(arr2d)
        pred, lo, hi = self._raw_intervals(arr2d, dists, use_mondrian=use_mondrian)
        finite_iv = np.isfinite(lo) & np.isfinite(hi)
        trusted = (dists <= self.threshold) & finite_iv
        width = hi - lo
        decisions = ["ANSWER" if bool(t) else "ABSTAIN_ACQUIRE" for t in trusted]

        if is_1d:
            return {
                "pred": float(pred[0]),
                "lo": float(lo[0]),
                "hi": float(hi[0]),
                "width": float(width[0]),
                "distance": float(dists[0]),
                "trusted": bool(trusted[0]),
                "decision": decisions[0],
            }
        return {
            "pred": pred,
            "lo": lo,
            "hi": hi,
            "width": width,
            "distance": dists,
            "trusted": trusted,
            "decision": decisions,
        }


def _compute_feature_weights(
    X_tr: np.ndarray,
    model: GradientBoostingRegressor,
    mode: str,
) -> np.ndarray:
    d = X_tr.shape[1]
    if mode == "uniform":
        return np.ones(d, dtype=np.float64)
    if mode == "importance":
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and float(np.sum(imp)) > 1e-12:
            w = np.sqrt(np.maximum(np.asarray(imp, dtype=np.float64), 0.0))
            norm = math.sqrt(float(np.mean(w ** 2)))
            return w / norm if norm > 1e-12 else np.ones(d, dtype=np.float64)
        return np.ones(d, dtype=np.float64)
    if mode == "iqr":
        q75 = np.percentile(X_tr, 75, axis=0)
        q25 = np.percentile(X_tr, 25, axis=0)
        iqr = q75 - q25
        std = np.std(X_tr, axis=0)
        # Downweight near-degenerate discrete columns whose IQR is 0 while std > 0
        ratio = np.where(std > 1e-12, np.where(iqr > 1e-12, 1.0, 0.20), 0.0)
        return ratio
    raise ValueError(f"Unknown distance_weighting {mode!r}; expected 'importance', 'uniform', or 'iqr'")


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
    gamma: float = 0.50,
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75),
    feature_names: Sequence[str] | None = None,
) -> ConformalTrustModel:
    """Fit surrogate, importance-weighted kNN applicability gate, and Mondrian conformal quantiles."""
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

    # Effective loss: scikit-learn Huber requires >= 2 distinct y values; guard tiny constant fixtures
    eff_loss = loss
    if eff_loss == "huber" and (X_tr_2d.shape[0] < 3 or np.ptp(y_tr_1d) < 1e-12):
        eff_loss = "squared_error"

    model = GradientBoostingRegressor(loss=eff_loss, random_state=seed).fit(X_tr_2d, y_tr_1d)

    # Strict train-only feature standardization with explicit zero-variance guard
    raw_scaler = StandardScaler().fit(X_tr_2d)
    mean_vec = np.asarray(raw_scaler.mean_, dtype=np.float64)
    raw_std = np.sqrt(np.asarray(raw_scaler.var_, dtype=np.float64))
    scale_vec = np.where(raw_std > 1e-12, raw_std, 1.0)
    f_weights = _compute_feature_weights(X_tr_2d, model, distance_weighting)

    k_eff = max(1, min(int(k), int(X_tr_2d.shape[0])))
    X_tr_trans = ((X_tr_2d - mean_vec) / scale_vec) * f_weights
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_tr_trans)

    X_cal_trans = ((X_cal_2d - mean_vec) / scale_vec) * f_weights
    d_cal = np.asarray(nn.kneighbors(X_cal_trans)[0].mean(axis=1), dtype=np.float64)
    threshold = conformal_threshold(d_cal, trust_pct=trust_pct)

    lower_model = None
    upper_model = None
    pred_cal = np.asarray(model.predict(X_cal_2d), dtype=np.float64)

    if nonconformity == "absolute":
        scores = np.abs(y_cal_1d - pred_cal)
    elif nonconformity == "normalized":
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0 for normalized residuals, got {gamma}")
        scores = np.abs(y_cal_1d - pred_cal) / (1.0 + float(gamma) * d_cal)
    elif nonconformity == "cqr":
        lo_alpha = max(0.01, float(alpha) / 2.0)
        hi_alpha = min(0.99, 1.0 - float(alpha) / 2.0)
        lower_model = GradientBoostingRegressor(
            loss="quantile", alpha=lo_alpha, random_state=seed
        ).fit(X_tr_2d, y_tr_1d)
        upper_model = GradientBoostingRegressor(
            loss="quantile", alpha=hi_alpha, random_state=seed
        ).fit(X_tr_2d, y_tr_1d)
        q_lo_cal = np.asarray(lower_model.predict(X_cal_2d), dtype=np.float64)
        q_hi_cal = np.asarray(upper_model.predict(X_cal_2d), dtype=np.float64)
        lo_c = np.minimum(q_lo_cal, q_hi_cal)
        hi_c = np.maximum(q_lo_cal, q_hi_cal)
        scores = np.maximum(lo_c - y_cal_1d, y_cal_1d - hi_c)
    else:
        raise ValueError(
            f"Unknown nonconformity {nonconformity!r}; expected 'absolute', 'normalized', or 'cqr'"
        )

    q_marginal = conformal_quantile(scores, alpha=alpha)

    in_mask = d_cal <= threshold
    out_mask = ~in_mask

    if int(np.sum(in_mask)) > 0:
        q_in = conformal_quantile(scores[in_mask], alpha=alpha)
    else:
        warnings.warn("No calibration samples fell inside threshold; setting q_in = inf.", UserWarning)
        q_in = float("inf")

    if int(np.sum(out_mask)) > 0:
        # Outside points are abstained on by the trust gate; calibrate q_out for diagnostic reporting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            q_out_cand = conformal_quantile(scores[out_mask], alpha=alpha)
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
    )


def summarize(
    y: np.ndarray,
    lo_or_pred: np.ndarray,
    hi_or_q: np.ndarray | float,
    mask: np.ndarray,
    *,
    pred: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute `(coverage, mean_width, mae)` over `mask` using per-sample `(lo, hi)` bounds.

    Also accepts legacy scalar half-width `(y, pred, q, mask)` seamlessly.
    """
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
