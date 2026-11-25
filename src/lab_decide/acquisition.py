"""Diverse boundary batch acquisition, pool-separated expansion, and trust-region BO."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from lab_decide.conformal import (
    ConformalTrustModel,
    compute_covariate_shift_weights,
    fit_and_calibrate,
    summarize,
)

_STD_NORM = NormalDist(0.0, 1.0)


def select_diverse_boundary_batch(
    advisor: ConformalTrustModel,
    X_train: np.ndarray,
    X_pool: np.ndarray,
    batch_size: int,
    *,
    shell_mult: float = 2.2,
) -> np.ndarray:
    """Select up to `batch_size` outside-region pool indices via sequential maximin kNN diversity.

    First restricts to outside pool points (`d_pool > threshold`) within the boundary shell
    `d_pool <= shell_mult * threshold` (expanding if fewer than `batch_size` points lie in shell).
    Then picks the closest boundary seed first and greedily maximizes minimum feature-space
    distance to `X_train U selected_so_far` so parallel wet-lab plates do not cluster on identical
    analogues.
    """
    if batch_size <= 0 or len(X_pool) == 0:
        return np.zeros(0, dtype=int)

    d_pool = advisor.distance(X_pool)
    outside = np.where(d_pool > advisor.threshold)[0]
    if outside.size == 0:
        return np.zeros(0, dtype=int)
    if outside.size <= batch_size:
        return outside[np.argsort(d_pool[outside], kind="stable")]

    shell = outside[d_pool[outside] <= shell_mult * advisor.threshold]
    if shell.size < batch_size:
        # Take at least 3 * batch_size nearest outside candidates as the diverse frontier
        frontier_size = min(outside.size, max(batch_size * 3, batch_size))
        frontier = outside[np.argsort(d_pool[outside], kind="stable")[:frontier_size]]
    else:
        frontier = shell

    Z_train = advisor.transform_features(X_train)
    Z_front = advisor.transform_features(X_pool[frontier])

    # Minimum distance from each frontier point to existing training points
    diff_tr = Z_front[:, None, :] - Z_train[None, :, :]
    min_dist_sq = np.min(np.sum(diff_tr ** 2, axis=2), axis=1)

    selected_local: list[int] = []
    # Seed batch with the single nearest boundary point so expansion stays anchored to the manifold
    first_local = int(np.argmin(d_pool[frontier]))
    selected_local.append(first_local)

    chosen_mask = np.zeros(frontier.size, dtype=bool)
    chosen_mask[first_local] = True

    while len(selected_local) < min(batch_size, frontier.size):
        last_vec = Z_front[selected_local[-1]]
        dist_last_sq = np.sum((Z_front - last_vec[None, :]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_last_sq)
        scores = np.where(chosen_mask, -1.0, min_dist_sq)
        nxt = int(np.argmax(scores))
        chosen_mask[nxt] = True
        selected_local.append(nxt)

    return np.asarray([int(frontier[i]) for i in selected_local], dtype=int)


def run_expansion_campaign(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    pool_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    rounds: int = 3,
    per_round: int = 20,
    cal_update_frac: float = 0.20,
    alpha: float = 0.10,
    k: int = 5,
    trust_pct: float = 80.0,
    seed: int = 0,
    loss: str = "huber",
    nonconformity: str = "normalized",
    distance_weighting: str = "importance",
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75),
    feature_names: Sequence[str] | None = None,
    reweight_covariate_shift: bool = True,
) -> tuple[list[dict[str, Any]], ConformalTrustModel]:
    """Run active trust-region expansion acquiring strictly from `pool_idx` while holding `test_idx` frozen.

    When `reweight_covariate_shift=True` (default), outer-shell calibration additions are
    reweighted via Tibshirani-Foygel-Barber likelihood-ratio importance weights relative to
    the initial i.i.d. reference calibration split `cal_idx`.
    """
    cur_tr = np.array(train_idx, dtype=int, copy=True)
    cur_pool = np.array(pool_idx, dtype=int, copy=True)
    ref_cal = np.array(cal_idx, dtype=int, copy=True)
    act_cal = np.zeros(0, dtype=int)
    cur_cal = np.array(cal_idx, dtype=int, copy=True)
    frozen_te = np.array(test_idx, dtype=int, copy=True)

    history: list[dict[str, Any]] = []
    advisor = fit_and_calibrate(
        X[cur_tr],
        y[cur_tr],
        X[cur_cal],
        y[cur_cal],
        alpha=alpha,
        k=k,
        trust_pct=trust_pct,
        seed=seed,
        loss=loss,
        nonconformity=nonconformity,
        distance_weighting=distance_weighting,
        domain_bounds=domain_bounds,
        feature_names=feature_names,
    )

    for r in range(rounds + 1):
        if r > 0:
            take_pos = select_diverse_boundary_batch(
                advisor,
                X[cur_tr],
                X[cur_pool],
                batch_size=per_round,
            )
            if take_pos.size == 0:
                # Stop early without redundant refit once pool has zero outside candidates left
                break

            acquired_global = cur_pool[take_pos]
            cur_pool = np.delete(cur_pool, take_pos)

            # Route a deterministic stratified slice of newly acquired points to update calibration
            n_to_cal = int(round(acquired_global.size * cal_update_frac))
            if n_to_cal > 0 and acquired_global.size >= 5:
                step = max(1, acquired_global.size // n_to_cal)
                cal_slice_mask = np.zeros(acquired_global.size, dtype=bool)
                cal_slice_mask[::step][:n_to_cal] = True
                act_cal = np.concatenate([act_cal, acquired_global[cal_slice_mask]])
                cur_cal = np.concatenate([ref_cal, act_cal])
                cur_tr = np.concatenate([cur_tr, acquired_global[~cal_slice_mask]])
            else:
                cur_tr = np.concatenate([cur_tr, acquired_global])

            cal_w: np.ndarray | None = None
            if reweight_covariate_shift and act_cal.size > 0:
                cal_w = compute_covariate_shift_weights(
                    X[ref_cal],
                    X[act_cal],
                    scaler_mean=advisor.scaler_mean,
                    scaler_scale=advisor.scaler_scale,
                    feature_weights=advisor.feature_weights,
                )

            advisor = fit_and_calibrate(
                X[cur_tr],
                y[cur_tr],
                X[cur_cal],
                y[cur_cal],
                alpha=alpha,
                k=k,
                trust_pct=trust_pct,
                seed=seed,
                loss=loss,
                nonconformity=nonconformity,
                distance_weighting=distance_weighting,
                domain_bounds=domain_bounds,
                feature_names=feature_names,
                cal_weights=cal_w,
            )

        res = advisor.predict(X[frozen_te], use_mondrian=True)
        yt = y[frozen_te]
        inside = np.asarray(res["trusted"], dtype=bool)
        outside = ~inside

        c_in, w_in, e_in = summarize(yt, res["lo"], res["hi"], inside, pred=res["pred"])
        c_out, w_out, e_out = summarize(yt, res["lo"], res["hi"], outside, pred=res["pred"])
        c_all, w_all, e_all = summarize(
            yt, res["lo"], res["hi"], np.ones(len(yt), dtype=bool), pred=res["pred"]
        )

        history.append(
            {
                "round": r,
                "n_train": int(cur_tr.size),
                "n_pool": int(cur_pool.size),
                "n_cal": int(cur_cal.size),
                "n_test": int(frozen_te.size),
                "n_in": int(np.sum(inside)),
                "n_out": int(np.sum(outside)),
                "answered": float(np.mean(inside)),
                "cov_in": c_in,
                "cov_out": c_out,
                "cov_all": c_all,
                "mae_in": e_in,
                "mae_out": e_out,
                "mae_all": e_all,
                "width_in": w_in,
                "width_out": w_out,
                "width_all": w_all,
                "threshold": float(advisor.threshold),
                "q_in": float(advisor.q_in),
            }
        )

    return history, advisor


@dataclass(frozen=True)
class AcquisitionProposal:
    """Ranked candidate acquisition recommendation from TrustRegionAdvisor."""

    pool_index: int
    acquisition_score: float
    pred: float
    lo: float
    hi: float
    distance: float
    trusted: bool
    mode_used: str


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.asarray([0.5 * (1.0 + math.erf(float(val) / math.sqrt(2.0))) for val in z], dtype=np.float64)


class TrustRegionAdvisor:
    """Conformal trust-region Bayesian optimization advisor (`conformal_ucb`, `conformal_ei`, `safe_expand`)."""

    def __init__(self, model: ConformalTrustModel) -> None:
        self.model = model

    def propose(
        self,
        X_pool: np.ndarray,
        *,
        mode: str = "conformal_ucb",
        batch_size: int = 5,
        maximize: bool = True,
        best_y: float | None = None,
        beta: float = 1.0,
    ) -> list[AcquisitionProposal]:
        """Rank candidate experiments inside the trust region (exploitation) or on its boundary."""
        if batch_size <= 0 or len(X_pool) == 0:
            return []

        res = self.model.predict(X_pool, use_mondrian=True)
        pred = np.asarray(res["pred"], dtype=np.float64)
        lo = np.asarray(res["lo"], dtype=np.float64)
        hi = np.asarray(res["hi"], dtype=np.float64)
        dists = np.asarray(res["distance"], dtype=np.float64)
        trusted = np.asarray(res["trusted"], dtype=bool)
        half_w = np.maximum(0.5 * (hi - lo), 0.0)
        upper_rad = np.maximum(hi - pred, 0.0)
        lower_rad = np.maximum(pred - lo, 0.0)

        if mode == "safe_expand":
            chosen = select_diverse_boundary_batch(
                self.model,
                # Reconstruct standardized training reference centers via fitted NearestNeighbors
                self._training_approx(),
                X_pool,
                batch_size=batch_size,
            )
            return [
                AcquisitionProposal(
                    pool_index=int(idx),
                    acquisition_score=float(half_w[idx] / (dists[idx] + 1e-6)),
                    pred=float(pred[idx]),
                    lo=float(lo[idx]),
                    hi=float(hi[idx]),
                    distance=float(dists[idx]),
                    trusted=bool(trusted[idx]),
                    mode_used="safe_expand",
                )
                for idx in chosen
            ]

        if mode == "conformal_ucb":
            # Respect asymmetric directional bounds (at beta=1: `hi` when maximize=True, `-lo` when False)
            raw_util = (pred + float(beta) * upper_rad) if maximize else -(pred - float(beta) * lower_rad)
        elif mode == "conformal_ei":
            if best_y is None:
                pool_vals = pred[trusted] if np.any(trusted) else pred
                ref = float(np.max(pool_vals) if maximize else np.min(pool_vals))
            else:
                ref = float(best_y)
            alpha_clamped = min(max(float(self.model.alpha), 1e-6), 1.0 - 1e-6)
            z_div = max(_STD_NORM.inv_cdf(1.0 - 0.5 * alpha_clamped), 1e-6)
            dir_rad = upper_rad if maximize else lower_rad
            eff_rad = np.where(dir_rad > 1e-9, dir_rad, half_w)
            sigma = np.maximum(eff_rad / z_div, 1e-6)
            imp = (pred - ref) if maximize else (ref - pred)
            z = imp / sigma
            raw_util = imp * _norm_cdf(z) + sigma * _norm_pdf(z)
        else:
            raise ValueError(
                f"Unknown acquisition mode {mode!r}; expected 'conformal_ucb', 'conformal_ei', or 'safe_expand'"
            )

        inside_idx = np.where(trusted)[0]
        if inside_idx.size == 0:
            # Safe fallback when zero pool candidates lie inside the calibrated trust region
            return self.propose(X_pool, mode="safe_expand", batch_size=batch_size, maximize=maximize)

        order = inside_idx[np.argsort(-raw_util[inside_idx], kind="stable")[:batch_size]]
        return [
            AcquisitionProposal(
                pool_index=int(idx),
                acquisition_score=float(raw_util[idx]),
                pred=float(pred[idx]),
                lo=float(lo[idx]),
                hi=float(hi[idx]),
                distance=float(dists[idx]),
                trusted=True,
                mode_used=mode,
            )
            for idx in order
        ]

    def _training_approx(self) -> np.ndarray:
        """Recover raw-space training points stored inside `nn._fit_X` for standalone diversity scoring."""
        fit_z = np.asarray(self.model.nn._fit_X, dtype=np.float64)
        safe_w = np.where(self.model.feature_weights > 1e-12, self.model.feature_weights, 1.0)
        return (fit_z / safe_w) * self.model.scaler_scale + self.model.scaler_mean


def benchmark_trust_region_bo(
    advisor: ConformalTrustModel,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    batch_size: int = 10,
    maximize: bool = True,
) -> dict[str, float]:
    """Compare trust-region constrained conformal UCB against unconstrained greedy optimization."""
    tr_adv = TrustRegionAdvisor(advisor)
    constrained = tr_adv.propose(X_pool, mode="conformal_ucb", batch_size=batch_size, maximize=maximize)
    c_idx = np.asarray([p.pool_index for p in constrained], dtype=int)

    pred_all = np.asarray(advisor.predict(X_pool, use_mondrian=True)["pred"], dtype=np.float64)
    dists_all = advisor.distance(X_pool)
    unconstrained_order = np.argsort(-pred_all if maximize else pred_all, kind="stable")[:batch_size]

    c_err = float(np.mean(np.abs(y_pool[c_idx] - pred_all[c_idx]))) if c_idx.size else float("nan")
    u_err = float(np.mean(np.abs(y_pool[unconstrained_order] - pred_all[unconstrained_order])))
    u_ood_frac = float(np.mean(dists_all[unconstrained_order] > advisor.threshold))

    c_best = float(np.max(y_pool[c_idx]) if maximize else np.min(y_pool[c_idx])) if c_idx.size else float("nan")
    u_best = float(np.max(y_pool[unconstrained_order]) if maximize else np.min(y_pool[unconstrained_order]))
    return {
        "constrained_top_y": c_best,
        "unconstrained_top_y": u_best,
        "constrained_pred_mae": c_err,
        "unconstrained_pred_mae": u_err,
        "unconstrained_ood_fraction": u_ood_frac,
    }
