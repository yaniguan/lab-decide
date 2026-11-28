"""Diverse boundary batch acquisition, pool-separated expansion, single-batch ranking, and sequential BO."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from lab_decide.conformal import (
    ConformalTrustModel,
    compute_covariate_shift_weights,
    fit_and_calibrate,
    summarize,
)

_STD_NORM = NormalDist(0.0, 1.0)

VALID_CAMPAIGN_METHODS: tuple[str, ...] = (
    "random",
    "greedy",
    "unconstrained_conformal_ucb",
    "tr_conformal_ucb",
    "tr_conformal_ei",
    "gp_ei",
    "lightgbm_cqr_ucb",
    "lightgbm_cqr_tr_ucb",
)


def select_diverse_boundary_batch(
    advisor: ConformalTrustModel,
    X_train: np.ndarray,
    X_pool: np.ndarray,
    batch_size: int,
    *,
    shell_mult: float = 2.2,
) -> np.ndarray:
    """Select up to `batch_size` outside-region pool indices via sequential maximin kNN diversity."""
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
        frontier_size = min(outside.size, max(batch_size * 3, batch_size))
        frontier = outside[np.argsort(d_pool[outside], kind="stable")[:frontier_size]]
    else:
        frontier = shell

    Z_train = advisor.transform_features(X_train)
    Z_front = advisor.transform_features(X_pool[frontier])

    diff_tr = Z_front[:, None, :] - Z_train[None, :, :]
    min_dist_sq = np.min(np.sum(diff_tr ** 2, axis=2), axis=1)

    selected_local: list[int] = []
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
    distance_metric: str = "euclidean",
    surrogate: str = "gbdt",
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75),
    feature_names: Sequence[str] | None = None,
    reweight_covariate_shift: bool = True,
) -> tuple[list[dict[str, Any]], ConformalTrustModel]:
    """Run active trust-region expansion acquiring strictly from `pool_idx` while holding `test_idx` frozen."""
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
        distance_metric=distance_metric,
        surrogate=surrogate,
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
                break

            acquired_global = cur_pool[take_pos]
            cur_pool = np.delete(cur_pool, take_pos)

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
                distance_metric=distance_metric,
                surrogate=surrogate,
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
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(val) / math.sqrt(2.0))) for val in z],
        dtype=np.float64,
    )


def _gaussian_expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    best_y: float,
    maximize: bool = True,
) -> np.ndarray:
    sig_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    imp = (mu - float(best_y)) if maximize else (float(best_y) - mu)
    z = imp / sig_safe
    return imp * _norm_cdf(z) + sig_safe * _norm_pdf(z)


class TrustRegionAdvisor:
    """Conformal trust-region advisor (`conformal_ucb`, `conformal_ei`, `safe_expand`)."""

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
        """Rank candidate experiments inside the trust region (or on its boundary in `safe_expand`).

        When multiple candidates hit physical domain clip bounds (e.g. `hi == 1.75` on ESOL),
        ties are broken deterministically by the unclipped conformal acquisition utility before
        falling back to unclipped directional point prediction and pool index.
        """
        if batch_size <= 0 or len(X_pool) == 0:
            return []

        res = self.model.predict(X_pool, use_mondrian=True, include_unclipped=True)
        pred = np.asarray(res["pred"], dtype=np.float64)
        lo = np.asarray(res["lo"], dtype=np.float64)
        hi = np.asarray(res["hi"], dtype=np.float64)
        pred_u = np.asarray(res["pred_unclipped"], dtype=np.float64)
        lo_u = np.asarray(res["lo_unclipped"], dtype=np.float64)
        hi_u = np.asarray(res["hi_unclipped"], dtype=np.float64)
        dists = np.asarray(res["distance"], dtype=np.float64)
        trusted = np.asarray(res["trusted"], dtype=bool)
        half_w = np.maximum(0.5 * (hi - lo), 0.0)
        upper_rad = np.maximum(hi - pred, 0.0)
        lower_rad = np.maximum(pred - lo, 0.0)
        upper_rad_u = np.maximum(hi_u - pred_u, 0.0)
        lower_rad_u = np.maximum(pred_u - lo_u, 0.0)

        if mode == "safe_expand":
            chosen = select_diverse_boundary_batch(
                self.model,
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
            raw_util = (pred + float(beta) * upper_rad) if maximize else -(pred - float(beta) * lower_rad)
            unclipped_util = (
                (pred_u + float(beta) * upper_rad_u)
                if maximize
                else -(pred_u - float(beta) * lower_rad_u)
            )
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
            raw_util = _gaussian_expected_improvement(pred, sigma, best_y=ref, maximize=maximize)

            half_w_u = np.maximum(0.5 * (hi_u - lo_u), 0.0)
            dir_rad_u = upper_rad_u if maximize else lower_rad_u
            eff_rad_u = np.where(dir_rad_u > 1e-9, dir_rad_u, half_w_u)
            sigma_u = np.maximum(eff_rad_u / z_div, 1e-6)
            unclipped_util = _gaussian_expected_improvement(pred_u, sigma_u, best_y=ref, maximize=maximize)
        else:
            raise ValueError(
                f"Unknown acquisition mode {mode!r}; expected 'conformal_ucb', 'conformal_ei', or 'safe_expand'"
            )

        inside_idx = np.where(trusted)[0]
        if inside_idx.size == 0:
            return self.propose(X_pool, mode="safe_expand", batch_size=batch_size, maximize=maximize)

        dir_pred_u = pred_u if maximize else -pred_u
        # np.lexsort sorts primarily by the LAST key (-raw_util), breaking clip-bound ties with -unclipped_util
        lex = np.lexsort(
            (
                inside_idx,
                -dir_pred_u[inside_idx],
                -unclipped_util[inside_idx],
                -raw_util[inside_idx],
            )
        )
        order = inside_idx[lex[:batch_size]]
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
        if self.model.whitening_matrix is not None:
            pinv = np.linalg.pinv(self.model.whitening_matrix)
            gate_x = (fit_z @ pinv) * self.model.scaler_scale + self.model.scaler_mean
        else:
            safe_w = np.where(self.model.feature_weights > 1e-12, self.model.feature_weights, 1.0)
            gate_x = (fit_z / safe_w) * self.model.scaler_scale + self.model.scaler_mean
        exp_dim = self.model.expected_input_dim
        if gate_x.shape[1] == exp_dim:
            return gate_x
        rem = exp_dim - gate_x.shape[1]
        if (
            self.model.tanimoto_ref_fp is not None
            and self.model.tanimoto_ref_fp.shape == (gate_x.shape[0], rem)
        ):
            return np.hstack([gate_x, self.model.tanimoto_ref_fp])
        return np.hstack([gate_x, np.zeros((gate_x.shape[0], rem), dtype=np.float64)])


def benchmark_trust_region_bo(
    advisor: ConformalTrustModel,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    batch_size: int = 10,
    maximize: bool = True,
) -> dict[str, float]:
    """Single-batch trust-region ranking comparison against unconstrained greedy selection."""
    tr_adv = TrustRegionAdvisor(advisor)
    constrained = tr_adv.propose(X_pool, mode="conformal_ucb", batch_size=batch_size, maximize=maximize)
    c_idx = np.asarray([p.pool_index for p in constrained], dtype=int)

    res_all = advisor.predict(X_pool, use_mondrian=True, include_unclipped=True)
    pred_all = np.asarray(res_all["pred"], dtype=np.float64)
    pred_u = np.asarray(res_all["pred_unclipped"], dtype=np.float64)
    dists_all = np.asarray(res_all["distance"], dtype=np.float64)
    pool_idx = np.arange(len(X_pool), dtype=int)
    dir_p = pred_all if maximize else -pred_all
    dir_pu = pred_u if maximize else -pred_u
    unconstrained_order = pool_idx[np.lexsort((pool_idx, -dir_pu, -dir_p))[:batch_size]]

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


def _propose_gp_ei_batch(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_pool: np.ndarray,
    *,
    batch_size: int,
    maximize: bool,
    seed: int,
    max_gp_dims: int = 24,
) -> np.ndarray:
    """Fit a scikit-learn GaussianProcessRegressor posterior and select top-`batch_size` by EI."""
    scaler = StandardScaler().fit(X_tr)
    scale = np.where(np.sqrt(scaler.var_) > 1e-12, np.sqrt(scaler.var_), 1.0)
    Z_tr = (X_tr - scaler.mean_) / scale
    Z_pool = (X_pool - scaler.mean_) / scale

    max_comp = min(max_gp_dims, Z_tr.shape[0] - 1, Z_tr.shape[1])
    if Z_tr.shape[1] > max_gp_dims and max_comp >= 2:
        pca = PCA(n_components=max_comp, random_state=seed).fit(Z_tr)
        Z_tr = pca.transform(Z_tr)
        Z_pool = pca.transform(Z_pool)

    # Cap exact Cholesky training size if campaign grows past 600 points to stay fast on CPU
    if Z_tr.shape[0] > 600:
        rng_sub = np.random.default_rng(seed)
        keep = rng_sub.choice(Z_tr.shape[0], size=600, replace=False)
        Z_tr_fit = Z_tr[keep]
        y_tr_fit = y_tr[keep]
    else:
        Z_tr_fit = Z_tr
        y_tr_fit = y_tr

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(Z_tr_fit, y_tr_fit)

    mu, sigma = gp.predict(Z_pool, return_std=True)
    best_obs = float(np.max(y_tr) if maximize else np.min(y_tr))
    ei = _gaussian_expected_improvement(mu, sigma, best_y=best_obs, maximize=maximize)
    pool_idx = np.arange(len(X_pool), dtype=int)
    dir_mu = mu if maximize else -mu
    order = pool_idx[np.lexsort((pool_idx, -dir_mu, -ei))[:batch_size]]
    return order


def run_sequential_bo_campaign(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    pool_idx: np.ndarray,
    cal_idx: np.ndarray,
    method: str = "tr_conformal_ucb",
    rounds: int = 6,
    batch_size: int = 8,
    cal_update_frac: float = 0.20,
    maximize: bool = True,
    alpha: float = 0.10,
    k: int = 5,
    trust_pct: float = 80.0,
    beta: float = 1.0,
    seed: int = 0,
    nonconformity: str = "normalized",
    distance_metric: str = "euclidean",
    surrogate: str = "gbdt",
    domain_bounds: tuple[float, float] | None = None,
    feature_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run an `R`-round sequential closed-loop acquisition campaign from `pool_idx`.

    At each round:
    1. Fit & calibrate the current round's conformal trust model on `(cur_tr, cur_cal)`.
    2. Select `batch_size` pool points according to `method`.
    3. Reveal true `y`, record `best_so_far`, `simple_regret` (against initial `pool_idx` optimum),
       and `ood_pick_fraction` (against the round's calibrated kNN applicability gate).
    4. Route acquired measurements into `(cur_tr, cur_cal)` and refit/recalibrate on the next round.
    """
    if method not in VALID_CAMPAIGN_METHODS:
        raise ValueError(f"Unknown campaign method {method!r}; expected one of {VALID_CAMPAIGN_METHODS}")
    if rounds < 1 or batch_size < 1:
        raise ValueError("rounds and batch_size must be >= 1")

    cur_tr = np.array(train_idx, dtype=int, copy=True)
    cur_cal = np.array(cal_idx, dtype=int, copy=True)
    cur_pool = np.array(pool_idx, dtype=int, copy=True)

    if cur_pool.size == 0:
        return []

    pool_opt = float(np.max(y[cur_pool]) if maximize else np.min(y[cur_pool]))
    init_train_best = float(np.max(y[cur_tr]) if maximize else np.min(y[cur_tr]))

    acquired_y_all: list[float] = []
    total_picks = 0
    total_ood_picks = 0
    history: list[dict[str, Any]] = []

    use_lgbm_cqr = method in ("lightgbm_cqr_ucb", "lightgbm_cqr_tr_ucb")
    eff_surrogate = "lightgbm" if use_lgbm_cqr else surrogate
    eff_nonconf = "cqr_r" if use_lgbm_cqr else nonconformity

    for r in range(1, rounds + 1):
        if cur_pool.size == 0:
            break
        n_take = min(int(batch_size), int(cur_pool.size))

        advisor = fit_and_calibrate(
            X[cur_tr],
            y[cur_tr],
            X[cur_cal],
            y[cur_cal],
            alpha=alpha,
            k=k,
            trust_pct=trust_pct,
            seed=seed,
            nonconformity=eff_nonconf,
            distance_metric=distance_metric,
            surrogate=eff_surrogate,
            domain_bounds=domain_bounds,
            feature_names=feature_names,
        )

        X_p = X[cur_pool]
        res_p = advisor.predict(X_p, use_mondrian=True, include_unclipped=True)
        trusted_p = np.asarray(res_p["trusted"], dtype=bool)
        pred_p = np.asarray(res_p["pred"], dtype=np.float64)
        pred_u = np.asarray(res_p["pred_unclipped"], dtype=np.float64)
        lo_p = np.asarray(res_p["lo"], dtype=np.float64)
        hi_p = np.asarray(res_p["hi"], dtype=np.float64)
        lo_u = np.asarray(res_p["lo_unclipped"], dtype=np.float64)
        hi_u = np.asarray(res_p["hi_unclipped"], dtype=np.float64)
        p_local_idx = np.arange(len(cur_pool), dtype=int)

        if method == "random":
            rng = np.random.default_rng(seed * 10007 + r)
            take_pos = np.sort(rng.choice(len(cur_pool), size=n_take, replace=False))
        elif method == "greedy":
            dir_p = pred_p if maximize else -pred_p
            dir_pu = pred_u if maximize else -pred_u
            take_pos = p_local_idx[np.lexsort((p_local_idx, -dir_pu, -dir_p))[:n_take]]
        elif method in ("unconstrained_conformal_ucb", "lightgbm_cqr_ucb"):
            up_rad = np.maximum(hi_p - pred_p, 0.0)
            lo_rad = np.maximum(pred_p - lo_p, 0.0)
            up_rad_u = np.maximum(hi_u - pred_u, 0.0)
            lo_rad_u = np.maximum(pred_u - lo_u, 0.0)
            ucb = (pred_p + float(beta) * up_rad) if maximize else -(pred_p - float(beta) * lo_rad)
            ucb_u = (pred_u + float(beta) * up_rad_u) if maximize else -(pred_u - float(beta) * lo_rad_u)
            dir_pu = pred_u if maximize else -pred_u
            take_pos = p_local_idx[np.lexsort((p_local_idx, -dir_pu, -ucb_u, -ucb))[:n_take]]
        elif method in ("tr_conformal_ucb", "lightgbm_cqr_tr_ucb"):
            props = TrustRegionAdvisor(advisor).propose(
                X_p, mode="conformal_ucb", batch_size=n_take, maximize=maximize, beta=beta
            )
            take_pos = np.asarray([p.pool_index for p in props], dtype=int)
        elif method == "tr_conformal_ei":
            cur_best_obs = float(np.max(y[cur_tr]) if maximize else np.min(y[cur_tr]))
            props = TrustRegionAdvisor(advisor).propose(
                X_p,
                mode="conformal_ei",
                batch_size=n_take,
                maximize=maximize,
                best_y=cur_best_obs,
            )
            take_pos = np.asarray([p.pool_index for p in props], dtype=int)
        elif method == "gp_ei":
            take_pos = _propose_gp_ei_batch(
                X[cur_tr],
                y[cur_tr],
                X_p,
                batch_size=n_take,
                maximize=maximize,
                seed=seed * 101 + r,
            )
        else:
            raise AssertionError(f"Unhandled method {method!r}")

        acquired_global = cur_pool[take_pos]
        revealed_y = y[acquired_global]
        acquired_y_all.extend(float(v) for v in revealed_y.tolist())

        ood_in_batch = int(np.sum(~trusted_p[take_pos]))
        total_picks += int(take_pos.size)
        total_ood_picks += ood_in_batch

        best_pool_so_far = float(max(acquired_y_all) if maximize else min(acquired_y_all))
        best_overall = (
            max(init_train_best, best_pool_so_far)
            if maximize
            else min(init_train_best, best_pool_so_far)
        )
        regret = max(
            0.0,
            (pool_opt - best_pool_so_far) if maximize else (best_pool_so_far - pool_opt),
        )

        history.append(
            {
                "round": r,
                "method": method,
                "n_acquired_batch": int(take_pos.size),
                "n_acquired_total": total_picks,
                "pool_optimum": pool_opt,
                "best_so_far": best_pool_so_far,
                "best_overall_so_far": best_overall,
                "simple_regret": regret,
                "round_ood_fraction": float(ood_in_batch) / float(take_pos.size),
                "cumulative_ood_fraction": float(total_ood_picks) / float(total_picks),
            }
        )

        cur_pool = np.delete(cur_pool, take_pos)
        n_to_cal = int(round(acquired_global.size * cal_update_frac))
        if n_to_cal > 0 and acquired_global.size >= 4:
            step = max(1, acquired_global.size // n_to_cal)
            cal_mask = np.zeros(acquired_global.size, dtype=bool)
            cal_mask[::step][:n_to_cal] = True
            cur_cal = np.concatenate([cur_cal, acquired_global[cal_mask]])
            cur_tr = np.concatenate([cur_tr, acquired_global[~cal_mask]])
        else:
            cur_tr = np.concatenate([cur_tr, acquired_global])

    return history
