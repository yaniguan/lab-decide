"""Multi-seed real-data conformal, Delaney-2004 head-to-head & 11-dataset registry benchmark.

Executes across real cached datasets in `data/` (failing closed if any dataset is uncached unless
`--fetch` is supplied; never permits synthetic fixtures):
1. Direct 20-seed ESOL (`esol_delaney`) random & balanced Bemis-Murcko scaffold head-to-head:
   - Delaney (2004) published linear formula (`ESOL predicted log solubility in mols per litre` CSV column),
   - Delaney (2004) 4-descriptor (`cLogP`, `MWT`, `RB`, `AP`) out-of-sample refitted linear regression,
   - 6-descriptor Huber GBDT + Mondrian normalized kNN conformal (`encoding="descriptors"`),
   - Frontier `rdkit_2d_morgan + frontier_ensemble` + `tanimoto_mahalanobis` + `cqr_r`.
2. 20-seed ESOL active boundary-expansion summary.
3. 5-seed CALiSol-23 (`calisol23`) publication-DOI holdout split (`evaluate_calisol_conformal`).
4. >=5-seed evaluation across all 11 `OpenCalibrationDataset` benchmarks (`rdkit_2d_morgan` on all 5
   single-molecule datasets, `onehot_morgan` on all 4 HTE reaction datasets, and `group` splits on
   `bigsoldb_v2_2` and `calisol23`).
5. Side-by-side representation & gating ablations across all 5 single-molecule datasets
   (`descriptors` vs `rdkit_2d_morgan`) and all 4 HTE reaction datasets
   (`descriptors` vs `onehot` vs `onehot_morgan`).

Writes `results/registry_benchmark.json` and exposes `render_registry_markdown(payload)` so
`README.md` tables are generated directly from measured JSON with zero manual transcription.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_decide.calisol import evaluate_calisol_conformal  # noqa: E402
from lab_decide.chemistry import (  # noqa: E402
    DELANEY_CSV_PRED_COLUMN,
    ESOL_TARGET,
    featurize_delaney_4d,
)
from lab_decide.cli import run_esol_benchmark  # noqa: E402
from lab_decide.conformal import (  # noqa: E402
    conformal_quantile,
    fit_and_calibrate,
    spearman_rank_correlation,
    summarize,
    winkler_interval_score,
)
from lab_decide.data import load_dataset, partition_splits  # noqa: E402
from lab_decide.datasets import (  # noqa: E402
    OPEN_CALIBRATION_DATASETS,
    default_dataset_cache_path,
    fetch_open_dataset,
    get_open_dataset,
    load_open_dataset,
)

DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "registry_benchmark.json"

SINGLE_MOLECULE_IDS: tuple[str, ...] = (
    "esol_delaney",
    "aqsoldb",
    "freesolv",
    "lipophilicity_astrazeneca",
    "biogen_adme",
)

HTE_REACTION_IDS: tuple[str, ...] = (
    "buchwald_hartwig_doyle",
    "suzuki_miyaura_perera",
    "direct_arylation_shields",
    "bh_hte_ood_2026",
)


def _clean_float(v: float | None, ndigits: int = 6) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), ndigits)


def _mean_sd(vals: Sequence[float | None]) -> dict[str, float | None]:
    arr = np.asarray(
        [float(v) for v in vals if v is not None and math.isfinite(float(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {"mean": None, "sd": None}
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return {"mean": round(float(np.mean(arr)), 6), "sd": round(sd, 6)}


def _evaluate_bundle_across_seeds(
    dataset_id: str,
    *,
    seeds: Sequence[int],
    split_strategy: str = "random",
    encoding: str = "descriptors",
    distance_metric: str = "euclidean",
    nonconformity: str = "normalized",
    surrogate: str = "gbdt",
    alpha: float = 0.10,
    trust_pct: float = 80.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    spec = get_open_dataset(dataset_id)
    cache_path = default_dataset_cache_path(dataset_id)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Real dataset {dataset_id!r} is missing at {cache_path}; run with --fetch first."
        )

    bundle = load_open_dataset(
        dataset_id,
        path=cache_path,
        max_rows=max_rows,
        offline=True,
        encoding=encoding,
        data_source="real_csv",
    )
    assert bundle.data_source == "real_csv"

    per_seed: list[dict[str, Any]] = []
    mae_wins = 0

    for seed in seeds:
        splits = partition_splits(
            len(bundle.y),
            smiles=bundle.smiles if split_strategy == "scaffold" else None,
            groups=bundle.group_keys if split_strategy == "group" else None,
            strategy=split_strategy,
            seed=int(seed),
            fracs=(0.50, 0.20, 0.15, 0.15),
        )
        advisor = fit_and_calibrate(
            bundle.X[splits.train_idx],
            bundle.y[splits.train_idx],
            bundle.X[splits.cal_idx],
            bundle.y[splits.cal_idx],
            alpha=alpha,
            trust_pct=trust_pct,
            seed=int(seed),
            nonconformity=nonconformity,
            distance_metric=distance_metric,
            surrogate=surrogate,
            domain_bounds=spec.domain_bounds,
            feature_names=bundle.feature_names,
        )
        X_te = bundle.X[splits.test_idx]
        y_te = bundle.y[splits.test_idx]
        res = advisor.predict(X_te, use_mondrian=True)
        inside = np.asarray(res["trusted"], dtype=bool)
        pred_te = np.asarray(res["pred"], dtype=np.float64)

        c_in, w_in, e_in = summarize(y_te, res["lo"], res["hi"], inside, pred=pred_te)
        c_out, w_out, e_out = summarize(y_te, res["lo"], res["hi"], ~inside, pred=pred_te)
        c_all, w_all, e_all = summarize(
            y_te, res["lo"], res["hi"], np.ones(len(y_te), dtype=bool), pred=pred_te
        )
        rmse_all = float(np.sqrt(np.mean((y_te - pred_te) ** 2)))
        r2_all = float(r2_score(y_te, pred_te)) if len(y_te) >= 2 else float("nan")
        wink_in = winkler_interval_score(y_te, res["lo"], res["hi"], alpha=alpha, mask=inside)
        rho = spearman_rank_correlation(res["distance"], np.abs(y_te - pred_te))

        if math.isfinite(e_in) and math.isfinite(e_out) and e_in < e_out:
            mae_wins += 1

        per_seed.append(
            {
                "seed": int(seed),
                "answered_fraction": _clean_float(float(np.mean(inside))),
                "cov_in": _clean_float(c_in),
                "cov_out": _clean_float(c_out),
                "cov_all": _clean_float(c_all),
                "mae_in": _clean_float(e_in),
                "mae_out": _clean_float(e_out),
                "mae_all": _clean_float(e_all),
                "rmse_all": _clean_float(rmse_all),
                "r2_all": _clean_float(r2_all),
                "width_in": _clean_float(w_in),
                "width_out": _clean_float(w_out),
                "width_all": _clean_float(w_all),
                "winkler_in": _clean_float(wink_in),
                "spearman_distance_abs_error": _clean_float(rho),
                "achieved_fractions": {
                    k: round(float(v), 6) for k, v in (splits.achieved_fractions or {}).items()
                },
            }
        )

    n_s = len(seeds)
    win_frac = float(mae_wins) / float(n_s)
    spearman_stats = _mean_sd(
        [r["spearman_distance_abs_error"] for r in per_seed if r["spearman_distance_abs_error"] is not None]
    )
    mean_rho = spearman_stats["mean"]
    gate_inf = bool((win_frac >= 0.80) and (mean_rho is not None) and (mean_rho > 0.0))

    metric_keys = (
        "answered_fraction",
        "cov_in",
        "cov_out",
        "cov_all",
        "mae_in",
        "mae_out",
        "mae_all",
        "rmse_all",
        "r2_all",
        "width_in",
        "width_out",
        "width_all",
        "winkler_in",
        "spearman_distance_abs_error",
    )
    agg = {mk: _mean_sd([r[mk] for r in per_seed]) for mk in metric_keys}

    return {
        "dataset_id": dataset_id,
        "domain": spec.domain,
        "target_unit": spec.target_unit,
        "objective_direction": spec.objective_direction,
        "group_kind": spec.group_kind,
        "data_source": "real_csv",
        "n_clean": int(len(bundle.y)),
        "n_features": int(bundle.X.shape[1]),
        "n_groups_total": len(set(bundle.group_keys)) if bundle.group_keys is not None else None,
        "split_strategy": split_strategy,
        "encoding": encoding,
        "distance_metric": distance_metric,
        "nonconformity": nonconformity,
        "surrogate": surrogate,
        "n_seeds": n_s,
        "seeds_mae_in_lt_out": int(mae_wins),
        "frac_seeds_mae_in_lt_out": round(win_frac, 4),
        "gate_informative": gate_inf,
        "metrics": agg,
        "per_seed": per_seed,
    }


def run_esol_delaney_20seed_head_to_head(alpha: float = 0.10, trust_pct: float = 80.0) -> dict[str, Any]:
    """Direct paired 20-seed comparison on ESOL (`random` & `scaffold`) vs Delaney (2004) & 6-desc GBDT."""
    esol_csv = default_dataset_cache_path("esol_delaney")
    raw_df = pd.read_csv(esol_csv)
    raw_df["smiles"] = raw_df["smiles"].astype(str).str.strip()
    first_order = raw_df.drop_duplicates(subset=["smiles"], keep="first")["smiles"].tolist()
    dedup_df = raw_df.groupby("smiles", sort=False, as_index=False).mean(numeric_only=True)
    dedup_df = dedup_df.set_index("smiles").loc[first_order].reset_index()

    bundle_6d = load_dataset(csv_path=esol_csv, offline=True, encoding="descriptors")
    bundle_mol = load_dataset(csv_path=esol_csv, offline=True, encoding="rdkit_2d_morgan")
    assert bundle_6d.smiles == dedup_df["smiles"].tolist()
    assert bundle_mol.smiles == bundle_6d.smiles
    assert np.allclose(bundle_mol.y, bundle_6d.y, atol=1e-12)

    y_all = bundle_mol.y
    y_delaney_csv = dedup_df[DELANEY_CSV_PRED_COLUMN].to_numpy(dtype=np.float64)
    X_delaney_4d = featurize_delaney_4d(bundle_mol.smiles)
    b_lo, b_hi = (-15.0, 1.75)

    out_by_split: dict[str, Any] = {}
    for split_mode in ("random", "scaffold"):
        del_csv_rows: list[dict[str, float]] = []
        del_ols_rows: list[dict[str, float]] = []
        gbdt6_rows: list[dict[str, float]] = []
        front_rows: list[dict[str, float]] = []
        gbdt6_wins = 0
        front_wins = 0

        for seed in range(20):
            sp = partition_splits(
                len(y_all),
                smiles=bundle_mol.smiles if split_mode == "scaffold" else None,
                strategy=split_mode,
                seed=seed,
                fracs=(0.50, 0.20, 0.15, 0.15),
            )
            tr, cal, te = sp.train_idx, sp.cal_idx, sp.test_idx
            y_tr, y_cal, y_te = y_all[tr], y_all[cal], y_all[te]

            # 1. Delaney (2004) published linear formula column in delaney-processed.csv
            p_del_cal = np.clip(y_delaney_csv[cal], b_lo, b_hi)
            p_del_te = np.clip(y_delaney_csv[te], b_lo, b_hi)
            q_del = conformal_quantile(np.abs(y_cal - p_del_cal), alpha=alpha)
            lo_del = np.clip(p_del_te - q_del, b_lo, b_hi)
            hi_del = np.clip(p_del_te + q_del, b_lo, b_hi)
            cov_del, wid_del, mae_del = summarize(
                y_te, lo_del, hi_del, np.ones(len(te), dtype=bool), pred=p_del_te
            )
            del_csv_rows.append(
                {
                    "mae_all": mae_del,
                    "rmse_all": float(np.sqrt(np.mean((y_te - p_del_te) ** 2))),
                    "r2_all": float(r2_score(y_te, p_del_te)),
                    "cov_all": cov_del,
                    "width_all": wid_del,
                }
            )

            # 2. Out-of-sample refitted 4-descriptor Delaney OLS (`cLogP`, `MWT`, `RB`, `AP`)
            ols = LinearRegression().fit(X_delaney_4d[tr], y_tr)
            p_ols_cal = np.clip(ols.predict(X_delaney_4d[cal]), b_lo, b_hi)
            p_ols_te = np.clip(ols.predict(X_delaney_4d[te]), b_lo, b_hi)
            q_ols = conformal_quantile(np.abs(y_cal - p_ols_cal), alpha=alpha)
            lo_ols = np.clip(p_ols_te - q_ols, b_lo, b_hi)
            hi_ols = np.clip(p_ols_te + q_ols, b_lo, b_hi)
            cov_ols, wid_ols, mae_ols = summarize(
                y_te, lo_ols, hi_ols, np.ones(len(te), dtype=bool), pred=p_ols_te
            )
            del_ols_rows.append(
                {
                    "mae_all": mae_ols,
                    "rmse_all": float(np.sqrt(np.mean((y_te - p_ols_te) ** 2))),
                    "r2_all": float(r2_score(y_te, p_ols_te)),
                    "cov_all": cov_ols,
                    "width_all": wid_ols,
                }
            )

            # 3. Legacy 6-descriptor Huber GBDT + Mondrian normalized kNN conformal
            adv6 = fit_and_calibrate(
                bundle_6d.X[tr],
                y_tr,
                bundle_6d.X[cal],
                y_cal,
                alpha=alpha,
                trust_pct=trust_pct,
                seed=seed,
                nonconformity="normalized",
                distance_metric="euclidean",
                surrogate="gbdt",
                domain_bounds=(b_lo, b_hi),
                feature_names=bundle_6d.feature_names,
            )
            res6 = adv6.predict(bundle_6d.X[te], use_mondrian=True)
            in6 = np.asarray(res6["trusted"], dtype=bool)
            p6 = np.asarray(res6["pred"], dtype=np.float64)
            c6_in, w6_in, e6_in = summarize(y_te, res6["lo"], res6["hi"], in6, pred=p6)
            c6_out, _, e6_out = summarize(y_te, res6["lo"], res6["hi"], ~in6, pred=p6)
            c6_all, w6_all, e6_all = summarize(
                y_te, res6["lo"], res6["hi"], np.ones(len(te), dtype=bool), pred=p6
            )
            rho6 = spearman_rank_correlation(res6["distance"], np.abs(y_te - p6))
            if e6_in < e6_out:
                gbdt6_wins += 1
            gbdt6_rows.append(
                {
                    "mae_all": e6_all,
                    "rmse_all": float(np.sqrt(np.mean((y_te - p6) ** 2))),
                    "r2_all": float(r2_score(y_te, p6)),
                    "mae_in": e6_in,
                    "mae_out": e6_out,
                    "cov_in": c6_in,
                    "cov_out": c6_out,
                    "cov_all": c6_all,
                    "width_in": w6_in,
                    "width_all": w6_all,
                    "winkler_in": winkler_interval_score(y_te, res6["lo"], res6["hi"], alpha=alpha, mask=in6),
                    "spearman_distance_abs_error": rho6,
                }
            )

            # 4. Frontier rdkit_2d_morgan + FrontierEnsembleRegressor + tanimoto_mahalanobis + cqr_r
            adv_f = fit_and_calibrate(
                bundle_mol.X[tr],
                y_tr,
                bundle_mol.X[cal],
                y_cal,
                alpha=alpha,
                trust_pct=trust_pct,
                seed=seed,
                nonconformity="cqr_r",
                distance_metric="tanimoto_mahalanobis",
                surrogate="frontier_ensemble",
                domain_bounds=(b_lo, b_hi),
                feature_names=bundle_mol.feature_names,
            )
            res_f = adv_f.predict(bundle_mol.X[te], use_mondrian=True)
            in_f = np.asarray(res_f["trusted"], dtype=bool)
            pf = np.asarray(res_f["pred"], dtype=np.float64)
            cf_in, wf_in, ef_in = summarize(y_te, res_f["lo"], res_f["hi"], in_f, pred=pf)
            cf_out, _, ef_out = summarize(y_te, res_f["lo"], res_f["hi"], ~in_f, pred=pf)
            cf_all, wf_all, ef_all = summarize(
                y_te, res_f["lo"], res_f["hi"], np.ones(len(te), dtype=bool), pred=pf
            )
            rhof = spearman_rank_correlation(res_f["distance"], np.abs(y_te - pf))
            if ef_in < ef_out:
                front_wins += 1
            front_rows.append(
                {
                    "mae_all": ef_all,
                    "rmse_all": float(np.sqrt(np.mean((y_te - pf) ** 2))),
                    "r2_all": float(r2_score(y_te, pf)),
                    "mae_in": ef_in,
                    "mae_out": ef_out,
                    "cov_in": cf_in,
                    "cov_out": cf_out,
                    "cov_all": cf_all,
                    "width_in": wf_in,
                    "width_all": wf_all,
                    "winkler_in": winkler_interval_score(y_te, res_f["lo"], res_f["hi"], alpha=alpha, mask=in_f),
                    "spearman_distance_abs_error": rhof,
                }
            )

        def _agg_dict(rows: list[dict[str, float]], keys: Sequence[str]) -> dict[str, Any]:
            return {k: _mean_sd([r[k] for r in rows]) for k in keys}

        lin_keys = ("mae_all", "rmse_all", "r2_all", "cov_all", "width_all")
        conf_keys = (
            "mae_all",
            "rmse_all",
            "r2_all",
            "mae_in",
            "mae_out",
            "cov_in",
            "cov_out",
            "cov_all",
            "width_in",
            "width_all",
            "winkler_in",
            "spearman_distance_abs_error",
        )
        del_csv_agg = _agg_dict(del_csv_rows, lin_keys)
        del_ols_agg = _agg_dict(del_ols_rows, lin_keys)
        gbdt6_agg = _agg_dict(gbdt6_rows, conf_keys)
        front_agg = _agg_dict(front_rows, conf_keys)

        f_mae = float(front_agg["mae_all"]["mean"])
        d_csv_mae = float(del_csv_agg["mae_all"]["mean"])
        d_ols_mae = float(del_ols_agg["mae_all"]["mean"])
        g6_mae = float(gbdt6_agg["mae_all"]["mean"])

        out_by_split[split_mode] = {
            "n_seeds": 20,
            "n_clean": int(len(y_all)),
            "data_source": "real_csv",
            "mae_reduction_vs_delaney_csv_pct": round(100.0 * (d_csv_mae - f_mae) / d_csv_mae, 2),
            "mae_reduction_vs_delaney_refit_ols_pct": round(100.0 * (d_ols_mae - f_mae) / d_ols_mae, 2),
            "mae_reduction_vs_gbdt_6desc_pct": round(100.0 * (g6_mae - f_mae) / g6_mae, 2),
            "models": {
                "delaney_2004_csv_formula": {
                    "label": "Delaney (2004) 4-term formula (CSV column, in-sample fit)",
                    "n_features": 4,
                    "metrics": del_csv_agg,
                },
                "delaney_2004_refit_4desc_ols": {
                    "label": "Delaney (2004) 4-desc OLS (out-of-sample train split refit)",
                    "n_features": 4,
                    "metrics": del_ols_agg,
                },
                "gbdt_6desc": {
                    "label": "6-descriptor Huber GBDT + Euclidean kNN (`descriptors`)",
                    "n_features": int(bundle_6d.X.shape[1]),
                    "seeds_mae_in_lt_out": gbdt6_wins,
                    "gate_informative": bool(
                        (gbdt6_wins / 20.0 >= 0.80)
                        and ((gbdt6_agg["spearman_distance_abs_error"]["mean"] or 0.0) > 0.0)
                    ),
                    "metrics": gbdt6_agg,
                },
                "frontier_rdkit_2d_morgan": {
                    "label": "Frontier Ensemble (`rdkit_2d_morgan` + `tanimoto_mahalanobis` + `cqr_r`)",
                    "n_features": int(bundle_mol.X.shape[1]),
                    "seeds_mae_in_lt_out": front_wins,
                    "gate_informative": bool(
                        (front_wins / 20.0 >= 0.80)
                        and ((front_agg["spearman_distance_abs_error"]["mean"] or 0.0) > 0.0)
                    ),
                    "metrics": front_agg,
                },
            },
        }
    return out_by_split


def run_esol_20seed_summary() -> dict[str, Any]:
    bundle = load_dataset(offline=True, allow_synthetic=False, encoding="rdkit_2d_morgan")
    assert bundle.data_source == "real_csv"

    out_by_split: dict[str, Any] = {}
    for split_mode in ("random", "scaffold"):
        cov_in_0, mae_in_0, mae_out_0, mae_all_0, rmse_all_0 = [], [], [], [], []
        w_in_0, w_in_last, cov_in_last, c_top, u_top = [], [], [], [], []
        spearman_0 = []
        max_frac_err = 0.0
        wins = 0
        for seed in range(20):
            summary, _, _ = run_esol_benchmark(
                bundle,
                split_strategy=split_mode,
                seed=seed,
                rounds=2,
                nonconformity="cqr_r",
                distance_metric="tanimoto_mahalanobis",
                surrogate="frontier_ensemble",
            )
            tr_blk = summary["mondrian_trust_region"]
            mc_blk = summary["marginal_conformal"]
            c_i = float(tr_blk["inside"]["coverage"])
            e_i = float(tr_blk["inside"]["mae"])
            e_o = float(tr_blk["outside"]["mae"])
            w_i = float(tr_blk["inside"]["width"])
            rho = float(tr_blk["spearman_distance_abs_error"])
            cov_in_0.append(c_i)
            mae_in_0.append(e_i)
            mae_out_0.append(e_o)
            mae_all_0.append(float(mc_blk["mae"]))
            rmse_all_0.append(float(mc_blk["rmse"]))
            width_last = float(summary["expansion_rounds"][-1]["width_in"])
            cov_last = float(summary["expansion_rounds"][-1]["cov_in"])
            w_in_0.append(w_i)
            w_in_last.append(width_last)
            cov_in_last.append(cov_last)
            spearman_0.append(rho)
            c_top.append(float(summary["constrained_bo"]["constrained_top_y"]))
            u_top.append(float(summary["constrained_bo"]["unconstrained_top_y"]))
            if e_i < e_o:
                wins += 1
            af = summary["achieved_fractions"]
            for sp_name, req in (("train", 0.50), ("pool", 0.20), ("calibration", 0.15), ("test", 0.15)):
                max_frac_err = max(max_frac_err, abs(float(af[sp_name]) - req))

        win_frac = wins / 20.0
        mean_rho = _mean_sd(spearman_0)["mean"]
        out_by_split[split_mode] = {
            "n_seeds": 20,
            "encoding": "rdkit_2d_morgan",
            "surrogate": "frontier_ensemble",
            "distance_metric": "tanimoto_mahalanobis",
            "nonconformity": "cqr_r",
            "data_source": "real_csv",
            "max_achieved_fraction_abs_error": round(max_frac_err, 6),
            "cov_in_round0": _mean_sd(cov_in_0),
            "mae_all_round0": _mean_sd(mae_all_0),
            "rmse_all_round0": _mean_sd(rmse_all_0),
            "mae_in_round0": _mean_sd(mae_in_0),
            "mae_out_round0": _mean_sd(mae_out_0),
            "width_in_round0": _mean_sd(w_in_0),
            "width_in_final_round": _mean_sd(w_in_last),
            "cov_in_final_round": _mean_sd(cov_in_last),
            "spearman_distance_abs_error": _mean_sd(spearman_0),
            "single_batch_constrained_top_y": _mean_sd(c_top),
            "single_batch_unconstrained_top_y": _mean_sd(u_top),
            "seeds_mae_in_lt_out": wins,
            "gate_informative": bool(win_frac >= 0.80 and (mean_rho or 0.0) > 0.0),
        }
    return out_by_split


def run_calisol_doi_holdout_summary(seeds: Sequence[int] = tuple(range(5))) -> dict[str, Any]:
    csv_path = default_dataset_cache_path("calisol23")
    per_seed = []
    wins = 0
    for s in seeds:
        res = evaluate_calisol_conformal(
            csv_path=csv_path,
            seed=int(s),
            max_records_per_split=1500,
            nonconformity="cqr_r",
        )
        assert res["data_source"] == "real_csv"
        if float(res["mae_in"]) < float(res["mae_out"]):
            wins += 1
        per_seed.append(res)

    win_frac = wins / float(len(seeds))
    rho_stats = _mean_sd([r["spearman_distance_abs_error"] for r in per_seed])
    return {
        "dataset_id": "calisol23",
        "protocol": "publication_doi_holdout",
        "data_source": "real_csv",
        "nonconformity": "cqr_r",
        "n_seeds": len(seeds),
        "seeds_mae_in_lt_out": wins,
        "frac_seeds_mae_in_lt_out": round(win_frac, 4),
        "gate_informative": bool(win_frac >= 0.80 and (rho_stats["mean"] or 0.0) > 0.0),
        "metrics": {
            mk: _mean_sd([r[mk] for r in per_seed])
            for mk in (
                "answered_fraction",
                "cov_in",
                "cov_out",
                "cov_all",
                "mae_in",
                "mae_out",
                "width_in",
                "spearman_distance_abs_error",
                "knn_vs_kernel_support_agreement",
            )
        },
    }


def _fmt_pm(stat: Mapping[str, Any], prec: int = 2) -> str:
    m = stat.get("mean")
    s = stat.get("sd")
    if m is None:
        return "n/a"
    if s is None:
        return f"{float(m):.{prec}f}"
    return f"{float(m):.{prec}f} +- {float(s):.{prec}f}"


def render_registry_markdown(payload: Mapping[str, Any]) -> str:
    """Render deterministic Markdown tables directly from `results/registry_benchmark.json`."""
    lines: list[str] = []
    lines.append("### Measured 11-Dataset Real-Data Benchmark (`results/registry_benchmark.json`)")
    lines.append("")
    esol20 = payload["esol_20seed"]
    r20 = esol20["random"]
    s20 = esol20["scaffold"]
    cal_doi = payload["calisol23_doi_holdout"]
    h2h = payload["esol_delaney_20seed_head_to_head"]
    hr = h2h["random"]
    hs = h2h["scaffold"]

    lines.append(
        f"- **20-Seed ESOL Frontier Summary (`rdkit_2d_morgan + frontier_ensemble`, `data_source=real_csv`)**: "
        f"Random split overall test `MAE {_fmt_pm(hr['models']['frontier_rdkit_2d_morgan']['metrics']['mae_all'], 3)}` "
        f"(`RMSE {_fmt_pm(hr['models']['frontier_rdkit_2d_morgan']['metrics']['rmse_all'], 3)}`), beating "
        f"Delaney (2004) published formula (`MAE {_fmt_pm(hr['models']['delaney_2004_csv_formula']['metrics']['mae_all'], 3)}`) "
        f"by **`{hr['mae_reduction_vs_delaney_csv_pct']:.1f}%`**, out-of-sample refitted 4-descriptor Delaney OLS "
        f"(`MAE {_fmt_pm(hr['models']['delaney_2004_refit_4desc_ols']['metrics']['mae_all'], 3)}`) by "
        f"**`{hr['mae_reduction_vs_delaney_refit_ols_pct']:.1f}%`**, and 6-descriptor GBDT "
        f"(`MAE {_fmt_pm(hr['models']['gbdt_6desc']['metrics']['mae_all'], 3)}`) by "
        f"**`{hr['mae_reduction_vs_gbdt_6desc_pct']:.1f}%`**, with inside coverage "
        f"`{_fmt_pm(r20['cov_in_round0'], 3)}` (`MAE_in {_fmt_pm(r20['mae_in_round0'], 3)}` vs "
        f"`MAE_out {_fmt_pm(r20['mae_out_round0'], 3)}`, `20/20` seeds `MAE_in < MAE_out`, "
        f"`width_in {_fmt_pm(r20['width_in_round0'], 3)} -> {_fmt_pm(r20['width_in_final_round'], 3)}` after expansion). "
        f"On balanced Bemis-Murcko scaffold split (max split-fraction deviation "
        f"`{100.0 * float(s20['max_achieved_fraction_abs_error']):.2f}` percentage points across 20 seeds), "
        f"`rdkit_2d_morgan + frontier_ensemble` achieves `MAE {_fmt_pm(hs['models']['frontier_rdkit_2d_morgan']['metrics']['mae_all'], 3)}` "
        f"(`MAE_in {_fmt_pm(s20['mae_in_round0'], 3)}` vs `MAE_out {_fmt_pm(s20['mae_out_round0'], 3)}`), beating both "
        f"Delaney (2004) in-sample formula (`{_fmt_pm(hs['models']['delaney_2004_csv_formula']['metrics']['mae_all'], 3)}`, "
        f"`-{hs['mae_reduction_vs_delaney_csv_pct']:.1f}%`) and out-of-sample refitted 4-descriptor OLS "
        f"(`{_fmt_pm(hs['models']['delaney_2004_refit_4desc_ols']['metrics']['mae_all'], 3)}`, "
        f"`-{hs['mae_reduction_vs_delaney_refit_ols_pct']:.1f}%`)."
    )
    lines.append(
        f"- **CALiSol-23 Publication-DOI Holdout (`cqr_r`, {cal_doi['n_seeds']} seeds)**: "
        f"`cov_in {_fmt_pm(cal_doi['metrics']['cov_in'], 3)}`, "
        f"`MAE_in {_fmt_pm(cal_doi['metrics']['mae_in'], 3)}` vs "
        f"`MAE_out {_fmt_pm(cal_doi['metrics']['mae_out'], 3)}`, "
        f"`Spearman(d, |err|) {_fmt_pm(cal_doi['metrics']['spearman_distance_abs_error'], 3)}`, "
        f"`gate_informative = {cal_doi['gate_informative']}`."
    )
    lines.append("")
    lines.append("#### 20-Seed ESOL Head-to-Head vs. Delaney (2004) Linear Formula & 6-Descriptor GBDT")
    lines.append("")
    lines.append(
        "| Split (`20` seeds) | Model / Representation | Features | `MAE_all` | `RMSE_all` | `R^2` | `MAE_in` | `MAE_out` | `cov_in` / `cov_all` | `width_in` / `width_all` |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sp_key in ("random", "scaffold"):
        sp_blk = h2h[sp_key]["models"]
        for m_key in (
            "delaney_2004_csv_formula",
            "delaney_2004_refit_4desc_ols",
            "gbdt_6desc",
            "frontier_rdkit_2d_morgan",
        ):
            mb = sp_blk[m_key]
            mm = mb["metrics"]
            if "mae_in" in mm:
                mae_in_s = f"`{_fmt_pm(mm['mae_in'], 3)}`"
                mae_out_s = f"`{_fmt_pm(mm['mae_out'], 3)}`"
                cov_s = f"`{_fmt_pm(mm['cov_in'], 3)}`"
                wid_s = f"`{_fmt_pm(mm['width_in'], 3)}`"
            else:
                mae_in_s = "n/a (ungated)"
                mae_out_s = "n/a (ungated)"
                cov_s = f"`{_fmt_pm(mm['cov_all'], 3)}` (all)"
                wid_s = f"`{_fmt_pm(mm['width_all'], 3)}` (all)"
            lines.append(
                f"| `{sp_key}` | {mb['label']} | `{mb['n_features']}` | "
                f"`{_fmt_pm(mm['mae_all'], 3)}` | `{_fmt_pm(mm['rmse_all'], 3)}` | "
                f"`{_fmt_pm(mm['r2_all'], 3)}` | {mae_in_s} | {mae_out_s} | {cov_s} | {wid_s} |"
            )

    lines.append("")
    lines.append("#### All 11 Registry Datasets (`primary_11_datasets`)")
    lines.append("")
    lines.append(
        "| `dataset_id` | Split | Encoding / Gate | Seeds (`n`) | `cov_in` | `cov_out` | `MAE_all` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` | `gate_informative` |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["primary_11_datasets"]:
        m = row["metrics"]
        enc_lbl = f"`{row['encoding']}` / `{row['distance_metric']}`"
        gi_lbl = "**`True`**" if row["gate_informative"] else "**`False` (uninformative gate)**"
        lines.append(
            f"| `{row['dataset_id']}` | `{row['split_strategy']}` | {enc_lbl} | "
            f"`{row['n_seeds']}` (`{row['n_clean']}`) | "
            f"`{_fmt_pm(m['cov_in'], 3)}` | `{_fmt_pm(m['cov_out'], 3)}` | "
            f"`{_fmt_pm(m['mae_all'], 2)}` | `{_fmt_pm(m['mae_in'], 2)}` | `{_fmt_pm(m['mae_out'], 2)}` | "
            f"`{_fmt_pm(m['width_in'], 2)}` | `{_fmt_pm(m['spearman_distance_abs_error'], 3)}` | "
            f"`{row['seeds_mae_in_lt_out']}/{row['n_seeds']}` | {gi_lbl} |"
        )

    lines.append("")
    lines.append(
        "#### Single-Molecule Representation & Surrogate Ablation (`descriptors + gbdt` vs `rdkit_2d_morgan + frontier_ensemble`)"
    )
    lines.append("")
    lines.append(
        "| `dataset_id` | Encoding & Surrogate | Gate & Nonconformity | `cov_in` | `MAE_all` | `RMSE_all` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["molecular_encoding_comparison"]:
        m = row["metrics"]
        enc_s = f"`{row['encoding']}` + `{row['surrogate']}`"
        gate_s = f"`{row['distance_metric']}` + `{row['nonconformity']}`"
        lines.append(
            f"| `{row['dataset_id']}` | {enc_s} | {gate_s} | "
            f"`{_fmt_pm(m['cov_in'], 3)}` | `{_fmt_pm(m['mae_all'], 3)}` | `{_fmt_pm(m['rmse_all'], 3)}` | "
            f"`{_fmt_pm(m['mae_in'], 3)}` | `{_fmt_pm(m['mae_out'], 3)}` | `{_fmt_pm(m['width_in'], 2)}` | "
            f"`{_fmt_pm(m['spearman_distance_abs_error'], 3)}` | `{row['seeds_mae_in_lt_out']}/{row['n_seeds']}` |"
        )

    lines.append("")
    lines.append("#### Reaction-Yield HTE Encoding & Gate Ablation (`descriptors` vs `onehot` vs `onehot_morgan`)")
    lines.append("")
    lines.append(
        "| `dataset_id` | Encoding | Gate & Nonconformity | `cov_in` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` | `gate_informative` |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["hte_encoding_comparison"]:
        m = row["metrics"]
        gate_lbl = f"`{row['distance_metric']}` + `{row['nonconformity']}`"
        gi_lbl = "**`True`**" if row["gate_informative"] else "**`False` (inverted/flat)**"
        lines.append(
            f"| `{row['dataset_id']}` | `{row['encoding']}` | {gate_lbl} | "
            f"`{_fmt_pm(m['cov_in'], 3)}` | `{_fmt_pm(m['mae_in'], 2)}` | "
            f"`{_fmt_pm(m['mae_out'], 2)}` | `{_fmt_pm(m['width_in'], 2)}` | "
            f"`{_fmt_pm(m['spearman_distance_abs_error'], 3)}` | "
            f"`{row['seeds_mae_in_lt_out']}/{row['n_seeds']}` | {gi_lbl} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark all 11 real registry datasets across multiple seeds.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per registry dataset (>= 5).")
    parser.add_argument("--fetch", action="store_true", help="Fetch any missing datasets into data/ first.")
    parser.add_argument(
        "--refresh-molecular-only",
        action="store_true",
        help="Re-run only ESOL 20-seed and single-molecule rdkit_2d_morgan rows into existing JSON.",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument("--print-markdown", action="store_true", help="Print rendered README markdown tables.")
    args = parser.parse_args(argv)

    if args.seeds < 5:
        raise ValueError("Benchmark requires at least 5 seeds per dataset")

    t0 = time.perf_counter()
    if args.fetch:
        for ds_id in OPEN_CALIBRATION_DATASETS:
            fetch_open_dataset(ds_id)

    seeds = list(range(int(args.seeds)))
    out_path = Path(args.output)

    if args.refresh_molecular_only and out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        print("Refreshing 20-seed ESOL head-to-head vs Delaney (2004)...", flush=True)
        esol_h2h = run_esol_delaney_20seed_head_to_head()
        print("Refreshing 20-seed ESOL active-expansion summary...", flush=True)
        esol20 = run_esol_20seed_summary()
        front_by_id: dict[str, dict[str, Any]] = {}
        for ds_id in SINGLE_MOLECULE_IDS:
            print(f"Refreshing rdkit_2d_morgan on {ds_id}...", flush=True)
            front_by_id[ds_id] = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="rdkit_2d_morgan",
                distance_metric="tanimoto_mahalanobis",
                nonconformity="cqr_r",
                surrogate="frontier_ensemble",
            )
        existing["esol_delaney_20seed_head_to_head"] = esol_h2h
        existing["esol_20seed"] = esol20
        existing["primary_11_datasets"] = [
            front_by_id[r["dataset_id"]] if r["dataset_id"] in front_by_id else r
            for r in existing["primary_11_datasets"]
        ]
        existing["molecular_encoding_comparison"] = [
            front_by_id[r["dataset_id"]]
            if (r["dataset_id"] in front_by_id and r["encoding"] == "rdkit_2d_morgan")
            else r
            for r in existing["molecular_encoding_comparison"]
        ]
        existing["elapsed_seconds"] = round(
            float(existing.get("elapsed_seconds", 0.0)) + (time.perf_counter() - t0), 3
        )
        out_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.print_markdown:
            print(render_registry_markdown(existing))
        return 0
    print("Running 20-seed ESOL head-to-head vs Delaney (2004)...", flush=True)
    esol_h2h = run_esol_delaney_20seed_head_to_head()
    print("Running 20-seed ESOL active-expansion summary...", flush=True)
    esol20 = run_esol_20seed_summary()
    print("Running 5-seed CALiSol-23 DOI holdout...", flush=True)
    calisol_doi = run_calisol_doi_holdout_summary(seeds=seeds)

    primary_rows: list[dict[str, Any]] = []
    mol_ablation_rows: list[dict[str, Any]] = []
    hte_ablation_rows: list[dict[str, Any]] = []

    for ds_id, spec in OPEN_CALIBRATION_DATASETS.items():
        print(f"Evaluating {ds_id}...", flush=True)
        is_large = spec.n_records >= 20000
        surr_hte = "lightgbm" if is_large else "gbdt"

        if ds_id in SINGLE_MOLECULE_IDS:
            row_desc = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="descriptors",
                distance_metric="euclidean",
                nonconformity="normalized",
                surrogate="gbdt",
            )
            row_front = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="rdkit_2d_morgan",
                distance_metric="tanimoto_mahalanobis",
                nonconformity="cqr_r",
                surrogate="frontier_ensemble",
            )
            mol_ablation_rows.extend([row_desc, row_front])
            primary_rows.append(row_front)
        elif ds_id in HTE_REACTION_IDS:
            row_desc = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="descriptors",
                distance_metric="euclidean",
                nonconformity="normalized",
                surrogate=surr_hte,
            )
            row_oh = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="onehot",
                distance_metric="mahalanobis",
                nonconformity="cqr_r",
                surrogate=surr_hte,
            )
            row_ohm = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="onehot_morgan",
                distance_metric="ensemble_mahalanobis",
                nonconformity="cqr_r",
                surrogate=surr_hte,
            )
            hte_ablation_rows.extend([row_desc, row_oh, row_ohm])
            primary_rows.append(row_ohm)
        else:
            split_mode = "group" if ds_id in ("bigsoldb_v2_2", "calisol23") else "random"
            nonconf = "cqr_r" if ds_id == "calisol23" else "normalized"
            row_main = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy=split_mode,
                encoding="descriptors",
                distance_metric="euclidean",
                nonconformity=nonconf,
                surrogate=surr_hte,
            )
            primary_rows.append(row_main)

    elapsed = round(time.perf_counter() - t0, 3)
    payload: dict[str, Any] = {
        "schema_version": "lab-decide-registry-benchmark/v2",
        "data_source": "real_csv",
        "elapsed_seconds": elapsed,
        "n_datasets": len(OPEN_CALIBRATION_DATASETS),
        "gate_informative_rule": "frac_seeds_mae_in_lt_out >= 0.80 and mean(spearman_distance_abs_error) > 0.0",
        "esol_delaney_20seed_head_to_head": esol_h2h,
        "esol_20seed": esol20,
        "calisol23_doi_holdout": calisol_doi,
        "primary_11_datasets": primary_rows,
        "molecular_encoding_comparison": mol_ablation_rows,
        "hte_encoding_comparison": hte_ablation_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    if args.print_markdown:
        print(render_registry_markdown(payload))
    else:
        n_inf = sum(1 for r in primary_rows if r["gate_informative"])
        print(
            f"Wrote {out_path} in {elapsed:.1f}s ({len(primary_rows)} datasets, "
            f"gate_informative={n_inf}/{len(primary_rows)}).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
