"""Multi-seed real-data conformal & applicability-domain benchmark across all 11 registry datasets.

Executes across real cached datasets in `data/` (failing closed if any dataset is uncached unless
`--fetch` is supplied; never permits synthetic fixtures):
1. 20-seed ESOL (`esol_delaney`) random & balanced scaffold splits + active boundary expansion.
2. 5-seed CALiSol-23 (`calisol23`) publication-DOI holdout split (`evaluate_calisol_conformal`).
3. >=5-seed evaluation across all 11 `OpenCalibrationDataset` benchmarks on both random and
   domain-group (`--split group`) splits.
4. Side-by-side HTE reaction encoding & gating comparison on all 4 reaction-yield datasets:
   `descriptors` (Euclidean kNN) vs `onehot` (Mahalanobis kNN + CQR-r) vs
   `onehot_morgan` (ensemble-Mahalanobis kNN + CQR-r).

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

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_decide.calisol import evaluate_calisol_conformal  # noqa: E402
from lab_decide.cli import run_esol_benchmark  # noqa: E402
from lab_decide.conformal import (  # noqa: E402
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


def _clean_float(v: float | None, ndigits: int = 6) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), ndigits)


def _mean_sd(vals: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
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

        c_in, w_in, e_in = summarize(y_te, res["lo"], res["hi"], inside, pred=res["pred"])
        c_out, w_out, e_out = summarize(y_te, res["lo"], res["hi"], ~inside, pred=res["pred"])
        c_all, w_all, e_all = summarize(
            y_te, res["lo"], res["hi"], np.ones(len(y_te), dtype=bool), pred=res["pred"]
        )
        wink_in = winkler_interval_score(y_te, res["lo"], res["hi"], alpha=alpha, mask=inside)
        rho = spearman_rank_correlation(res["distance"], np.abs(y_te - res["pred"]))

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
    spearman_stats = _mean_sd([r["spearman_distance_abs_error"] for r in per_seed if r["spearman_distance_abs_error"] is not None])
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
        "width_in",
        "width_out",
        "width_all",
        "winkler_in",
        "spearman_distance_abs_error",
    )
    agg = {
        mk: _mean_sd([r[mk] for r in per_seed if r[mk] is not None])
        for mk in metric_keys
    }

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


def run_esol_20seed_summary() -> dict[str, Any]:
    bundle = load_dataset(offline=True, allow_synthetic=False)
    assert bundle.data_source == "real_csv"

    out_by_split: dict[str, Any] = {}
    for split_mode in ("random", "scaffold"):
        cov_in_0, mae_in_0, mae_out_0, w_in_0, w_in_last, cov_in_last, c_top, u_top = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        spearman_0 = []
        max_frac_err = 0.0
        wins = 0
        for seed in range(20):
            summary, _, _ = run_esol_benchmark(bundle, split_strategy=split_mode, seed=seed)
            tr_blk = summary["mondrian_trust_region"]
            c_i = float(tr_blk["inside"]["coverage"])
            e_i = float(tr_blk["inside"]["mae"])
            e_o = float(tr_blk["outside"]["mae"])
            w_i = float(tr_blk["inside"]["width"])
            rho = float(tr_blk["spearman_distance_abs_error"])
            cov_in_0.append(c_i)
            mae_in_0.append(e_i)
            mae_out_0.append(e_o)
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
            "data_source": "real_csv",
            "max_achieved_fraction_abs_error": round(max_frac_err, 6),
            "cov_in_round0": _mean_sd(cov_in_0),
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
    lines.append(
        f"- **20-Seed ESOL Summary (`data_source=real_csv`)**: Random split inside coverage "
        f"`{_fmt_pm(r20['cov_in_round0'], 3)}` (`MAE_in {_fmt_pm(r20['mae_in_round0'], 3)}` vs "
        f"`MAE_out {_fmt_pm(r20['mae_out_round0'], 3)}`, `width_in {_fmt_pm(r20['width_in_round0'], 3)} -> "
        f"{_fmt_pm(r20['width_in_final_round'], 3)}` after expansion); balanced Bemis-Murcko scaffold split "
        f"inside coverage `{_fmt_pm(s20['cov_in_round0'], 3)}` (`MAE_in {_fmt_pm(s20['mae_in_round0'], 3)}` vs "
        f"`MAE_out {_fmt_pm(s20['mae_out_round0'], 3)}`, max split-fraction deviation "
        f"`{100.0 * float(s20['max_achieved_fraction_abs_error']):.2f}` percentage points across all 20 seeds)."
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
    lines.append(
        "| `dataset_id` | Split | Encoding / Gate | Seeds (`n`) | `cov_in` | `cov_out` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` | `gate_informative` |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["primary_11_datasets"]:
        m = row["metrics"]
        enc_lbl = f"`{row['encoding']}` / `{row['distance_metric']}`"
        gi_lbl = "**`True`**" if row["gate_informative"] else "**`False` (uninformative gate)**"
        lines.append(
            f"| `{row['dataset_id']}` | `{row['split_strategy']}` | {enc_lbl} | "
            f"`{row['n_seeds']}` (`{row['n_clean']}`) | "
            f"`{_fmt_pm(m['cov_in'], 3)}` | `{_fmt_pm(m['cov_out'], 3)}` | "
            f"`{_fmt_pm(m['mae_in'], 2)}` | `{_fmt_pm(m['mae_out'], 2)}` | "
            f"`{_fmt_pm(m['width_in'], 2)}` | `{_fmt_pm(m['spearman_distance_abs_error'], 3)}` | "
            f"`{row['seeds_mae_in_lt_out']}/{row['n_seeds']}` | {gi_lbl} |"
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
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument("--print-markdown", action="store_true", help="Print rendered README markdown tables.")
    args = parser.parse_args(argv)

    if args.seeds < 5:
        raise ValueError("Issue #34 requires at least 5 seeds per dataset")

    t0 = time.perf_counter()
    if args.fetch:
        for ds_id in OPEN_CALIBRATION_DATASETS:
            fetch_open_dataset(ds_id)

    seeds = list(range(int(args.seeds)))
    esol20 = run_esol_20seed_summary()
    calisol_doi = run_calisol_doi_holdout_summary(seeds=seeds)

    hte_ids = (
        "buchwald_hartwig_doyle",
        "suzuki_miyaura_perera",
        "direct_arylation_shields",
        "bh_hte_ood_2026",
    )

    primary_rows: list[dict[str, Any]] = []
    hte_ablation_rows: list[dict[str, Any]] = []

    for ds_id, spec in OPEN_CALIBRATION_DATASETS.items():
        # Use fast exact LightGBM / HistGBDT surrogate only on the two 27k-125k datasets so all 11
        # full datasets finish comfortably within ~3-5 minutes on CPU without synthetic shortcuts;
        # cap row count ONLY if >25k? Wait: with lightgbm, 125k rows takes ~1.5s per seed (zero row cap needed!).
        is_large = spec.n_records >= 20000
        surr = "lightgbm" if is_large else "gbdt"

        if ds_id in hte_ids:
            row_desc = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="descriptors",
                distance_metric="euclidean",
                nonconformity="normalized",
                surrogate=surr,
            )
            row_oh = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="onehot",
                distance_metric="mahalanobis",
                nonconformity="cqr_r",
                surrogate=surr,
            )
            row_ohm = _evaluate_bundle_across_seeds(
                ds_id,
                seeds=seeds,
                split_strategy="random",
                encoding="onehot_morgan",
                distance_metric="ensemble_mahalanobis",
                nonconformity="cqr_r",
                surrogate=surr,
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
                surrogate=surr,
            )
            primary_rows.append(row_main)

    elapsed = round(time.perf_counter() - t0, 3)
    payload: dict[str, Any] = {
        "schema_version": "lab-decide-registry-benchmark/v1",
        "data_source": "real_csv",
        "elapsed_seconds": elapsed,
        "n_datasets": len(OPEN_CALIBRATION_DATASETS),
        "gate_informative_rule": "frac_seeds_mae_in_lt_out >= 0.80 and mean(spearman_distance_abs_error) > 0.0",
        "esol_20seed": esol20,
        "calisol23_doi_holdout": calisol_doi,
        "primary_11_datasets": primary_rows,
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
            f"gate_informative={n_inf}/{len(primary_rows)})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
