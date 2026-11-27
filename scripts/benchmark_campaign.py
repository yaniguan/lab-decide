"""Sequential closed-loop active optimization campaign benchmark (CPU only).

Compares 8 sequential batch acquisition strategies over >=10 seeds on real cached datasets:
- `esol_delaney` (aqueous solubility maximization, `log10(mol/L)`),
- `direct_arylation_shields` (Pd C-H arylation yield maximization, `%`),
- `buchwald_hartwig_doyle` (Buchwald-Hartwig C-N coupling yield maximization, `%`).

Methods evaluated each round (`pick batch -> reveal y -> refit & recalibrate`):
- `random`
- `greedy` (GBDT point prediction)
- `unconstrained_conformal_ucb`
- `tr_conformal_ucb` (`TrustRegionAdvisor`, `conformal_ucb`)
- `tr_conformal_ei` (`TrustRegionAdvisor`, `conformal_ei`)
- `gp_ei` (`GaussianProcessRegressor` Expected Improvement)
- `lightgbm_cqr_ucb` (LightGBM + CQR-r unconstrained UCB)
- `lightgbm_cqr_tr_ucb` (LightGBM + CQR-r trust-region UCB)

Writes `results/campaign_benchmark.json` and exposes `render_campaign_markdown(payload)` so the
`README.md` sequential campaign table and winner statement are generated directly from JSON.
"""
from __future__ import annotations

import argparse
import json
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

from lab_decide.acquisition import VALID_CAMPAIGN_METHODS, run_sequential_bo_campaign  # noqa: E402
from lab_decide.data import partition_splits  # noqa: E402
from lab_decide.datasets import (  # noqa: E402
    default_dataset_cache_path,
    fetch_open_dataset,
    get_open_dataset,
    load_open_dataset,
)

DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "campaign_benchmark.json"

CAMPAIGN_DATASETS: tuple[tuple[str, str, str, str, tuple[float, float, float, float]], ...] = (
    ("esol_delaney", "descriptors", "euclidean", "gbdt", (0.15, 0.55, 0.15, 0.15)),
    ("direct_arylation_shields", "onehot_morgan", "mahalanobis", "hist_gbdt", (0.15, 0.55, 0.15, 0.15)),
    ("buchwald_hartwig_doyle", "onehot_morgan", "mahalanobis", "hist_gbdt", (0.10, 0.60, 0.15, 0.15)),
)


def _mean_sd(vals: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return {"mean": round(float(np.mean(arr)), 6), "sd": round(sd, 6)}


def _fmt_pm(stat: Mapping[str, Any], prec: int = 3) -> str:
    return f"{float(stat['mean']):.{prec}f} +- {float(stat['sd']):.{prec}f}"


def benchmark_dataset_campaigns(
    dataset_id: str,
    *,
    encoding: str,
    distance_metric: str,
    surrogate: str = "gbdt",
    fracs: tuple[float, float, float, float],
    seeds: Sequence[int],
    rounds: int = 6,
    batch_size: int = 8,
    methods: Sequence[str] = VALID_CAMPAIGN_METHODS,
) -> dict[str, Any]:
    spec = get_open_dataset(dataset_id)
    if spec.objective_direction not in ("maximize", "minimize"):
        raise ValueError(f"Dataset {dataset_id!r} has objective_direction={spec.objective_direction!r}")
    maximize = spec.objective_direction == "maximize"

    cache_path = default_dataset_cache_path(dataset_id)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Real dataset {dataset_id!r} is not cached at {cache_path}; run with --fetch first."
        )

    bundle = load_open_dataset(
        dataset_id,
        path=cache_path,
        offline=True,
        encoding=encoding,
        data_source="real_csv",
    )
    assert bundle.data_source == "real_csv"

    method_summaries: dict[str, Any] = {}
    for method in methods:
        seed_histories: list[list[dict[str, Any]]] = []
        for seed in seeds:
            splits = partition_splits(
                len(bundle.y),
                strategy="random",
                seed=int(seed),
                fracs=fracs,
            )
            hist = run_sequential_bo_campaign(
                bundle.X,
                bundle.y,
                train_idx=splits.train_idx,
                pool_idx=splits.pool_idx,
                cal_idx=splits.cal_idx,
                method=method,
                rounds=rounds,
                batch_size=batch_size,
                maximize=maximize,
                seed=int(seed),
                distance_metric=distance_metric,
                surrogate=surrogate,
                domain_bounds=spec.domain_bounds,
                feature_names=bundle.feature_names,
            )
            seed_histories.append(hist)

        rounds_agg: list[dict[str, Any]] = []
        for r_idx in range(rounds):
            r_rows = [sh[r_idx] for sh in seed_histories if len(sh) > r_idx]
            rounds_agg.append(
                {
                    "round": r_idx + 1,
                    "n_acquired_total": int(r_rows[0]["n_acquired_total"]),
                    "best_so_far": _mean_sd([row["best_so_far"] for row in r_rows]),
                    "simple_regret": _mean_sd([row["simple_regret"] for row in r_rows]),
                    "round_ood_fraction": _mean_sd([row["round_ood_fraction"] for row in r_rows]),
                    "cumulative_ood_fraction": _mean_sd([row["cumulative_ood_fraction"] for row in r_rows]),
                }
            )

        final_blk = rounds_agg[-1]
        method_summaries[method] = {
            "method": method,
            "final_best_so_far": final_blk["best_so_far"],
            "final_simple_regret": final_blk["simple_regret"],
            "final_cumulative_ood_fraction": final_blk["cumulative_ood_fraction"],
            "rounds": rounds_agg,
        }

    # Determine lowest-regret overall method and lowest-regret zero-OOD trust-region method honestly
    ordered_by_regret = sorted(
        method_summaries.values(),
        key=lambda m: (float(m["final_simple_regret"]["mean"]), float(m["final_cumulative_ood_fraction"]["mean"])),
    )
    overall_best = ordered_by_regret[0]
    n_s = max(1, len(seeds))
    best_se = float(overall_best["final_simple_regret"]["sd"]) / math.sqrt(float(n_s))
    tied_within_se = [
        m["method"]
        for m in ordered_by_regret
        if float(m["final_simple_regret"]["mean"]) <= float(overall_best["final_simple_regret"]["mean"]) + best_se + 1e-9
    ]

    safe_methods = [
        m for m in ordered_by_regret if float(m["final_cumulative_ood_fraction"]["mean"]) <= 0.02
    ]
    best_safe = safe_methods[0]["method"] if safe_methods else None

    return {
        "dataset_id": dataset_id,
        "target_unit": spec.target_unit,
        "objective_direction": spec.objective_direction,
        "data_source": "real_csv",
        "encoding": encoding,
        "distance_metric": distance_metric,
        "split_fractions": list(fracs),
        "n_clean": int(len(bundle.y)),
        "n_seeds": len(seeds),
        "rounds": rounds,
        "batch_size": batch_size,
        "lowest_mean_regret_method": overall_best["method"],
        "methods_within_1se_of_best_regret": tied_within_se,
        "lowest_regret_trust_region_safe_method": best_safe,
        "methods": method_summaries,
    }


import math  # noqa: E402


def render_campaign_markdown(payload: Mapping[str, Any]) -> str:
    """Render deterministic Markdown summary tables from `results/campaign_benchmark.json`."""
    lines: list[str] = []
    lines.append("### Sequential Closed-Loop Active Optimization Benchmark (`results/campaign_benchmark.json`)")
    lines.append("")
    lines.append(
        f"CPU sequential campaign evaluation (`{payload['n_seeds']}` seeds, `{payload['rounds']}` rounds x "
        f"`{payload['batch_size']}` candidates/round, refitting and recalibrating after every round on `real_csv` data):"
    )
    lines.append("")
    for ds_blk in payload["datasets"]:
        ds_id = ds_blk["dataset_id"]
        unit = ds_blk["target_unit"]
        prec = 3 if "log" in unit else 2
        best_m = ds_blk["lowest_mean_regret_method"]
        tied = ds_blk["methods_within_1se_of_best_regret"]
        safe_m = ds_blk["lowest_regret_trust_region_safe_method"]
        if len(tied) == 1:
            verdict = f"Overall simple-regret winner: **`{best_m}`**; lowest-regret zero-OOD trust-region method: **`{safe_m}`**."
        else:
            tied_str = ", ".join(f"`{t}`" for t in tied)
            verdict = (
                f"Lowest mean simple regret: **`{best_m}`** (statistically tied within 1 SE with {tied_str}); "
                f"lowest-regret zero-OOD trust-region method: **`{safe_m}`**."
            )
        lines.append(f"#### `{ds_id}` (`{unit}`, `{ds_blk['objective_direction']}`)")
        lines.append("")
        lines.append(f"- {verdict}")
        lines.append("")
        lines.append(
            "| Method | Round 1 Best | Final Best-So-Far | Final Simple Regret | Cumulative OOD Pick Frac |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for m_name in VALID_CAMPAIGN_METHODS:
            if m_name not in ds_blk["methods"]:
                continue
            m_blk = ds_blk["methods"][m_name]
            r1_best = m_blk["rounds"][0]["best_so_far"]
            fin_best = m_blk["final_best_so_far"]
            fin_reg = m_blk["final_simple_regret"]
            fin_ood = m_blk["final_cumulative_ood_fraction"]
            lines.append(
                f"| `{m_name}` | `{_fmt_pm(r1_best, prec)}` | `{_fmt_pm(fin_best, prec)}` | "
                f"`{_fmt_pm(fin_reg, prec)}` | `{_fmt_pm(fin_ood, 3)}` |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 10-seed sequential BO campaigns across real datasets.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds (>= 10).")
    parser.add_argument("--rounds", type=int, default=6, help="Sequential rounds per seed.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch acquisitions per round.")
    parser.add_argument("--fetch", action="store_true", help="Fetch missing datasets into data/ first.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument("--print-markdown", action="store_true", help="Print rendered README markdown.")
    args = parser.parse_args(argv)

    if args.seeds < 10:
        raise ValueError("Issue #33 requires at least 10 seeds")

    t0 = time.perf_counter()
    if args.fetch:
        for ds_id, _, _, _, _ in CAMPAIGN_DATASETS:
            fetch_open_dataset(ds_id)

    seeds = list(range(int(args.seeds)))
    ds_results = []
    for ds_id, enc, d_metric, surr, fracs in CAMPAIGN_DATASETS:
        ds_res = benchmark_dataset_campaigns(
            ds_id,
            encoding=enc,
            distance_metric=d_metric,
            surrogate=surr,
            fracs=fracs,
            seeds=seeds,
            rounds=args.rounds,
            batch_size=args.batch_size,
        )
        ds_results.append(ds_res)

    elapsed = round(time.perf_counter() - t0, 3)
    payload: dict[str, Any] = {
        "schema_version": "lab-decide-campaign-benchmark/v1",
        "data_source": "real_csv",
        "elapsed_seconds": elapsed,
        "n_seeds": int(args.seeds),
        "rounds": int(args.rounds),
        "batch_size": int(args.batch_size),
        "methods": list(VALID_CAMPAIGN_METHODS),
        "datasets": ds_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    if args.print_markdown:
        print(render_campaign_markdown(payload))
    else:
        print(f"Wrote {out_path} in {elapsed:.1f}s across {len(ds_results)} datasets x {args.seeds} seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
