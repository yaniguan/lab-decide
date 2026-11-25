"""Command-line interface and ESOL / CALiSol-23 conformal benchmark runner for lab-decide."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from lab_decide.acquisition import (
    TrustRegionAdvisor,
    benchmark_trust_region_bo,
    run_expansion_campaign,
)
from lab_decide.calisol import evaluate_calisol_conformal
from lab_decide.chemistry import ESOL_FEATURES, ESOL_TARGET, featurize_smiles_batch
from lab_decide.conformal import ConformalTrustModel, fit_and_calibrate, summarize
from lab_decide.data import (
    DatasetBundle,
    load_dataset,
    partition_splits,
)
from lab_decide.schema import (
    DecisionEnvelope,
    build_decision_envelopes,
    evaluate_epistemic_probes,
    export_decisions_jsonl,
)


def _fmt_cell(val: float, width: int = 8, prec: int = 3) -> str:
    if val is None or not math.isfinite(float(val)):
        return f"{'n/a':>{width}}"
    return f"{float(val):>{width}.{prec}f}"


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    return obj


def run_esol_benchmark(
    bundle: DatasetBundle,
    *,
    split_strategy: str = "random",
    alpha: float = 0.10,
    k: int = 5,
    trust_pct: float = 80.0,
    rounds: int = 3,
    per_round: int = 20,
    seed: int = 0,
    nonconformity: str = "normalized",
    domain_bounds: tuple[float, float] | None = (-15.0, 1.75),
) -> tuple[dict[str, Any], ConformalTrustModel, np.ndarray]:
    """Run full 4-way split Mondrian conformal + frozen-test active expansion + constrained BO benchmark."""
    X, y = bundle.X, bundle.y
    splits = partition_splits(
        len(y),
        smiles=bundle.smiles,
        strategy=split_strategy,
        seed=seed,
        fracs=(0.50, 0.20, 0.15, 0.15),
    )

    advisor0 = fit_and_calibrate(
        X[splits.train_idx],
        y[splits.train_idx],
        X[splits.cal_idx],
        y[splits.cal_idx],
        alpha=alpha,
        k=k,
        trust_pct=trust_pct,
        seed=seed,
        loss="huber",
        nonconformity=nonconformity,
        distance_weighting="importance",
        domain_bounds=domain_bounds,
        feature_names=bundle.feature_names,
    )

    te_idx = splits.test_idx
    yt = y[te_idx]

    # Marginal split conformal summary on frozen test set
    res_marg = advisor0.predict(X[te_idx], use_mondrian=False)
    all_mask = np.ones(len(te_idx), dtype=bool)
    cov_marg, width_marg, mae_marg = summarize(
        yt, res_marg["lo"], res_marg["hi"], all_mask, pred=res_marg["pred"]
    )

    # Selective Mondrian conformal trust-region summary on frozen test set
    res_mond = advisor0.predict(X[te_idx], use_mondrian=True)
    inside = np.asarray(res_mond["trusted"], dtype=bool)
    c_in, w_in, e_in = summarize(yt, res_mond["lo"], res_mond["hi"], inside, pred=res_mond["pred"])
    c_out, w_out, e_out = summarize(yt, res_mond["lo"], res_mond["hi"], ~inside, pred=res_mond["pred"])

    history, final_advisor = run_expansion_campaign(
        X,
        y,
        train_idx=splits.train_idx,
        pool_idx=splits.pool_idx,
        cal_idx=splits.cal_idx,
        test_idx=splits.test_idx,
        rounds=rounds,
        per_round=per_round,
        cal_update_frac=0.20,
        alpha=alpha,
        k=k,
        trust_pct=trust_pct,
        seed=seed,
        loss="huber",
        nonconformity=nonconformity,
        distance_weighting="importance",
        domain_bounds=domain_bounds,
        feature_names=bundle.feature_names,
    )

    bo_summary = benchmark_trust_region_bo(
        advisor0,
        X[splits.pool_idx],
        y[splits.pool_idx],
        batch_size=min(10, len(splits.pool_idx)),
        maximize=True,
    )

    target_pct = int(round((1.0 - alpha) * 100))
    summary: dict[str, Any] = {
        "rows_clean": len(y),
        "rows_raw": bundle.raw_row_count,
        "deduplicated_smiles": bundle.deduplicated_count,
        "dropped_nonfinite": bundle.dropped_nonfinite_count,
        "split_strategy": split_strategy,
        "splits": {
            "train": int(splits.train_idx.size),
            "pool": int(splits.pool_idx.size),
            "calibration": int(splits.cal_idx.size),
            "test": int(splits.test_idx.size),
        },
        "features": bundle.feature_names,
        "alpha": alpha,
        "target_coverage_pct": target_pct,
        "k": k,
        "trust_pct": trust_pct,
        "nonconformity": nonconformity,
        "marginal_conformal": {
            "q_marginal": advisor0.q_marginal,
            "coverage": cov_marg,
            "mean_width": width_marg,
            "mae": mae_marg,
        },
        "mondrian_trust_region": {
            "threshold": advisor0.threshold,
            "q_in": advisor0.q_in,
            "answered_fraction": float(np.mean(inside)),
            "abstained_fraction": float(1.0 - np.mean(inside)),
            "inside": {"n": int(np.sum(inside)), "coverage": c_in, "width": w_in, "mae": e_in},
            "outside": {"n": int(np.sum(~inside)), "coverage": c_out, "width": w_out, "mae": e_out},
        },
        "expansion_rounds": history,
        "constrained_bo": bo_summary,
    }
    return summary, final_advisor, te_idx


def format_esol_report(summary: Mapping[str, Any]) -> str:
    sp = summary["splits"]
    mc = summary["marginal_conformal"]
    tr = summary["mondrian_trust_region"]
    bo = summary["constrained_bo"]
    target_pct = summary["target_coverage_pct"]
    k = summary["k"]
    tpct = int(round(float(summary["trust_pct"])))

    lines: list[str] = []
    lines.append(
        f"ESOL clean rows: {summary['rows_clean']} (deduped {summary['deduplicated_smiles']} duplicate SMILES) | "
        f"train {sp['train']} / pool {sp['pool']} / calibration {sp['calibration']} / test {sp['test']}"
    )
    lines.append(f"Features: {', '.join(summary['features'])}")
    lines.append("")
    lines.append(f"== Split conformal (target {target_pct}%, domain-clipped) ==")
    lines.append(f"Marginal half-width q_all: {_fmt_cell(mc['q_marginal'], 5, 3).strip()} log units | Mondrian inside q_in: {_fmt_cell(tr['q_in'], 5, 3).strip()} log units")
    lines.append(
        f"Test coverage: {_fmt_cell(mc['coverage'], 5, 3).strip()} | "
        f"mean interval width: {_fmt_cell(mc['mean_width'], 5, 3).strip()} | "
        f"MAE: {_fmt_cell(mc['mae'], 5, 3).strip()}"
    )
    lines.append("")
    lines.append(
        f"== Mondrian trust region (importance-weighted kNN k={k}, conformal {tpct}th-pct threshold = {_fmt_cell(tr['threshold'], 5, 3).strip()}) =="
    )
    lines.append(f"{'region':<8}{'n':>5}{'coverage':>10}{'width':>8}{'MAE':>8}")
    for name in ("inside", "outside"):
        blk = tr[name]
        lines.append(
            f"{name:<8}{blk['n']:>5}"
            f"{_fmt_cell(blk['coverage'], 10)}"
            f"{_fmt_cell(blk['width'], 8)}"
            f"{_fmt_cell(blk['mae'], 8)}"
        )
    lines.append(
        f"API answers {tr['answered_fraction']:.3f} of frozen test points, "
        f"abstains on {tr['abstained_fraction']:.3f}"
    )
    lines.append("")
    lines.append(
        "== Expansion on frozen test set (diverse boundary acquisition from separate pool + 20% cal update) =="
    )
    lines.append(
        f"{'round':<6}{'train':>6}{'pool':>6}{'test':>6}{'n_in':>6}{'n_out':>6}"
        f"{'answered':>10}{'cov_in':>8}{'cov_out':>9}{'cov_all':>9}"
        f"{'MAE_in':>8}{'MAE_out':>9}{'width_in':>9}"
    )
    for row in summary["expansion_rounds"]:
        lines.append(
            f"{row['round']:<6}{row['n_train']:>6}{row['n_pool']:>6}{row['n_test']:>6}"
            f"{row['n_in']:>6}{row['n_out']:>6}"
            f"{_fmt_cell(row['answered'], 10)}"
            f"{_fmt_cell(row['cov_in'], 8)}"
            f"{_fmt_cell(row['cov_out'], 9)}"
            f"{_fmt_cell(row['cov_all'], 9)}"
            f"{_fmt_cell(row['mae_in'], 8)}"
            f"{_fmt_cell(row['mae_out'], 9)}"
            f"{_fmt_cell(row['width_in'], 9)}"
        )
    lines.append("")
    lines.append(
        f"== Trust-region constrained BO (conformal UCB top-10 on pool) ==\n"
        f"Constrained top logS: {_fmt_cell(bo['constrained_top_y'], 5, 3).strip()} "
        f"(pred MAE {_fmt_cell(bo['constrained_pred_mae'], 5, 3).strip()}) | "
        f"Unconstrained greedy pred MAE: {_fmt_cell(bo['unconstrained_pred_mae'], 5, 3).strip()} "
        f"(OOD frac {_fmt_cell(bo['unconstrained_ood_fraction'], 5, 3).strip()})"
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-decide",
        description="Conformal trust-region experiment advisor and constrained BO planner.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to custom tabular CSV dataset.")
    parser.add_argument(
        "--dataset",
        choices=("esol", "calisol23", "synthetic"),
        default="esol",
        help="Benchmark dataset adapter to execute.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature column names (defaults to 6 ESOL descriptors).",
    )
    parser.add_argument("--target", type=str, default=ESOL_TARGET, help="Target column name.")
    parser.add_argument(
        "--split",
        choices=("random", "scaffold"),
        default="random",
        help="Data split protocol ('random' or Bemis-Murcko 'scaffold').",
    )
    parser.add_argument(
        "--nonconformity",
        choices=("normalized", "cqr", "cqr_r", "absolute"),
        default="normalized",
        help="Conformal nonconformity score type ('normalized', 'cqr', 'cqr_r', or 'absolute').",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha in (0, 1).")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors for trust distance.")
    parser.add_argument("--trust-pct", type=float, default=80.0, help="Calibration distance percentile in (0, 100].")
    parser.add_argument("--rounds", type=int, default=3, help="Number of active expansion rounds.")
    parser.add_argument("--per-round", type=int, default=20, help="Candidate pool acquisitions per round.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed.")
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Comma-separated SMILES strings to score via RDKit 6D descriptor bridge.",
    )
    parser.add_argument(
        "--probes-jsonl",
        type=str,
        default=None,
        help="Evaluate EpistemicProbeResult JSONL from lab-ask-why and emit DecisionEnvelopes.",
    )
    parser.add_argument("--offline", action="store_true", help="Disable network downloads; use cached/synthetic data.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument("--json", action="store_true", help="Shorthand alias for --format json.")
    parser.add_argument(
        "--export-decisions",
        type=str,
        default=None,
        help="Export per-candidate DecisionEnvelope JSONL records to this path.",
    )
    parser.add_argument(
        "--export-schema",
        type=str,
        default=None,
        help="Write DecisionEnvelope JSON Schema to this path and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_fmt = "json" if args.json else args.format

    if not (0.0 < args.alpha < 1.0):
        print("Error: --alpha must lie strictly between 0 and 1", file=sys.stderr)
        return 2
    if not (0.0 < args.trust_pct <= 100.0) or args.k < 1 or args.rounds < 0 or args.per_round < 1:
        print("Error: invalid --trust-pct, --k, --rounds, or --per-round parameter", file=sys.stderr)
        return 2

    try:
        if args.export_schema:
            schema = DecisionEnvelope.json_schema()
            with open(args.export_schema, mode="w", encoding="utf-8") as fh:
                json.dump(schema, fh, indent=2)
            if out_fmt == "json":
                print(json.dumps(schema, indent=2))
            else:
                print(f"Wrote DecisionEnvelope JSON Schema to {args.export_schema}")
            return 0

        if args.dataset == "calisol23":
            cal_res = evaluate_calisol_conformal(
                csv_path=args.csv,
                alpha=args.alpha,
                trust_pct=args.trust_pct,
                seed=args.seed,
                nonconformity=args.nonconformity,
            )
            clean_cal = _json_clean(cal_res)
            if out_fmt == "json":
                print(json.dumps(clean_cal, indent=2))
            else:
                print(
                    f"CALiSol-23 DOI-split conformal ({clean_cal['source']}): "
                    f"train={clean_cal['n_train']} cal={clean_cal['n_cal']} test={clean_cal['n_test']} | "
                    f"answered={clean_cal['answered_fraction']:.3f} cov_in={clean_cal['cov_in']:.3f} "
                    f"MAE_in={clean_cal['mae_in']:.3f} vs MAE_out={clean_cal['mae_out']:.3f} | "
                    f"kernel-gate agreement={clean_cal['knn_vs_kernel_support_agreement']:.3f}"
                )
            return 0

        features = (
            [f.strip() for f in args.features.split(",") if f.strip()]
            if args.features
            else ESOL_FEATURES
        )

        if args.dataset == "synthetic":
            from lab_decide.data import make_synthetic_solubility_dataframe, validate_and_clean_dataframe

            bundle = validate_and_clean_dataframe(
                make_synthetic_solubility_dataframe(n=240, seed=args.seed),
                features=features,
                target=args.target,
            )
        else:
            bundle = load_dataset(
                csv_path=args.csv,
                features=features,
                target=args.target,
                offline=args.offline,
            )

        summary, final_advisor, te_idx = run_esol_benchmark(
            bundle,
            split_strategy=args.split,
            alpha=args.alpha,
            k=args.k,
            trust_pct=args.trust_pct,
            rounds=args.rounds,
            per_round=args.per_round,
            seed=args.seed,
            nonconformity=args.nonconformity,
        )

        if args.smiles:
            smi_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
            X_smi = featurize_smiles_batch(smi_list, feature_names=bundle.feature_names)
            envs = build_decision_envelopes(
                final_advisor,
                X_smi,
                candidate_ids=[f"smi-{i}" for i in range(len(smi_list))],
                smiles=smi_list,
            )
            if args.export_decisions:
                export_decisions_jsonl(envs, args.export_decisions)
            if out_fmt == "json":
                print(json.dumps([e.to_dict() for e in envs], indent=2))
            else:
                for e in envs:
                    print(
                        f"{e.candidate_id} ({e.smiles}): pred={e.prediction:.3f} "
                        f"[{e.interval_lo:.3f}, {e.interval_hi:.3f}] d={e.knn_distance:.3f} -> {e.decision}"
                    )
            return 0

        if args.probes_jsonl:
            with open(args.probes_jsonl, mode="r", encoding="utf-8") as fh:
                probes = [json.loads(line) for line in fh if line.strip()]
            envs = evaluate_epistemic_probes(final_advisor, probes)
            if args.export_decisions:
                export_decisions_jsonl(envs, args.export_decisions)
            print(json.dumps([e.to_dict() for e in envs], indent=2))
            return 0

        if args.export_decisions:
            te_envs = build_decision_envelopes(
                final_advisor,
                bundle.X[te_idx],
                candidate_ids=[bundle.compound_ids[int(i)] for i in te_idx],
                smiles=[bundle.smiles[int(i)] for i in te_idx],
            )
            export_decisions_jsonl(te_envs, args.export_decisions)

        if out_fmt == "json":
            print(json.dumps(_json_clean(summary), indent=2))
        else:
            print(format_esol_report(summary))
        return 0

    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
