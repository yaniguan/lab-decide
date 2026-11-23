"""Comprehensive offline unit & regression test suite for lab-decide (Issues #1..#22)."""
from __future__ import annotations

import io
import json
import math
import pickle
import warnings
from pathlib import Path
from unittest import mock

import jsonschema
import numpy as np
import pandas as pd
import pytest

from lab_decide import (
    CALISOL_TEST_DOIS,
    CALISOL_VAL_DOIS,
    DECISION_ENVELOPE_JSON_SCHEMA,
    ESOL_FEATURES,
    ESOL_TARGET,
    HAS_RDKIT,
    ConformalTrustModel,
    DecisionEnvelope,
    TrustRegionAdvisor,
    assign_calisol_doi_split,
    benchmark_trust_region_bo,
    build_decision_envelopes,
    compute_scaffold_groups,
    conformal_quantile,
    conformal_threshold,
    download_file_atomic,
    ensure_esol_csv,
    evaluate_calisol_conformal,
    evaluate_epistemic_probes,
    export_decisions_jsonl,
    featurize_smiles_batch,
    fit_and_calibrate,
    formulation_kernel_support,
    load_decisions_jsonl,
    make_synthetic_solubility_dataframe,
    partition_splits,
    run_expansion_campaign,
    select_diverse_boundary_batch,
    smiles_to_descriptors,
    summarize,
    validate_and_clean_dataframe,
    weighted_conformal_quantile,
)
from lab_decide.cli import format_esol_report, main as cli_main, run_esol_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_issue_01_disjoint_4way_split_and_frozen_test_set_across_expansion() -> None:
    df = make_synthetic_solubility_dataframe(n=200, seed=0)
    bundle = validate_and_clean_dataframe(df)
    splits = partition_splits(len(bundle.y), strategy="random", seed=0)

    sets = [
        set(splits.train_idx.tolist()),
        set(splits.pool_idx.tolist()),
        set(splits.cal_idx.tolist()),
        set(splits.test_idx.tolist()),
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[i].isdisjoint(sets[j])
    assert set().union(*sets) == set(range(len(bundle.y)))

    history, _ = run_expansion_campaign(
        bundle.X,
        bundle.y,
        train_idx=splits.train_idx,
        pool_idx=splits.pool_idx,
        cal_idx=splits.cal_idx,
        test_idx=splits.test_idx,
        rounds=2,
        per_round=10,
        seed=0,
    )
    assert len(history) >= 2
    n_test_0 = history[0]["n_test"]
    for row in history:
        assert row["n_test"] == n_test_0, "Held-out test_idx must remain untouched across rounds"
        assert row["n_in"] + row["n_out"] == n_test_0
    assert history[-1]["n_train"] > history[0]["n_train"]
    assert history[-1]["n_pool"] < history[0]["n_pool"]


def test_issue_02_mondrian_selective_conformal_coverage_monte_carlo() -> None:
    # Heteroscedastic OOD regime: inside x in [-1, 1]^2 has low noise (0.2), OOD x in [4, 6]^2 has huge noise (3.0)
    cov_in_trials: list[float] = []
    width_in_mond: list[float] = []
    width_in_marg: list[float] = []

    for trial in range(60):
        rng = np.random.default_rng(100 + trial)
        X_tr = rng.uniform(-1.0, 1.0, size=(80, 2))
        y_tr = 2.0 * X_tr[:, 0] - X_tr[:, 1] + rng.normal(0.0, 0.2, size=80)

        def _draw_mixed(n: int) -> tuple[np.ndarray, np.ndarray]:
            is_ood = rng.random(n) < 0.25
            x_in = rng.uniform(-1.0, 1.0, size=(n, 2))
            x_out = rng.uniform(4.0, 6.0, size=(n, 2))
            X_m = np.where(is_ood[:, None], x_out, x_in)
            noise = np.where(is_ood, rng.normal(0.0, 3.0, size=n), rng.normal(0.0, 0.2, size=n))
            y_m = 2.0 * X_m[:, 0] - X_m[:, 1] + noise
            return X_m, y_m

        X_cal, y_cal = _draw_mixed(80)
        X_te, y_te = _draw_mixed(100)

        adv = fit_and_calibrate(
            X_tr,
            y_tr,
            X_cal,
            y_cal,
            alpha=0.10,
            trust_pct=75.0,
            nonconformity="absolute",
            domain_bounds=None,
            seed=trial,
        )
        res_m = adv.predict(X_te, use_mondrian=True)
        res_g = adv.predict(X_te, use_mondrian=False)
        ins = res_m["trusted"]
        if np.any(ins):
            c_in, w_in, _ = summarize(y_te, res_m["lo"], res_m["hi"], ins, pred=res_m["pred"])
            _, w_marg, _ = summarize(y_te, res_g["lo"], res_g["hi"], ins, pred=res_g["pred"])
            cov_in_trials.append(c_in)
            width_in_mond.append(w_in)
            width_in_marg.append(w_marg)

    # Selective Mondrian inside coverage meets >= 90% target while avoiding OOD width inflation!
    assert float(np.mean(cov_in_trials)) >= 0.885
    assert float(np.mean(width_in_mond)) < float(np.mean(width_in_marg))


def test_issue_03_small_calibration_returns_inf_and_emits_userwarning() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        conformal_quantile([], alpha=0.10)

    # n = 5, alpha = 0.10 -> ceil(6 * 0.9) / 5 = 6 / 5 = 1.2 > 1.0 -> must warn and return math.inf
    with pytest.warns(UserWarning, match="too small"):
        q = conformal_quantile([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.10)
    assert q == math.inf

    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(20, 3))
    y_tr = X_tr[:, 0] + rng.normal(scale=0.1, size=20)
    X_cal = rng.normal(size=(5, 3))
    y_cal = X_cal[:, 0]
    with pytest.warns(UserWarning):
        adv = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, alpha=0.10)
    assert adv.q_in == math.inf
    pred_out = adv.predict(X_tr[:2])
    assert np.all(np.isinf(pred_out["hi"]))
    assert not np.any(pred_out["trusted"])
    assert pred_out["decision"] == ["ABSTAIN_ACQUIRE", "ABSTAIN_ACQUIRE"]


def test_issue_04_smiles_whitespace_dedup_scaffold_split_and_rdkit_bridge() -> None:
    df = make_synthetic_solubility_dataframe(n=40, seed=1)
    dup_row = df.iloc[[0]].copy()
    dup_row["smiles"] = dup_row["smiles"].iloc[0] + "   "
    dup_row[ESOL_TARGET] = dup_row[ESOL_TARGET].iloc[0] + 1.0
    df_with_dup = pd.concat([df, dup_row], ignore_index=True)

    bundle = validate_and_clean_dataframe(df_with_dup, deduplicate_smiles=True)
    assert bundle.deduplicated_count == 1
    assert len(bundle.y) == 40
    assert bundle.y[0] == pytest.approx(float(df[ESOL_TARGET].iloc[0]) + 0.5, abs=1e-4)

    # Scaffold split ensures zero scaffold overlap across train / pool / cal / test
    diverse_smiles = [
        "c1ccccc1" + "C" * (i % 3) if i < 10
        else "c1ccncc1" + "C" * (i % 3) if i < 20
        else "C1CCCCC1" + "O" * (i % 2 + 1) if i < 30
        else "c1ccc2ccccc2c1" + "C" * (i % 2) if i < 40
        else "CC(=O)O" + "C" * (i % 4)
        for i in range(50)
    ]
    scaf_groups = compute_scaffold_groups(diverse_smiles)
    sp = partition_splits(50, smiles=diverse_smiles, strategy="scaffold", seed=0)
    g_tr = set(scaf_groups[sp.train_idx])
    g_pool = set(scaf_groups[sp.pool_idx])
    g_cal = set(scaf_groups[sp.cal_idx])
    g_te = set(scaf_groups[sp.test_idx])
    assert g_tr.isdisjoint(g_te) and g_cal.isdisjoint(g_te) and g_tr.isdisjoint(g_cal) and g_pool.isdisjoint(g_te)

    if HAS_RDKIT:
        desc = smiles_to_descriptors("c1ccccc1")
        assert desc["Minimum Degree"] == 2.0
        assert desc["Number of Rings"] == 1.0
        assert desc["Molecular Weight"] == pytest.approx(78.114, abs=0.02)
        mat = featurize_smiles_batch(["CCO", "c1ccccc1"])
        assert mat.shape == (2, 6)
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_descriptors("not_a_valid_smiles_###")


def test_issue_05_locally_adaptive_cqr_and_normalized_intervals_and_vector_summarize() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.5, 1.0, 2.9])
    hi = np.array([1.5, 3.5, 3.1])
    cov, mean_w, mae = summarize(y, lo, hi, np.array([True, True, True]), pred=np.array([1.0, 2.2, 3.0]))
    assert cov == pytest.approx(1.0)
    assert mean_w == pytest.approx((1.0 + 2.5 + 0.2) / 3.0)
    assert mae == pytest.approx((0.0 + 0.2 + 0.0) / 3.0)

    rng = np.random.default_rng(12)
    X_tr = rng.uniform(0.0, 1.0, size=(120, 2))
    y_tr = 3.0 * X_tr[:, 0] + rng.normal(0.0, 0.1 + 1.2 * X_tr[:, 0], size=120)
    X_cal = rng.uniform(0.0, 1.0, size=(80, 2))
    y_cal = 3.0 * X_cal[:, 0] + rng.normal(0.0, 0.1 + 1.2 * X_cal[:, 0], size=80)

    adv_norm = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, nonconformity="normalized", domain_bounds=None)
    near_pt = np.array([[0.5, 0.5]])
    far_pt = np.array([[3.5, 3.5]])
    assert adv_norm.predict(near_pt)["width"][0] < adv_norm.predict(far_pt)["width"][0]

    adv_cqr = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, nonconformity="cqr", domain_bounds=None)
    low_noise = np.array([[0.05, 0.5]])
    high_noise = np.array([[0.95, 0.5]])
    assert adv_cqr.predict(low_noise)["width"][0] < adv_cqr.predict(high_noise)["width"][0]


def test_issue_06_importance_weighted_knn_prevents_discrete_degree_false_abstention() -> None:
    rng = np.random.default_rng(42)
    # Col 0 is near-constant discrete Minimum Degree (1 on 96% of rows, 2 on 4%) with zero impact on y
    col0 = np.array([1.0] * 96 + [2.0] * 4)
    col1 = rng.uniform(50.0, 400.0, size=100)
    col2 = rng.uniform(0.0, 120.0, size=100)
    X_tr = np.column_stack([col0, col1, col2])
    y_tr = -0.01 * col1 + 0.02 * col2 + rng.normal(0.0, 0.1, size=100)

    X_cal = np.column_stack([np.ones(50), rng.uniform(50.0, 400.0, size=50), rng.uniform(0.0, 120.0, size=50)])
    y_cal = -0.01 * X_cal[:, 1] + 0.02 * X_cal[:, 2] + rng.normal(0.0, 0.1, size=50)

    benzene_like = np.array([[2.0, 200.0, 60.0]])
    adv_unif = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, distance_weighting="uniform")
    adv_imp = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, distance_weighting="importance")

    assert not adv_unif.predict(benzene_like)["trusted"][0]
    assert adv_imp.predict(benzene_like)["trusted"][0]


def test_issue_07_cold_start_knn_fewer_than_k_training_samples() -> None:
    X_tr = np.array([[1.0, 100.0], [1.0, 150.0], [1.0, 200.0]])
    y_tr = np.array([-1.0, -2.0, -3.0])
    X_cal = np.linspace(100.0, 200.0, 12).reshape(-1, 1)
    X_cal = np.column_stack([np.ones(12), X_cal])
    y_cal = -0.02 * X_cal[:, 1]

    adv = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, k=5)
    assert adv.k_eff == 3
    out = adv.predict(np.array([1.0, 140.0]))
    assert math.isfinite(out["distance"])

    with pytest.raises(ValueError, match="at least 1 sample"):
        fit_and_calibrate(np.empty((0, 2)), np.empty(0), X_cal, y_cal)


def test_issue_08_conformal_threshold_finite_sample_order_statistic() -> None:
    dists = np.arange(1.0, 11.0)  # n = 10: [1, 2, ..., 10]
    # ceil((10 + 1) * 0.80) / 10 = ceil(8.8) / 10 = 9 / 10 -> 9th order statistic = 9.0
    thr = conformal_threshold(dists, trust_pct=80.0)
    assert thr == pytest.approx(9.0)
    # Small n = 3 at 80% -> ceil(4 * 0.8) / 3 = 4 / 3 > 1.0 -> inf
    assert conformal_threshold([0.2, 0.4, 0.8], trust_pct=80.0) == math.inf


def test_issue_09_safe_atomic_download_sha256_and_corrupt_cache_recovery(tmp_path: Path) -> None:
    bad_csv = tmp_path / "delaney-processed.csv"
    bad_csv.write_text("<html>S3 503 SlowDown</html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        ensure_esol_csv(str(bad_csv), expected_sha256="a" * 64, offline=True)
    assert not bad_csv.exists(), "Corrupt cached CSV must be unlinked automatically"

    target = tmp_path / "dl" / "data.csv"
    fake_resp = io.BytesIO(b"a,b\n1,2\n")
    fake_resp.__enter__ = lambda self: self  # type: ignore[attr-defined]
    fake_resp.__exit__ = lambda self, *a: False  # type: ignore[attr-defined]
    with mock.patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download_file_atomic("https://example.org/esol.csv", str(target), expected_sha256="f" * 64)
    assert not target.exists()
    assert list(target.parent.glob("*.part*")) == []


def test_issue_10_schema_validation_nan_inf_filtering_and_min_rows() -> None:
    bad_df = pd.DataFrame({"Molecular Weight": [100.0, 120.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_and_clean_dataframe(bad_df)

    df = make_synthetic_solubility_dataframe(n=12, seed=0)
    df.loc[0, "Molecular Weight"] = float("nan")
    df.loc[1, ESOL_TARGET] = float("inf")
    bundle = validate_and_clean_dataframe(df)
    assert bundle.dropped_nonfinite_count == 2
    assert len(bundle.y) == 10

    with pytest.raises(ValueError, match="at least 4 rows"):
        validate_and_clean_dataframe(df.iloc[:4])  # 2 NaN/Inf leaves only 2 rows


def test_issue_11_empty_outside_pool_early_stop_and_clean_na_formatting() -> None:
    df = make_synthetic_solubility_dataframe(n=80, seed=3)
    bundle = validate_and_clean_dataframe(df)
    splits = partition_splits(len(bundle.y), seed=3)
    # trust_pct = 100 -> threshold = inf -> 0 outside points -> loop stops at round 0 without redundant refits
    history, _ = run_expansion_campaign(
        bundle.X,
        bundle.y,
        train_idx=splits.train_idx,
        pool_idx=splits.pool_idx,
        cal_idx=splits.cal_idx,
        test_idx=splits.test_idx,
        trust_pct=100.0,
        rounds=5,
        per_round=10,
    )
    assert len(history) == 1
    assert history[0]["n_out"] == 0

    summary, _, _ = run_esol_benchmark(bundle, trust_pct=100.0, rounds=3)
    report = format_esol_report(summary)
    assert "n/a" in report
    assert "nan" not in report.lower()


def test_issue_12_pinned_dependencies_in_requirements_and_pyproject() -> None:
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproj = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for spec in ("numpy>=1.26,<3.0", "pandas>=2.1,<3.0", "scikit-learn>=1.4,<2.0"):
        assert spec in req
        assert spec in pyproj


def test_issue_13_physical_solubility_domain_bounds_clipping() -> None:
    rng = np.random.default_rng(5)
    X_tr = rng.uniform(0.0, 1.0, size=(40, 2))
    y_tr = np.full(40, 1.4) + rng.normal(0.0, 0.25, size=40)
    X_cal = rng.uniform(0.0, 1.0, size=(30, 2))
    y_cal = np.clip(np.full(30, 1.4) + rng.normal(0.0, 0.8, size=30), -15.0, 1.70)

    adv = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, domain_bounds=(-15.0, 1.75))
    res = adv.predict(X_tr[:10])
    assert np.all(res["hi"] <= 1.75 + 1e-12)
    assert np.all(res["lo"] >= -15.0 - 1e-12)


def test_issue_14_diverse_boundary_batch_acquisition_beats_clustered_greedy() -> None:
    rng = np.random.default_rng(9)
    X_tr = rng.normal(0.0, 0.3, size=(40, 2))
    y_tr = X_tr[:, 0]
    X_cal = rng.normal(0.0, 0.3, size=(30, 2))
    y_cal = X_cal[:, 0]
    adv = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal, distance_weighting="uniform")

    # Pool has 10 nearly identical clones at [0.95, 0.0] (closest to origin) plus 3 diverse boundary directions
    clones = np.array([[0.95 + 1e-4 * i, 0.0] for i in range(10)])
    diverse_dirs = np.array([[0.0, 1.35], [-1.35, 0.0], [0.0, -1.35]])
    X_pool = np.vstack([clones, diverse_dirs])

    d_pool = adv.distance(X_pool)
    outside = np.where(d_pool > adv.threshold)[0]
    greedy_4 = outside[np.argsort(d_pool[outside])[:4]]
    diverse_4 = select_diverse_boundary_batch(adv, X_tr, X_pool, batch_size=4, shell_mult=5.0)

    def _pairwise_stats(pts: np.ndarray) -> tuple[float, float]:
        diffs = pts[:, None, :] - pts[None, :, :]
        dmat = np.sqrt(np.sum(diffs ** 2, axis=2))
        upper = dmat[np.triu_indices(len(pts), k=1)]
        return float(np.min(upper)), float(np.mean(upper))

    min_div, mean_div = _pairwise_stats(X_pool[diverse_4])
    min_grd, mean_grd = _pairwise_stats(X_pool[greedy_4])
    assert min_div > 100.0 * min_grd
    assert mean_div > 2.0 * mean_grd


def test_issue_15_huber_surrogate_and_weighted_conformal_quantile() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.5, 2.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    qw = weighted_conformal_quantile(scores, weights, alpha=0.20, test_weight=1.0)
    assert qw == pytest.approx(2.0)

    # Verify default surrogate uses Huber loss
    X = np.random.default_rng(0).normal(size=(30, 3))
    adv = fit_and_calibrate(X[:15], X[:15, 0], X[15:], X[15:, 0])
    assert adv.model.loss == "huber"


def test_issue_16_ci_workflow_caches_esol_runs_pytest_and_checks_tolerance() -> None:
    wf = (REPO_ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
    assert "actions/cache@v4" in wf
    assert "pytest" in wf
    assert "5e-3" in wf


def test_issue_17_trust_region_advisor_conformal_ucb_ei_and_safe_expand() -> None:
    df = make_synthetic_solubility_dataframe(n=160, seed=2)
    bundle = validate_and_clean_dataframe(df)
    sp = partition_splits(len(bundle.y), seed=2)
    adv = fit_and_calibrate(
        bundle.X[sp.train_idx],
        bundle.y[sp.train_idx],
        bundle.X[sp.cal_idx],
        bundle.y[sp.cal_idx],
    )
    planner = TrustRegionAdvisor(adv)

    ucb_props = planner.propose(bundle.X[sp.pool_idx], mode="conformal_ucb", batch_size=5)
    assert len(ucb_props) == 5
    assert all(p.trusted and p.mode_used == "conformal_ucb" for p in ucb_props)

    ei_props = planner.propose(bundle.X[sp.pool_idx], mode="conformal_ei", batch_size=5)
    assert len(ei_props) == 5 and all(p.trusted for p in ei_props)

    exp_props = planner.propose(bundle.X[sp.pool_idx], mode="safe_expand", batch_size=4)
    assert len(exp_props) == 4 and all(not p.trusted for p in exp_props)

    bo_cmp = benchmark_trust_region_bo(adv, bundle.X[sp.pool_idx], bundle.y[sp.pool_idx], batch_size=8)
    assert math.isfinite(bo_cmp["constrained_top_y"])


def test_issue_18_cli_flags_custom_csv_smiles_and_exit_codes(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom.csv"
    make_synthetic_solubility_dataframe(n=80, seed=0).to_csv(csv_path, index=False)

    assert cli_main(["--csv", str(csv_path), "--alpha", "0.15", "--trust-pct", "75", "--rounds", "1", "--json"]) == 0
    if HAS_RDKIT:
        assert cli_main(["--csv", str(csv_path), "--smiles", "CCO,c1ccccc1", "--json"]) == 0

    assert cli_main(["--alpha", "1.5", "--offline"]) == 2
    assert cli_main(["--trust-pct", "0", "--offline"]) == 2


def test_issue_19_offline_end_to_end_synthetic_suite(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["--dataset", "synthetic", "--offline", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["splits"]["test"] > 0
    assert payload["mondrian_trust_region"]["inside"]["mae"] < payload["mondrian_trust_region"]["outside"]["mae"]


def test_issue_20_picklable_conformal_trust_model_1d_2d_and_zero_variance() -> None:
    rng = np.random.default_rng(0)
    # Feature 0 is strictly constant (zero variance = 0.0)
    X_tr = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y_tr = 2.0 * X_tr[:, 1] + rng.normal(scale=0.1, size=30)
    X_cal = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
    y_cal = 2.0 * X_cal[:, 1] + rng.normal(scale=0.1, size=20)

    adv = fit_and_calibrate(X_tr, y_tr, X_cal, y_cal)
    blob = pickle.dumps(adv)
    restored: ConformalTrustModel = pickle.loads(blob)

    p1 = restored.predict(X_tr[0])  # 1D vector shape (3,)
    p2 = restored.predict(X_tr[:2])  # 2D matrix shape (2, 3)
    assert isinstance(p1["pred"], float) and isinstance(p1["trusted"], bool)
    assert p2["pred"][0] == pytest.approx(p1["pred"])

    # Legacy tuple unpacking remains supported
    model, q, dist_fn, thr = restored
    assert q == pytest.approx(restored.q_in)
    assert thr == pytest.approx(restored.threshold)
    assert dist_fn(X_tr[:2]).shape == (2,)


def test_issue_21_calisol23_doi_split_and_kernel_support_agreement() -> None:
    for d in CALISOL_VAL_DOIS:
        assert assign_calisol_doi_split(d) == "cal"
    for d in CALISOL_TEST_DOIS:
        assert assign_calisol_doi_split(d) == "test"
    assert assign_calisol_doi_split("10.1021/other") == "train"

    ref = [{"salt": "LiPF6", "c": 1.0, "T_K": 298.15, "solvents": {"EC": 0.5, "DMC": 0.5}}]
    sup_same, d_same = formulation_kernel_support(ref[0], ref)
    sup_diff, d_diff = formulation_kernel_support(
        {"salt": "LiTFSI", "c": 2.5, "T_K": 340.0, "solvents": {"PC": 1.0}}, ref
    )
    assert d_same == pytest.approx(0.0) and sup_same == pytest.approx(1.0)
    assert sup_diff < 0.05 and d_diff > 1.0

    cal_metrics = evaluate_calisol_conformal(csv_path=None, seed=0)
    assert cal_metrics["mae_in"] < cal_metrics["mae_out"]
    assert 0.5 <= cal_metrics["knn_vs_kernel_support_agreement"] <= 1.0


def test_issue_22_decision_envelope_json_schema_and_lab_ask_why_probe_bridge(tmp_path: Path) -> None:
    schema_path = tmp_path / "decision.schema.json"
    assert cli_main(["--export-schema", str(schema_path)]) == 0
    assert json.loads(schema_path.read_text(encoding="utf-8"))["title"] == "DecisionEnvelope"

    df = make_synthetic_solubility_dataframe(n=100, seed=0)
    bundle = validate_and_clean_dataframe(df)
    sp = partition_splits(len(bundle.y), seed=0)
    adv = fit_and_calibrate(
        bundle.X[sp.train_idx],
        bundle.y[sp.train_idx],
        bundle.X[sp.cal_idx],
        bundle.y[sp.cal_idx],
        feature_names=ESOL_FEATURES,
    )

    envs = build_decision_envelopes(adv, bundle.X[sp.test_idx[:5]], candidate_ids=[f"c{i}" for i in range(5)])
    for e in envs:
        jsonschema.validate(instance=e.to_dict(), schema=DECISION_ENVELOPE_JSON_SCHEMA)

    out_jsonl = tmp_path / "decisions.jsonl"
    export_decisions_jsonl(envs, str(out_jsonl))
    reloaded = load_decisions_jsonl(str(out_jsonl))
    assert [r.to_dict() for r in reloaded] == [e.to_dict() for e in envs]

    # Direct bridge from lab-ask-why EpistemicProbeResult payload
    probe_payload = [
        {
            "claim": "Polar hydroxyl substitution increases aqueous solubility.",
            "discriminating_observable": {
                "observable_name": "ethanol_solubility_check",
                "candidate_descriptors": {f: float(bundle.X[0, j]) for j, f in enumerate(ESOL_FEATURES)},
                "candidate_smiles": "CCO",
            },
        }
    ]
    bridged = evaluate_epistemic_probes(adv, probe_payload)
    assert len(bridged) == 1
    assert bridged[0].source_claim == "Polar hydroxyl substitution increases aqueous solubility."
    jsonschema.validate(instance=bridged[0].to_dict(), schema=DECISION_ENVELOPE_JSON_SCHEMA)
