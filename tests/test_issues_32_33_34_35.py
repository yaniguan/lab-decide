"""Comprehensive regression tests for Issues #32, #33, #34, and #35 ( plus README drift guards)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

import lab_decide.chemistry as chem_mod
from lab_decide import (
    BUNDLED_ESOL_RESOURCE_PATH,
    HAS_RDKIT,
    OPEN_CALIBRATION_DATASETS,
    TrustRegionAdvisor,
    calibrate_open_dataset,
    default_dataset_cache_path,
    fit_and_calibrate,
    load_dataset,
    load_open_dataset,
    murcko_scaffold_group,
    partition_splits,
    run_sequential_bo_campaign,
    write_schema_sample_fixture,
)
from lab_decide.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str) -> Any:
    script_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Issue #35: Balanced group-packing split, explicit group_keys, no regex fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for ESOL/FreeSolv Bemis-Murcko scaffolds")
def test_issue_35_esol_and_freesolv_20seed_balanced_scaffold_fractions(tmp_path: Path) -> None:
    req = (0.50, 0.20, 0.15, 0.15)
    esol_bundle = load_dataset(csv_path=BUNDLED_ESOL_RESOURCE_PATH, offline=True)

    for seed in range(20):
        sp = partition_splits(len(esol_bundle.y), smiles=esol_bundle.smiles, strategy="scaffold", seed=seed, fracs=req)
        assert sp.achieved_fractions is not None
        for k_name, f_req in zip(("train", "pool", "calibration", "test"), req):
            assert abs(sp.achieved_fractions[k_name] - f_req) <= 0.03, (
                f"ESOL seed {seed} {k_name} fraction {sp.achieved_fractions[k_name]:.4f} deviated > 3pp from {f_req}"
            )
        # Disjoint scaffold check
        assert esol_bundle.group_keys is not None
        gk = np.asarray(esol_bundle.group_keys, dtype=object)
        assert set(gk[sp.train_idx]).isdisjoint(set(gk[sp.test_idx]))
        assert set(gk[sp.cal_idx]).isdisjoint(set(gk[sp.test_idx]))

    # FreeSolv: use cached real database.json if present, else bundled real 642-compound minimal CSV
    fs_cache = Path(default_dataset_cache_path("freesolv"))
    if fs_cache.is_file():
        fs_bundle = load_open_dataset("freesolv", path=fs_cache, offline=True)
    else:
        fs_min = REPO_ROOT / "src" / "lab_decide" / "resources" / "freesolv_minimal.csv"
        fs_bundle = load_open_dataset("freesolv", path=fs_min, fmt="csv", offline=True)
    # Note: on real FreeSolv (n=642), acyclic=320 (49.84%) goes to train and single-scaffold benzene=152
    # (23.68%) goes to pool, making 152/642 - 0.20 = 0.0368 the exact integer minimax bound on pool while
    # train/cal/test stay within < 2.0pp and mean absolute split deviation stays < 2.0pp across all 20 seeds.
    for seed in range(20):
        sp_fs = partition_splits(
            len(fs_bundle.y),
            smiles=fs_bundle.smiles,
            strategy="scaffold",
            seed=seed,
            fracs=req,
        )
        assert sp_fs.achieved_fractions is not None
        errs = [
            abs(sp_fs.achieved_fractions[k_name] - f_req)
            for k_name, f_req in zip(("train", "pool", "calibration", "test"), req)
        ]
        assert float(np.mean(errs)) <= 0.025
        for k_name, f_req in (("train", 0.50), ("calibration", 0.15), ("test", 0.15)):
            assert abs(sp_fs.achieved_fractions[k_name] - f_req) <= 0.03, (
                f"FreeSolv seed {seed} {k_name} fraction {sp_fs.achieved_fractions[k_name]:.4f} deviated > 3pp"
            )
        assert abs(sp_fs.achieved_fractions["pool"] - 0.20) <= 0.0375


def test_issue_35_no_regex_fallback_and_disjoint_solute_and_doi_group_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. BigSolDB group split leaves zero shared solutes between train, pool, cal, and test
    bigsol_path = tmp_path / "bigsol.csv"
    write_schema_sample_fixture("bigsoldb_v2_2", bigsol_path, n=160, seed=0)
    b_sol = load_open_dataset("bigsoldb_v2_2", path=bigsol_path)
    assert b_sol.group_keys is not None
    assert b_sol.group_kind == "solute"
    assert not any(g.startswith("fallback_r") for g in b_sol.group_keys)

    _, _, sp_sol, m_sol = calibrate_open_dataset(
        "bigsoldb_v2_2", path=bigsol_path, split_strategy="group", seed=0
    )
    sol_arr = np.asarray(b_sol.group_keys, dtype=object)
    assert set(sol_arr[sp_sol.train_idx]).isdisjoint(set(sol_arr[sp_sol.test_idx]))
    assert set(sol_arr[sp_sol.train_idx]).isdisjoint(set(sol_arr[sp_sol.cal_idx]))
    assert m_sol["n_groups_total"] >= 4
    assert m_sol["achieved_fractions"] is not None

    # 2. CALiSol-23 group split leaves zero shared DOIs between train, pool, cal, and test
    cal_path = tmp_path / "calisol.csv"
    write_schema_sample_fixture("calisol23", cal_path, n=160, seed=0)
    b_cal = load_open_dataset("calisol23", path=cal_path)
    assert b_cal.group_keys is not None
    assert b_cal.group_kind == "doi"
    assert not any(g.startswith("fallback_r") for g in b_cal.group_keys)

    _, _, sp_cal, m_cal = calibrate_open_dataset(
        "calisol23", path=cal_path, split_strategy="group", seed=0
    )
    doi_arr = np.asarray(b_cal.group_keys, dtype=object)
    assert set(doi_arr[sp_cal.train_idx]).isdisjoint(set(doi_arr[sp_cal.test_idx]))
    assert set(doi_arr[sp_cal.cal_idx]).isdisjoint(set(doi_arr[sp_cal.test_idx]))
    assert m_cal["group_counts"] is not None

    # 3. Composite-key datasets reject strategy='scaffold' instead of falling back to regex
    with pytest.raises(ValueError, match="group"):
        calibrate_open_dataset("calisol23", path=cal_path, split_strategy="scaffold", seed=0)

    # 4. Missing RDKit raises explicit ImportError in scaffold mode
    monkeypatch.setattr(chem_mod, "HAS_RDKIT", False)
    with pytest.raises(ImportError, match="RDKit"):
        murcko_scaffold_group("c1ccccc1")
    with pytest.raises(ImportError, match="RDKit"):
        partition_splits(16, smiles=["CCO"] * 16, strategy="scaffold", seed=0)


# ---------------------------------------------------------------------------
# Issue #34: Strict --offline fail-closed, data_source provenance & HTE encodings
# ---------------------------------------------------------------------------


def test_issue_34_offline_without_cache_fails_unless_allow_synthetic_and_sets_data_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_esol = str(tmp_path / "no_esol.csv")
    missing_cal = str(tmp_path / "no_calisol.csv")
    missing_aq = str(tmp_path / "no_aqsoldb.csv")

    with mock.patch("lab_decide.data.DEFAULT_DATA_PATH", missing_esol):
        assert cli_main(["--offline", "--json"]) != 0
        capsys.readouterr()

        assert cli_main(["--offline", "--allow-synthetic", "--json"]) == 0
        esol_syn = json.loads(capsys.readouterr().out)
        assert esol_syn["data_source"] == "synthetic_fixture"

        assert cli_main(["--offline", "--allow-synthetic"]) == 0
        esol_txt = capsys.readouterr().out
        assert "data_source=synthetic_fixture" in esol_txt

    with mock.patch("lab_decide.cli.default_dataset_cache_path", return_value=missing_cal):
        assert cli_main(["--dataset", "calisol23", "--offline", "--json"]) != 0
        capsys.readouterr()

        assert cli_main(["--dataset", "calisol23", "--offline", "--allow-synthetic", "--json"]) == 0
        cal_syn = json.loads(capsys.readouterr().out)
        assert cal_syn["data_source"] == "synthetic_fixture"

    with mock.patch("lab_decide.cli.default_dataset_cache_path", return_value=missing_aq):
        assert cli_main(["--dataset", "aqsoldb", "--offline", "--json"]) != 0
        capsys.readouterr()

        assert cli_main(["--dataset", "aqsoldb", "--offline", "--allow-synthetic", "--json"]) == 0
        aq_syn = json.loads(capsys.readouterr().out)
        assert aq_syn["data_source"] == "synthetic_fixture"

    # Real CSV explicitly reports data_source == 'real_csv'
    assert cli_main(["--csv", BUNDLED_ESOL_RESOURCE_PATH, "--json"]) == 0
    real_payload = json.loads(capsys.readouterr().out)
    assert real_payload["data_source"] == "real_csv"


def test_issue_34_hte_onehot_and_mahalanobis_encodings(tmp_path: Path) -> None:
    bh_path = tmp_path / "bh.csv"
    write_schema_sample_fixture("buchwald_hartwig_doyle", bh_path, n=140, seed=0)

    _, b_oh, _, m_oh = calibrate_open_dataset(
        "buchwald_hartwig_doyle",
        path=bh_path,
        encoding="onehot",
        distance_metric="mahalanobis",
        nonconformity="cqr_r",
        seed=0,
    )
    assert b_oh.X.shape[1] >= 10
    assert m_oh["encoding"] == "onehot"
    assert m_oh["distance_metric"] == "mahalanobis"
    assert m_oh["data_source"] == "real_csv"

    if HAS_RDKIT:
        _, b_ohm, _, m_ohm = calibrate_open_dataset(
            "buchwald_hartwig_doyle",
            path=bh_path,
            encoding="onehot_morgan",
            distance_metric="ensemble_mahalanobis",
            nonconformity="cqr_r",
            seed=0,
        )
        assert b_ohm.X.shape[1] > b_oh.X.shape[1]
        assert m_ohm["encoding"] == "onehot_morgan"
        assert m_ohm["distance_metric"] == "ensemble_mahalanobis"


# ---------------------------------------------------------------------------
# Issue #33: Explicit objective_direction, UCB clip-bound tie-break & campaigns
# ---------------------------------------------------------------------------


def test_issue_33_explicit_objective_direction_and_ucb_clip_bound_tie_break() -> None:
    assert OPEN_CALIBRATION_DATASETS["freesolv"].objective_direction == "calibration_only"
    assert OPEN_CALIBRATION_DATASETS["lipophilicity_astrazeneca"].objective_direction == "calibration_only"
    assert OPEN_CALIBRATION_DATASETS["biogen_adme"].objective_direction == "minimize"
    assert OPEN_CALIBRATION_DATASETS["esol_delaney"].objective_direction == "maximize"
    assert OPEN_CALIBRATION_DATASETS["buchwald_hartwig_doyle"].objective_direction == "maximize"

    # Construct a controlled 1D surrogate where multiple trusted pool candidates hit the upper clip bound 1.75
    X_tr = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y_tr = 2.0 * X_tr[:, 0]
    X_cal = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
    y_cal = 2.0 * X_cal[:, 0] + np.where(np.arange(30) % 2 == 0, 1.2, -1.2)

    adv = fit_and_calibrate(
        X_tr,
        y_tr,
        X_cal,
        y_cal,
        alpha=0.10,
        trust_pct=95.0,
        nonconformity="absolute",
        domain_bounds=(-15.0, 1.75),
        seed=0,
    )
    # Candidate 0 (x=0.50 -> pred~1.00, unclipped hi~2.20, clipped hi=1.75)
    # Candidate 1 (x=0.82 -> pred~1.64, unclipped hi~2.84, clipped hi=1.75)
    X_pool = np.array([[0.50], [0.82]])
    props = TrustRegionAdvisor(adv).propose(
        X_pool, mode="conformal_ucb", batch_size=2, maximize=True, beta=1.0
    )
    assert len(props) == 2
    assert props[0].hi == pytest.approx(1.75)
    assert props[1].hi == pytest.approx(1.75)
    assert props[0].acquisition_score == pytest.approx(1.75)
    assert props[1].acquisition_score == pytest.approx(1.75)
    # Candidate 1 MUST rank ahead of Candidate 0 via unclipped UCB tie-breaking!
    assert props[0].pool_index == 1
    assert props[1].pool_index == 0


def test_issue_33_sequential_bo_campaign_and_demo_output_both_top_values() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(120, 3))
    y = 2.5 * X[:, 0] - 1.2 * (X[:, 1] ** 2) + rng.normal(0.0, 0.15, size=120)
    sp = partition_splits(120, seed=0, fracs=(0.25, 0.45, 0.15, 0.15))

    for method in ("random", "greedy", "unconstrained_conformal_ucb", "tr_conformal_ucb", "tr_conformal_ei", "gp_ei"):
        hist = run_sequential_bo_campaign(
            X,
            y,
            train_idx=sp.train_idx,
            pool_idx=sp.pool_idx,
            cal_idx=sp.cal_idx,
            method=method,
            rounds=3,
            batch_size=4,
            maximize=True,
            domain_bounds=None,
            seed=0,
        )
        assert len(hist) == 3
        assert hist[-1]["best_so_far"] >= hist[0]["best_so_far"] - 1e-12
        assert hist[-1]["simple_regret"] <= hist[0]["simple_regret"] + 1e-12
        assert 0.0 <= hist[-1]["cumulative_ood_fraction"] <= 1.0
        if method in ("tr_conformal_ucb", "tr_conformal_ei"):
            assert hist[0]["round_ood_fraction"] == 0.0

    demo_out = (REPO_ROOT / "demo" / "output.txt").read_text(encoding="utf-8")
    assert "Single-batch trust-region ranking" in demo_out
    assert "Constrained top logS:" in demo_out
    assert "Unconstrained top logS:" in demo_out


def test_committed_benchmark_jsons_and_readme_drift_guard() -> None:
    reg_path = REPO_ROOT / "results" / "registry_benchmark.json"
    camp_path = REPO_ROOT / "results" / "campaign_benchmark.json"
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert reg_path.is_file(), "results/registry_benchmark.json must be committed"
    assert camp_path.is_file(), "results/campaign_benchmark.json must be committed"

    reg_payload = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg_payload["data_source"] == "real_csv"
    assert len(reg_payload["primary_11_datasets"]) == 11
    for row in reg_payload["primary_11_datasets"]:
        assert row["n_seeds"] >= 5
        assert row["data_source"] == "real_csv"
        assert isinstance(row["gate_informative"], bool)
        assert row["gate_informative"] is True, f"Primary gate must be informative on {row['dataset_id']}"

    assert "esol_delaney_20seed_head_to_head" in reg_payload
    assert "molecular_encoding_comparison" in reg_payload
    h2h_r = reg_payload["esol_delaney_20seed_head_to_head"]["random"]
    h2h_s = reg_payload["esol_delaney_20seed_head_to_head"]["scaffold"]
    front_r = h2h_r["models"]["frontier_rdkit_2d_morgan"]["metrics"]
    front_s = h2h_s["models"]["frontier_rdkit_2d_morgan"]["metrics"]
    del_s = h2h_s["models"]["delaney_2004_csv_formula"]["metrics"]
    assert float(front_r["mae_all"]["mean"]) <= 0.44
    assert float(front_r["mae_in"]["mean"]) <= 0.38
    assert float(front_r["rmse_all"]["mean"]) <= 0.60
    assert float(h2h_r["mae_reduction_vs_delaney_csv_pct"]) >= 35.0
    assert float(h2h_r["mae_reduction_vs_gbdt_6desc_pct"]) >= 34.0
    assert float(front_s["mae_all"]["mean"]) < float(del_s["mae_all"]["mean"])
    assert h2h_r["models"]["frontier_rdkit_2d_morgan"]["gate_informative"] is True
    assert h2h_s["models"]["frontier_rdkit_2d_morgan"]["gate_informative"] is True

    camp_payload = json.loads(camp_path.read_text(encoding="utf-8"))
    assert camp_payload["data_source"] == "real_csv"
    assert camp_payload["n_seeds"] >= 10
    assert {d["dataset_id"] for d in camp_payload["datasets"]} == {
        "esol_delaney",
        "direct_arylation_shields",
        "buchwald_hartwig_doyle",
    }

    reg_mod = _load_script_module("_bench_reg_mod", "scripts/benchmark_registry.py")
    camp_mod = _load_script_module("_bench_camp_mod", "scripts/benchmark_campaign.py")

    expected_reg_md = reg_mod.render_registry_markdown(reg_payload)
    expected_camp_md = camp_mod.render_campaign_markdown(camp_payload)
    assert expected_reg_md in readme_text, "README.md registry benchmark table drifted from results/registry_benchmark.json"
    assert expected_camp_md in readme_text, "README.md campaign benchmark table drifted from results/campaign_benchmark.json"
