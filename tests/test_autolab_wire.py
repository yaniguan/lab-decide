"""Unit tests for the Auto-Lab 15-repo wire node (`autolab_wire.py`, Issues #24..#32)."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import autolab_wire
import lab_decide.chemistry as chem_mod
from autolab_wire import (
    LAYER,
    REPO_DIR,
    REPO_ID,
    SCHEMA_VERSION,
    get_cached_esol_model,
    process_packet,
)
from lab_decide.chemistry import HAS_RDKIT
from lab_decide.conformal import summarize
from lab_decide.data import BUNDLED_ESOL_RESOURCE_PATH, load_dataset, partition_splits

C60_FULLERENE_SMILES = (
    "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41"
)
CYCLOSPORIN_A_SMILES = (
    "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)"
    "C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O)C(C)C"
)
SUCROSE_SMILES = "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O"


def test_wire_metadata_and_repo_dir_resolution() -> None:
    assert REPO_ID == "lab-decide"
    assert LAYER == "L6"
    assert SCHEMA_VERSION == "autolab-wire/v1"
    assert REPO_DIR == Path(autolab_wire.__file__).resolve().parent
    assert (REPO_DIR / "src" / "lab_decide" / "conformal.py").is_file()


def test_halted_packet_passes_through_unchanged() -> None:
    halted_in = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L2_SPEECH_PRIVACY_BLOCK",
        "trace": [{"repo": "lab-ears", "layer": "L2", "status": "BLOCK", "summary": "speech band"}],
        "bottom_reasoning_traces": [],
    }
    out = process_packet(halted_in)
    assert out == halted_in
    assert out is not halted_in
    assert "decision" not in out


def test_fail_closed_when_rdkit_missing_invalid_smiles_or_esol_uncached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 1. Missing RDKit -> fails closed without string-length descriptor fallback
    monkeypatch.setattr(autolab_wire, "HAS_RDKIT", False)
    monkeypatch.setattr(chem_mod, "HAS_RDKIT", False)
    out_no_rdkit = process_packet({"candidate_smiles": "CCO"})
    assert out_no_rdkit["halted"] is True
    assert out_no_rdkit["halt_reason"] == "L6_ABSTAIN"
    assert out_no_rdkit["decision"]["abstain"] is True
    assert out_no_rdkit["decision"]["abstain_reason"] == "rdkit_unavailable"
    json.dumps(out_no_rdkit, allow_nan=False)


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for SMILES descriptor scoring")
def test_fail_closed_on_unparseable_smiles_and_missing_or_corrupt_esol_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Invalid SMILES -> fails closed
    bad_smi = process_packet({"candidate_smiles": "not_a_valid_smiles_###"})
    assert bad_smi["halted"] is True
    assert bad_smi["halt_reason"] == "L6_ABSTAIN"
    assert bad_smi["decision"]["abstain"] is True
    assert bad_smi["decision"]["abstain_reason"] == "invalid_candidate_smiles"

    # Missing ESOL CSV cache -> fails closed with clear message
    missing_csv = tmp_path / "missing_esol.csv"
    monkeypatch.setattr(autolab_wire, "_DEFAULT_ESOL_CANDIDATE_PATHS", (missing_csv,))
    out_missing = process_packet({"candidate_smiles": "CCO"})
    assert out_missing["halted"] is True
    assert out_missing["halt_reason"] == "L6_ABSTAIN"
    assert out_missing["decision"]["abstain"] is True
    assert out_missing["decision"]["abstain_reason"] == "esol_cache_unavailable"
    assert "not cached" in out_missing["trace"][-1]["summary"]

    # Corrupt ESOL CSV SHA-256 -> fails closed
    corrupt_csv = tmp_path / "corrupt_esol.csv"
    corrupt_csv.write_text("smiles,measured log solubility in mols per litre\nCCO,0.5\n", encoding="utf-8")
    out_corrupt = process_packet({"candidate_smiles": "CCO", "esol_csv_path": str(corrupt_csv)})
    assert out_corrupt["halted"] is True
    assert out_corrupt["decision"]["abstain"] is True
    assert out_corrupt["decision"]["abstain_reason"] == "esol_cache_unavailable"
    assert "SHA-256" in out_corrupt["trace"][-1]["summary"]


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for SMILES descriptor scoring")
def test_issue_32_c60_cyclosporin_and_sucrose_abstain_without_knn_distance() -> None:
    adv = get_cached_esol_model()
    for name, smi in (
        ("C60", C60_FULLERENE_SMILES),
        ("cyclosporin_A", CYCLOSPORIN_A_SMILES),
        ("sucrose", SUCROSE_SMILES),
    ):
        pkt = {"candidate_smiles": smi}
        out = process_packet(pkt)
        assert "decision" not in pkt, "Input packet must not be mutated"
        assert out["halted"] is True, f"{name} should halt with L6_ABSTAIN"
        assert out["halt_reason"] == "L6_ABSTAIN"
        dec = out["decision"]
        assert dec["abstain"] is True, f"{name} must abstain"
        assert dec["in_trust_region"] is False
        assert dec["distance_source"] == "fitted_esol_model"
        assert dec["knn_distance"] > adv.threshold
        json.dumps(out, allow_nan=False)


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for SMILES descriptor scoring")
def test_issue_32_exact_parity_with_conformal_trust_model_on_seed0_esol_test_split() -> None:
    bundle = load_dataset(csv_path=BUNDLED_ESOL_RESOURCE_PATH, offline=True)
    splits = partition_splits(len(bundle.y), strategy="random", seed=0, fracs=(0.50, 0.20, 0.15, 0.15))
    adv = get_cached_esol_model(BUNDLED_ESOL_RESOURCE_PATH)

    te_idx = splits.test_idx
    assert len(te_idx) == 169
    ref = adv.predict(bundle.X[te_idx], use_mondrian=True)

    wire_lo = np.empty(len(te_idx), dtype=np.float64)
    wire_hi = np.empty(len(te_idx), dtype=np.float64)
    wire_pred = np.empty(len(te_idx), dtype=np.float64)
    wire_trusted = np.empty(len(te_idx), dtype=bool)

    for pos, row_i in enumerate(te_idx):
        smi = bundle.smiles[int(row_i)]
        out = process_packet(
            {
                "candidate_smiles": smi,
                "esol_csv_path": BUNDLED_ESOL_RESOURCE_PATH,
                # Packet knn_distance without distance_source='upstream' must be ignored!
                "knn_distance": 0.001,
            }
        )
        dec = out["decision"]
        ref_1d = adv.predict(bundle.X[int(row_i)], use_mondrian=True)
        assert dec["distance_source"] == "fitted_esol_model"
        assert dec["in_trust_region"] is bool(ref_1d["trusted"])
        assert dec["abstain"] is (not bool(ref_1d["trusted"]))
        assert dec["knn_distance"] == float(ref_1d["distance"])
        assert dec["predicted_log_s"] == float(ref_1d["pred"])
        assert dec["conformal_interval"] == [float(ref_1d["lo"]), float(ref_1d["hi"])]
        assert dec["conformal_interval"] == pytest.approx(
            [float(ref["lo"][pos]), float(ref["hi"][pos])], rel=0.0, abs=1e-12
        )

        wire_trusted[pos] = bool(dec["in_trust_region"])
        wire_lo[pos] = float(dec["conformal_interval"][0])
        wire_hi[pos] = float(dec["conformal_interval"][1])
        wire_pred[pos] = float(dec["predicted_log_s"])

    cov_in, _, _ = summarize(bundle.y[te_idx], wire_lo, wire_hi, wire_trusted, pred=wire_pred)
    assert cov_in >= 0.88
    assert 0.10 <= float(np.mean(~wire_trusted)) <= 0.25


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for SMILES descriptor scoring")
def test_issue_32_distance_source_upstream_and_residual_gated_ledger_recalibration() -> None:
    adv = get_cached_esol_model()

    # Default aspirin is inside the ESOL trust region
    in_dom = process_packet({"candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert in_dom["halted"] is False
    assert in_dom["decision"]["abstain"] is False
    assert in_dom["decision"]["distance_source"] == "fitted_esol_model"
    assert in_dom["decision"]["effective_n_calib"] == 169
    assert in_dom["decision"]["effective_threshold"] == pytest.approx(adv.threshold)

    # Packet knn_distance is ignored unless distance_source == 'upstream'
    ignored_d = process_packet({"candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O", "knn_distance": 9.99})
    assert ignored_d["decision"]["abstain"] is False
    assert ignored_d["decision"]["distance_source"] == "fitted_esol_model"

    upstream_ood = process_packet(
        {
            "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "knn_distance": 0.90,
            "distance_source": "upstream",
        }
    )
    assert upstream_ood["halted"] is True
    assert upstream_ood["halt_reason"] == "L6_ABSTAIN"
    assert upstream_ood["decision"]["abstain"] is True
    assert upstream_ood["decision"]["distance_source"] == "upstream"
    assert upstream_ood["decision"]["knn_distance"] == pytest.approx(0.90)

    # Issue #32 Acceptance Criterion: 10 ledger entries with residual 5.0 MUST NOT turn
    # an abstaining candidate into an answer and MUST NOT widen effective_threshold!
    large_res_entries = [
        {
            "run_id": f"high-err-{i:03d}",
            "candidate_smiles": SUCROSE_SMILES,
            "admit_to_memory": True,
            "ledger_entry_sha256": f"{100 + i:064x}",
            "calibration_residual": 5.0,
        }
        for i in range(10)
    ]
    still_abstain_sucrose = process_packet(
        {
            "candidate_smiles": SUCROSE_SMILES,
            "certified_ledger_entries": large_res_entries,
        }
    )
    assert still_abstain_sucrose["halted"] is True
    assert still_abstain_sucrose["decision"]["abstain"] is True
    assert still_abstain_sucrose["decision"]["effective_n_calib"] == 179
    assert still_abstain_sucrose["decision"]["effective_threshold"] == pytest.approx(adv.threshold)

    still_abstain_upstream = process_packet(
        {
            "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "knn_distance": 0.90,
            "distance_source": "upstream",
            "certified_ledger_entries": large_res_entries,
        }
    )
    assert still_abstain_upstream["halted"] is True
    assert still_abstain_upstream["decision"]["abstain"] is True
    assert still_abstain_upstream["decision"]["effective_threshold"] == pytest.approx(adv.threshold)

    # Genuine well-calibrated boundary measurements (measured_y close to model prediction)
    # recompute threshold from empirical calibration distances without entry-counting heuristics.
    sucrose_pred = float(process_packet({"candidate_smiles": SUCROSE_SMILES})["decision"]["predicted_log_s"])
    accurate_boundary_entries = [
        {
            "run_id": f"calib-ok-{i:03d}",
            "candidate_smiles": SUCROSE_SMILES,
            "measured_y": sucrose_pred + 0.05,
            "admit_to_memory": True,
            "ledger_entry_sha256": f"{500 + i:064x}",
        }
        for i in range(5)
    ]
    recal_ok = process_packet(
        {
            "candidate_smiles": "CCO",
            "certified_ledger_entries": accurate_boundary_entries,
        }
    )
    assert recal_ok["decision"]["effective_n_calib"] == 174
    assert recal_ok["decision"]["effective_threshold"] > adv.threshold


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for SMILES descriptor scoring")
def test_issue_28_sha256_deduplication_and_rank_guards_on_fitted_model() -> None:
    single_obs = {
        "run_id": "run-001",
        "candidate_smiles": "CCO",
        "measured_y": 0.80,
        "admit_to_memory": True,
        "ledger_entry_sha256": "a" * 64,
    }
    out_single = process_packet({"candidate_smiles": "CCO", "certified_ledger_entries": [single_obs]})
    out_replay = process_packet({"candidate_smiles": "CCO", "certified_ledger_entries": [single_obs] * 25})
    assert out_replay["decision"]["effective_n_calib"] == 170
    assert out_replay["decision"]["duplicate_ledger_entries_ignored"] == 24
    assert out_replay["decision"]["conformal_interval"] == out_single["decision"]["conformal_interval"]

    # Undersized calibration bucket at alpha=0.10 (n=5 -> rank=6 > 5) must abstain
    undersized = process_packet({"candidate_smiles": "CCO", "n_calib_override": 5, "alpha": 0.10})
    assert undersized["halted"] is True
    assert undersized["halt_reason"] == "L6_ABSTAIN"
    assert undersized["decision"]["abstain"] is True
    assert undersized["decision"]["effective_n_calib"] == 5

    # Malformed SHA-256 and quarantined entries are rejected
    malformed = process_packet(
        {
            "candidate_smiles": "CCO",
            "certified_ledger_entries": [
                {"admit_to_memory": True},
                {"admit_to_memory": False, "ledger_entry_sha256": "b" * 64, "measured_y": 0.1},
                {"admit_to_memory": True, "ledger_entry_sha256": "bad", "measured_y": 0.1},
                {"ledger_entry_sha256": "c" * 64, "calibration_residual": float("nan")},
            ],
        }
    )
    assert malformed["decision"]["invalid_ledger_entries_ignored"] == 4
    assert malformed["decision"]["effective_n_calib"] == 169


def test_collision_free_dynamic_load_when_community_extensions_already_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_pkg = types.ModuleType("community_extensions")
    monkeypatch.setitem(sys.modules, "community_extensions", dummy_pkg)
    monkeypatch.delitem(sys.modules, "_lab_decide_mondrian_drift_conformal", raising=False)

    spec = importlib.util.spec_from_file_location("_test_decide_wire_isolated", REPO_DIR / "autolab_wire.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_decide_wire_isolated"] = mod
    try:
        spec.loader.exec_module(mod)
        res = mod.process_packet({})
        if HAS_RDKIT:
            assert res["decision"]["in_trust_region"] is True
        else:
            assert res["decision"]["abstain"] is True
        assert sys.modules["community_extensions"] is dummy_pkg
    finally:
        sys.modules.pop("_test_decide_wire_isolated", None)
