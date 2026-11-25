"""Unit tests for the Auto-Lab 15-repo wire node (`autolab_wire.py`) in lab-decide."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import autolab_wire
from autolab_wire import LAYER, REPO_DIR, REPO_ID, SCHEMA_VERSION, process_packet


def test_wire_metadata_and_repo_dir_resolution() -> None:
    assert REPO_ID == "lab-decide"
    assert LAYER == "L6"
    assert SCHEMA_VERSION == "autolab-wire/v1"
    assert REPO_DIR == Path(autolab_wire.__file__).resolve().parent
    assert (REPO_DIR / "src" / "lab_decide" / "conformal.py").is_file()


def test_in_domain_candidate_passes_conformal_gate() -> None:
    inp = {
        "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "knn_distance": 0.412,
        "knn_threshold": 0.758,
        "n_calib": 226,
        "alpha": 0.10,
    }
    out = process_packet(inp)

    # Original packet must never be mutated
    assert "decision" not in inp

    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None

    dec = out["decision"]
    assert dec["candidate_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert dec["in_trust_region"] is True
    assert dec["abstain"] is False
    assert dec["knn_distance"] == pytest.approx(0.412)
    assert dec["effective_threshold"] == pytest.approx(0.758)
    assert dec["effective_n_calib"] == 226

    lo, hi = dec["conformal_interval"]
    assert -15.0 <= lo < hi <= 1.75

    assert len(out["trace"]) == 1
    assert out["trace"][0]["repo"] == "lab-decide"
    assert out["trace"][0]["layer"] == "L6"
    assert out["trace"][0]["status"] == "PASS"

    assert len(out["bottom_reasoning_traces"]) == 1
    brt = out["bottom_reasoning_traces"][0]
    assert brt["repo"] == "lab-decide"
    assert brt["invariant"] == "out_of_domain_always_abstains"
    assert "lab-attrib" in brt["first_principles"]

    # Strict RFC 8259 JSON serializability (no NaN / Infinity)
    serialized = json.dumps(out, allow_nan=False)
    assert "lab-decide" in serialized


def test_out_of_domain_candidate_triggers_l6_abstain() -> None:
    out = process_packet({"knn_distance": 1.38, "knn_threshold": 0.758, "n_calib": 226})

    assert out["halted"] is True
    assert out["halt_reason"] == "L6_ABSTAIN"
    dec = out["decision"]
    assert dec["abstain"] is True
    assert dec["in_trust_region"] is False
    assert dec["knn_distance"] == pytest.approx(1.38)
    assert dec["effective_threshold"] == pytest.approx(0.758)
    assert out["trace"][-1]["status"] == "ABSTAIN"


def test_undersized_calibration_set_triggers_l6_abstain() -> None:
    # n_calib = 5 at alpha = 0.10 -> ceil(6 * 0.90) = 6 > 5 -> finite-sample rank impossible
    out = process_packet({"knn_distance": 0.30, "knn_threshold": 0.758, "n_calib": 5, "alpha": 0.10})

    assert out["halted"] is True
    assert out["halt_reason"] == "L6_ABSTAIN"
    dec = out["decision"]
    assert dec["abstain"] is True
    assert dec["in_trust_region"] is False
    assert dec["effective_n_calib"] == 5
    assert dec["conformal_interval"] == [-15.0, 1.75]
    assert out["trace"][-1]["status"] == "ABSTAIN"


def test_closed_loop_certified_ledger_entries_expand_calibration_and_threshold() -> None:
    # Borderline candidate at knn_distance = 0.785 exceeds base_threshold = 0.758 prior to feedback
    baseline = process_packet(
        {
            "candidate_smiles": "c1ccccc1O",
            "knn_distance": 0.785,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [],
        }
    )
    assert baseline["halted"] is True
    assert baseline["halt_reason"] == "L6_ABSTAIN"
    assert baseline["decision"]["abstain"] is True

    # Supply two valid certified entries from lab-attrib plus one quarantined entry that must be ignored
    recalibrated = process_packet(
        {
            "candidate_smiles": "c1ccccc1O",
            "knn_distance": 0.785,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [
                {
                    "run_id": "run-001",
                    "candidate_smiles": "c1ccccc1",
                    "outcome": "CERTIFIED_POSITIVE",
                    "admit_to_memory": True,
                    "ledger_entry_sha256": "a" * 64,
                },
                {
                    "run_id": "run-002",
                    "ledger_entry_sha256": "b" * 64,
                },
                {
                    "run_id": "run-quarantined",
                    "outcome": "AMBIGUOUS_CHARACTERIZATION",
                    "admit_to_memory": False,
                    "ledger_entry_sha256": None,
                },
            ],
        }
    )
    assert recalibrated["halted"] is False
    assert recalibrated["halt_reason"] is None
    dec = recalibrated["decision"]
    assert dec["effective_n_calib"] == 228
    assert dec["effective_threshold"] == pytest.approx(0.798)
    assert dec["in_trust_region"] is True
    assert dec["abstain"] is False
    assert recalibrated["trace"][-1]["status"] == "PASS"

    # Also verify certified entries can lift an undersized n_calib=8 bucket across the n=9 rank threshold
    unlocked_rank = process_packet(
        {
            "knn_distance": 0.35,
            "knn_threshold": 0.758,
            "n_calib": 8,
            "alpha": 0.10,
            "certified_ledger_entries": [{"admit_to_memory": True, "ledger_entry_sha256": "c" * 64}],
        }
    )
    assert unlocked_rank["halted"] is False
    assert unlocked_rank["decision"]["effective_n_calib"] == 9
    assert unlocked_rank["decision"]["abstain"] is False


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


def test_collision_free_dynamic_load_when_community_extensions_already_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys
    import types

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
        assert res["decision"]["in_trust_region"] is True
        assert sys.modules["community_extensions"] is dummy_pkg
    finally:
        sys.modules.pop("_test_decide_wire_isolated", None)


def test_issue_28_sha256_validation_deduplication_and_round2_idempotence() -> None:
    single_run = {
        "run_id": "run-001",
        "admit_to_memory": True,
        "ledger_entry_sha256": "a" * 64,
        "calibration_residual": 0.25,
    }
    # Replaying 35 copies of one certified entry must be idempotent (226 -> 227, 0.758 -> 0.778)
    out_single = process_packet(
        {
            "knn_distance": 1.38,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [single_run],
        }
    )
    out_replay = process_packet(
        {
            "knn_distance": 1.38,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [single_run] * 35,
        }
    )
    assert out_replay["halted"] is True
    assert out_replay["halt_reason"] == "L6_ABSTAIN"
    assert out_replay["decision"]["abstain"] is True
    assert out_replay["decision"]["effective_n_calib"] == 227
    assert out_replay["decision"]["effective_threshold"] == pytest.approx(0.778)
    assert out_replay["decision"]["duplicate_ledger_entries_ignored"] == 34
    assert out_replay["decision"]["effective_n_calib"] == out_single["decision"]["effective_n_calib"]
    assert out_replay["decision"]["effective_threshold"] == pytest.approx(
        out_single["decision"]["effective_threshold"]
    )

    # Case-insensitive SHA-256 deduplication ("A"*64 vs "a"*64)
    out_case = process_packet(
        {
            "knn_distance": 0.770,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [
                {"run_id": "r-upper", "ledger_entry_sha256": "A" * 64},
                {"run_id": "r-lower", "ledger_entry_sha256": "a" * 64},
            ],
        }
    )
    assert out_case["halted"] is False
    assert out_case["decision"]["effective_n_calib"] == 227
    assert out_case["decision"]["effective_threshold"] == pytest.approx(0.778)
    assert out_case["decision"]["duplicate_ledger_entries_ignored"] == 1

    # Explicit admit_to_memory=True with unique run_id still supports Round-2 recalibration (226 -> 227, 0.758 -> 0.778)
    out_unique_id = process_packet(
        {
            "knn_distance": 0.770,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": [{"run_id": "round2-unique-001", "admit_to_memory": True}] * 5,
        }
    )
    assert out_unique_id["halted"] is False
    assert out_unique_id["decision"]["effective_n_calib"] == 227
    assert out_unique_id["decision"]["effective_threshold"] == pytest.approx(0.778)
    assert out_unique_id["decision"]["duplicate_ledger_entries_ignored"] == 4

    # Anonymous digest-free entries and malformed SHA-256 strings must be rejected
    out_malformed = process_packet(
        {
            "knn_distance": 0.30,
            "knn_threshold": 0.758,
            "n_calib": 5,
            "alpha": 0.10,
            "certified_ledger_entries": [
                {"admit_to_memory": True},
                {"admit_to_memory": True, "ledger_entry_sha256": "   "},
                {"admit_to_memory": True, "ledger_entry_sha256": "bad-hash", "run_id": "r1"},
                {"ledger_entry_sha256": "g" * 64},
                {"ledger_entry_sha256": "a" * 63},
                {"ledger_entry_sha256": "b" * 64, "calibration_residual": float("nan")},
            ],
        }
    )
    assert out_malformed["halted"] is True
    assert out_malformed["halt_reason"] == "L6_ABSTAIN"
    assert out_malformed["decision"]["effective_n_calib"] == 5
    assert out_malformed["decision"]["invalid_ledger_entries_ignored"] == 6

    # Bounded threshold expansion prevents even 35 distinct hashes from accepting deep OOD (1.38)
    distinct_35 = [{"ledger_entry_sha256": f"{i:064x}", "admit_to_memory": True} for i in range(35)]
    out_capped = process_packet(
        {
            "knn_distance": 1.38,
            "knn_threshold": 0.758,
            "n_calib": 226,
            "certified_ledger_entries": distinct_35,
        }
    )
    assert out_capped["halted"] is True
    assert out_capped["halt_reason"] == "L6_ABSTAIN"
    assert out_capped["decision"]["effective_threshold"] == pytest.approx(0.958)


def test_issue_24_27_29_wire_rank_parity_covariate_weights_and_descriptor_cqr_r() -> None:
    # Exact rank parity at n=149, alpha=0.18 (rank=123, not 124)
    out_rank = process_packet({"n_calib": 149, "alpha": 0.18, "knn_distance": 0.30})
    assert out_rank["halted"] is False
    assert "rank=123/149" in out_rank["trace"][-1]["summary"]

    # Descriptor-conditioned heteroscedastic CQR-r: low-MW ethanol narrower than 4-ring pyrene at equal knn_distance
    out_small = process_packet({"candidate_smiles": "CCO", "knn_distance": 0.40, "n_calib": 226})
    out_poly = process_packet(
        {"candidate_smiles": "c1cc2ccc3cccc4ccc(c1)c2c34", "knn_distance": 0.40, "n_calib": 226}
    )
    w_small = out_small["decision"]["conformal_interval"][1] - out_small["decision"]["conformal_interval"][0]
    w_poly = out_poly["decision"]["conformal_interval"][1] - out_poly["decision"]["conformal_interval"][0]
    assert 0.0 < w_small < w_poly

    # Likelihood-ratio downweighting of noisy outer-shell boundary entries prevents interval blow-up
    noisy_shell_entries = [
        {
            "run_id": f"shell-{i}",
            "ledger_entry_sha256": f"{1000 + i:064x}",
            "calibration_residual": 4.5,
            "likelihood_ratio_weight": 0.10,
        }
        for i in range(15)
    ]
    unweighted_entries = [
        {
            "run_id": f"shell-{i}",
            "ledger_entry_sha256": f"{1000 + i:064x}",
            "calibration_residual": 4.5,
            "likelihood_ratio_weight": 1.0,
        }
        for i in range(15)
    ]
    out_lw = process_packet({"candidate_smiles": "CCO", "n_calib": 60, "certified_ledger_entries": noisy_shell_entries})
    out_uw = process_packet({"candidate_smiles": "CCO", "n_calib": 60, "certified_ledger_entries": unweighted_entries})
    w_lw = out_lw["decision"]["conformal_interval"][1] - out_lw["decision"]["conformal_interval"][0]
    w_uw = out_uw["decision"]["conformal_interval"][1] - out_uw["decision"]["conformal_interval"][0]
    assert w_lw < w_uw
    assert out_lw["decision"]["conformal_interval"][0] <= out_lw["decision"]["conformal_interval"][1]


