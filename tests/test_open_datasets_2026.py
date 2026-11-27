"""Unit & integration tests for the 2026 OpenCalibrationDataset registry and schema loaders."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import pickle
from pathlib import Path
from typing import Any
from unittest import mock

import jsonschema
import numpy as np
import pandas as pd
import pytest

from lab_decide import (
    APPROVED_COMMERCIAL_SPDX_LICENSES,
    DECISION_ENVELOPE_JSON_SCHEMA,
    NEW_OPEN_DATASET_IDS_2026,
    OPEN_CALIBRATION_DATASETS,
    VALID_DOMAINS,
    ConformalTrustModel,
    OpenCalibrationDataset,
    TrustRegionAdvisor,
    build_decision_envelopes,
    calibrate_open_dataset,
    default_dataset_cache_path,
    fetch_open_dataset,
    fit_and_calibrate,
    get_open_dataset,
    list_open_datasets,
    load_open_dataset,
    make_schema_sample_records,
    write_schema_sample_fixture,
)
from lab_decide.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_script_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "fetch_open_datasets.py"
    spec = importlib.util.spec_from_file_location("_fetch_open_datasets_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_completeness_domains_and_license_metadata() -> None:
    assert len(NEW_OPEN_DATASET_IDS_2026) >= 10
    assert len(OPEN_CALIBRATION_DATASETS) == len(NEW_OPEN_DATASET_IDS_2026) + 1

    # Every scientific domain must have at least one registered open dataset
    for dom in VALID_DOMAINS:
        by_dom = list_open_datasets(domain=dom)
        assert len(by_dom) >= 1, f"Expected >= 1 dataset in domain {dom!r}"

    comm_only = list_open_datasets(commercial_only=True)
    assert len(comm_only) == len(OPEN_CALIBRATION_DATASETS)

    for ds_id, spec in OPEN_CALIBRATION_DATASETS.items():
        assert spec.dataset_id == ds_id
        assert spec.domain in VALID_DOMAINS
        assert len(spec.target_unit) > 0
        assert spec.n_records >= 600
        assert spec.url.startswith("https://")
        assert spec.landing_url.startswith("https://")
        assert len(spec.citation_doi) > 0
        assert spec.spdx_license in APPROVED_COMMERCIAL_SPDX_LICENSES
        assert spec.commercial_use_ok is True
        assert "NC" not in spec.spdx_license.upper()
        assert spec.domain_bounds is not None
        assert spec.domain_bounds[0] < spec.domain_bounds[1]

        cache_path = Path(default_dataset_cache_path(ds_id))
        assert cache_path.parent == (REPO_ROOT / "data")
        assert get_open_dataset(ds_id) == spec

        as_dict = spec.to_dict()
        assert as_dict["dataset_id"] == ds_id
        assert as_dict["spdx_license"] == spec.spdx_license
        assert as_dict["commercial_use_ok"] is True


def test_open_calibration_dataset_validation_guards() -> None:
    with pytest.raises(KeyError, match="Unknown dataset_id"):
        get_open_dataset("nonexistent_dataset_xyz")

    with pytest.raises(ValueError, match="Unknown domain"):
        list_open_datasets(domain="quantum_gravity")

    with pytest.raises(ValueError, match="Invalid dataset_id"):
        OpenCalibrationDataset(
            dataset_id="Bad-ID!",
            domain="admet",
            target_unit="kcal/mol",
            n_records=100,
            url="https://example.org/data.csv",
            spdx_license="MIT",
            commercial_use_ok=True,
        )

    with pytest.raises(ValueError, match="Non-commercial SPDX license"):
        OpenCalibrationDataset(
            dataset_id="nc_test",
            domain="admet",
            target_unit="kcal/mol",
            n_records=100,
            url="https://example.org/data.csv",
            spdx_license="CC-BY-NC-4.0",
            commercial_use_ok=True,
        )

    with pytest.raises(ValueError, match="HTTPS URL"):
        OpenCalibrationDataset(
            dataset_id="http_insecure",
            domain="admet",
            target_unit="kcal/mol",
            n_records=100,
            url="http://example.org/data.csv",
            spdx_license="MIT",
            commercial_use_ok=True,
        )

    with pytest.raises(ValueError, match="domain_bounds"):
        OpenCalibrationDataset(
            dataset_id="bad_bounds",
            domain="admet",
            target_unit="kcal/mol",
            n_records=100,
            url="https://example.org/data.csv",
            spdx_license="MIT",
            commercial_use_ok=True,
            domain_bounds=(10.0, -5.0),
        )


@pytest.mark.parametrize("dataset_id", list(OPEN_CALIBRATION_DATASETS.keys()))
def test_end_to_end_schema_loaders_csv_jsonl_parity_and_conformal_gating(
    dataset_id: str,
    tmp_path: Path,
) -> None:
    spec = get_open_dataset(dataset_id)
    native_path = tmp_path / f"{dataset_id}.{spec.file_format}"
    jsonl_path = tmp_path / f"{dataset_id}.jsonl"

    write_schema_sample_fixture(dataset_id, native_path, n=240, seed=0)
    write_schema_sample_fixture(dataset_id, jsonl_path, fmt="jsonl", n=240, seed=0)

    bundle_native = load_open_dataset(dataset_id, path=native_path)
    bundle_jsonl = load_open_dataset(dataset_id, path=jsonl_path, fmt="jsonl")

    # Deterministic CSV/TSV/JSON vs JSONL schema loader parity
    assert bundle_native.feature_names == bundle_jsonl.feature_names
    assert bundle_native.target_name == bundle_jsonl.target_name
    assert bundle_native.X.shape == bundle_jsonl.X.shape
    assert np.allclose(bundle_native.X, bundle_jsonl.X, atol=1e-8)
    assert np.allclose(bundle_native.y, bundle_jsonl.y, atol=1e-8)

    # End-to-end conformal calibration + Mondrian trust-region gating + constrained BO
    advisor, bundle, splits, metrics = calibrate_open_dataset(
        dataset_id,
        path=native_path,
        alpha=0.10,
        trust_pct=80.0,
        nonconformity="normalized",
        seed=0,
    )
    assert isinstance(advisor, ConformalTrustModel)
    assert math.isfinite(advisor.q_in) and advisor.q_in >= 0.0
    assert math.isfinite(advisor.threshold) and advisor.threshold > 0.0

    # Verify disjoint 4-way splits
    idx_sets = [
        set(splits.train_idx.tolist()),
        set(splits.pool_idx.tolist()),
        set(splits.cal_idx.tolist()),
        set(splits.test_idx.tolist()),
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            assert idx_sets[i].isdisjoint(idx_sets[j])

    # Inside vs outside trust-region gating invariants
    assert 0.05 < metrics["answered_fraction"] < 1.0
    assert metrics["cov_in"] >= 0.78
    assert metrics["mae_in"] < metrics["mae_out"]
    assert math.isfinite(metrics["winkler_in"])

    res_te = advisor.predict(bundle.X[splits.test_idx], use_mondrian=True)
    assert np.all(res_te["lo"] <= res_te["pred"] + 1e-12)
    assert np.all(res_te["pred"] <= res_te["hi"] + 1e-12)
    if spec.domain_bounds is not None:
        b_lo, b_hi = spec.domain_bounds
        assert np.all(res_te["lo"] >= b_lo - 1e-12)
        assert np.all(res_te["hi"] <= b_hi + 1e-12)

    # Explicit extreme out-of-domain probe must always abstain
    extreme_ood = advisor.scaler_mean + 15.0 * advisor.scaler_scale
    ood_pred = advisor.predict(extreme_ood)
    assert ood_pred["trusted"] is False
    assert ood_pred["decision"] == "ABSTAIN_ACQUIRE"
    assert ood_pred["distance"] > advisor.threshold

    # ConformalTrustModel pickle round-trip & TrustRegionAdvisor constrained proposals
    restored: ConformalTrustModel = pickle.loads(pickle.dumps(advisor))
    assert restored.predict(bundle.X[0])["pred"] == pytest.approx(
        advisor.predict(bundle.X[0])["pred"]
    )
    bo_props = TrustRegionAdvisor(advisor).propose(
        bundle.X[splits.pool_idx], mode="conformal_ucb", batch_size=4
    )
    assert len(bo_props) == 4
    assert all(p.trusted for p in bo_props)

    # DecisionEnvelope RFC 8259 JSON Schema compliance
    envs = build_decision_envelopes(
        advisor,
        bundle.X[splits.test_idx[:3]],
        candidate_ids=bundle.compound_ids[:3],
        smiles=bundle.smiles[:3],
    )
    for env in envs:
        jsonschema.validate(instance=env.to_dict(), schema=DECISION_ENVELOPE_JSON_SCHEMA)


@pytest.mark.parametrize(
    "dataset_id",
    [
        ds_id
        for ds_id, s in OPEN_CALIBRATION_DATASETS.items()
        if len(s.leakage_excluded_columns) > 0
    ],
)
def test_zero_target_leakage_from_excluded_columns(dataset_id: str, tmp_path: Path) -> None:
    spec = get_open_dataset(dataset_id)
    recs_clean = make_schema_sample_records(dataset_id, n=40, seed=3)
    recs_poisoned = []
    for rec in recs_clean:
        copy_rec = dict(rec)
        for leak_col in spec.leakage_excluded_columns:
            if leak_col in copy_rec:
                copy_rec[leak_col] = 9.99e11
        recs_poisoned.append(copy_rec)

    p_clean = tmp_path / f"{dataset_id}_clean.jsonl"
    p_poison = tmp_path / f"{dataset_id}_poison.jsonl"
    p_clean.write_text("\n".join(json.dumps(r) for r in recs_clean) + "\n", encoding="utf-8")
    p_poison.write_text("\n".join(json.dumps(r) for r in recs_poisoned) + "\n", encoding="utf-8")

    b_clean = load_open_dataset(dataset_id, path=p_clean, fmt="jsonl")
    b_poison = load_open_dataset(dataset_id, path=p_poison, fmt="jsonl")

    for leak_col in spec.leakage_excluded_columns:
        assert leak_col not in b_clean.feature_names
    assert np.array_equal(b_clean.X, b_poison.X)
    assert np.array_equal(b_clean.y, b_poison.y)


def test_schema_validation_nan_filtering_dedup_and_suzuki_percentage_scaling(
    tmp_path: Path,
) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(bad_csv, index=False)
    for ds_id in NEW_OPEN_DATASET_IDS_2026:
        with pytest.raises(ValueError):
            load_open_dataset(ds_id, path=bad_csv, fmt="csv")

    # Verify NaN dropping and replicate SMILES deduplication on lipophilicity_astrazeneca
    recs = make_schema_sample_records("lipophilicity_astrazeneca", n=30, seed=1)
    dup = dict(recs[0])
    dup["exp"] = round(float(recs[0]["exp"]) + 1.0, 3)
    bad_nan = dict(recs[1])
    bad_nan["exp"] = float("nan")
    mixed_path = tmp_path / "lipo_mixed.csv"
    pd.DataFrame([*recs, dup, bad_nan]).to_csv(mixed_path, index=False)

    bundle = load_open_dataset("lipophilicity_astrazeneca", path=mixed_path)
    assert bundle.dropped_nonfinite_count == 1
    assert bundle.deduplicated_count == 1
    assert len(bundle.y) == 30
    assert bundle.y[0] == pytest.approx(float(recs[0]["exp"]) + 0.5, abs=1e-3)

    # Verify Suzuki-Miyaura accepts both fractional [0, 1] and percentage [0, 100] yields
    sz_recs = make_schema_sample_records("suzuki_miyaura_perera", n=24, seed=2)
    sz_pct_recs = [{**r, "y": round(100.0 * float(r["y"]), 3)} for r in sz_recs]
    p_frac = tmp_path / "sz_frac.tsv"
    p_pct = tmp_path / "sz_pct.tsv"
    pd.DataFrame(sz_recs).to_csv(p_frac, sep="\t", index=False)
    pd.DataFrame(sz_pct_recs).to_csv(p_pct, sep="\t", index=False)

    b_frac = load_open_dataset("suzuki_miyaura_perera", path=p_frac, fmt="tsv")
    b_pct = load_open_dataset("suzuki_miyaura_perera", path=p_pct, fmt="tsv")
    assert np.allclose(b_frac.y, b_pct.y, atol=1e-2)

    # Verify Biogen ADME supports switching target_column to solubility endpoint
    bg_recs = make_schema_sample_records("biogen_adme", n=32, seed=4)
    bg_path = tmp_path / "biogen.csv"
    pd.DataFrame(bg_recs).to_csv(bg_path, index=False)
    b_sol = load_open_dataset(
        "biogen_adme",
        path=bg_path,
        target_column="LOG SOLUBILITY PH 6.8 (ug/mL)",
    )
    assert b_sol.target_name == "LOG SOLUBILITY PH 6.8 (ug/mL)"
    adv_bg = fit_and_calibrate(b_sol.X[:20], b_sol.y[:20], b_sol.X[20:], b_sol.y[20:])
    assert math.isfinite(adv_bg.q_in)


def test_fetch_open_datasets_atomic_download_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fetch_mod = _load_fetch_script_module()

    # 1. --list --json outputs all registered datasets
    assert fetch_mod.main(["--list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == len(OPEN_CALIBRATION_DATASETS)

    # 2. Offline load without cache raises FileNotFoundError
    with mock.patch("lab_decide.datasets.DEFAULT_OPEN_DATA_DIR", str(tmp_path / "empty_cache")):
        with pytest.raises(FileNotFoundError, match="offline=True"):
            load_open_dataset("freesolv", path=None, offline=True)
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_open_dataset("freesolv", path=tmp_path / "missing.json", offline=True)

    # 3. Atomic fetch with SHA-256 verification and --calibrate via fetch_open_datasets.py
    sample_fs_path = tmp_path / "src_freesolv.json"
    write_schema_sample_fixture("freesolv", sample_fs_path, n=80, seed=0)
    raw_bytes = sample_fs_path.read_bytes()
    real_sha = hashlib.sha256(raw_bytes).hexdigest()

    class _FakeResp(io.BytesIO):
        status = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    patched_spec = OpenCalibrationDataset(
        **{
            **OPEN_CALIBRATION_DATASETS["freesolv"].to_dict(),
            "domain_bounds": (-30.0, 5.0),
            "leakage_excluded_columns": OPEN_CALIBRATION_DATASETS["freesolv"].leakage_excluded_columns,
            "expected_sha256": real_sha,
        }
    )
    dl_dir = tmp_path / "cache_data"
    with mock.patch.dict(OPEN_CALIBRATION_DATASETS, {"freesolv": patched_spec}):
        with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _FakeResp(raw_bytes)):
            rc_check = fetch_mod.main(["--check-urls", "--dataset", "freesolv", "--json"])
            assert rc_check == 0
            chk_out = json.loads(capsys.readouterr().out)
            assert chk_out[0]["ok"] is True

            rc_fetch = fetch_mod.main(
                [
                    "--dataset",
                    "freesolv",
                    "--data-dir",
                    str(dl_dir),
                    "--calibrate",
                    "--json",
                ]
            )
            assert rc_fetch == 0
            fetched_out = json.loads(capsys.readouterr().out)
            assert fetched_out[0]["dataset_id"] == "freesolv"
            assert Path(fetched_out[0]["local_path"]).is_file()
            assert fetched_out[0]["calibration"]["cov_in"] >= 0.75

        # Corrupt SHA-256 stream must raise ValueError and leave zero .part files
        bad_dir = tmp_path / "bad_cache"
        with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _FakeResp(b"corrupted")):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                fetch_open_dataset("freesolv", data_dir=bad_dir, force=True)
        assert list(bad_dir.glob("*.part*")) == []

    # 4. Main lab-decide CLI --offline fails closed without cache unless --allow-synthetic is given
    with mock.patch(
        "lab_decide.cli.default_dataset_cache_path",
        return_value=str(tmp_path / "uncached_aqsoldb.csv"),
    ):
        rc_fail = cli_main(["--dataset", "aqsoldb", "--offline", "--json"])
        assert rc_fail != 0
        capsys.readouterr()

        rc_cli = cli_main(["--dataset", "aqsoldb", "--offline", "--allow-synthetic", "--json"])
        assert rc_cli == 0
        cli_payload = json.loads(capsys.readouterr().out)
        assert cli_payload["dataset_id"] == "aqsoldb"
        assert cli_payload["spdx_license"] == "CC0-1.0"
        assert cli_payload["data_source"] == "synthetic_fixture"
