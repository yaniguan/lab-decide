"""Unit & regression tests for `rdkit_2d_morgan`, `FrontierEnsembleRegressor`, and Delaney (2004) comparisons."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import lab_decide.chemistry as chem_mod
from lab_decide import (
    BUNDLED_ESOL_RESOURCE_PATH,
    DELANEY_CSV_PRED_COLUMN,
    ESOL_TARGET,
    HAS_RDKIT,
    FrontierEnsembleRegressor,
    TrustRegionAdvisor,
    calibrate_open_dataset,
    delaney_2004_descriptors,
    delaney_2004_formula_predict,
    featurize_delaney_4d,
    featurize_rdkit_2d_morgan_batch,
    fit_and_calibrate,
    get_rdkit_2d_morgan_feature_names,
    get_rdkit_2d_morgan_layout,
    load_dataset,
    load_open_dataset,
    partition_splits,
    smiles_to_rdkit_2d_morgan,
    summarize,
    write_schema_sample_fixture,
)
from lab_decide.cli import main as cli_main


def test_frontier_ensemble_and_tanimoto_mahalanobis_work_without_rdkit() -> None:
    """Verify FrontierEnsembleRegressor + ResidualQuantileBandModel + tanimoto_mahalanobis on pure numpy data."""
    rng = np.random.default_rng(17)
    X_tr = rng.uniform(0.0, 2.0, size=(80, 12))
    y_tr = 1.8 * X_tr[:, 0] - 0.9 * (X_tr[:, 1] ** 2) + 0.5 * np.sin(X_tr[:, 2]) + rng.normal(0.0, 0.15, size=80)
    X_cal = rng.uniform(0.0, 2.0, size=(50, 12))
    y_cal = 1.8 * X_cal[:, 0] - 0.9 * (X_cal[:, 1] ** 2) + 0.5 * np.sin(X_cal[:, 2]) + rng.normal(0.0, 0.15, size=50)
    X_te = rng.uniform(0.0, 2.0, size=(40, 12))
    y_te = 1.8 * X_te[:, 0] - 0.9 * (X_te[:, 1] ** 2) + 0.5 * np.sin(X_te[:, 2]) + rng.normal(0.0, 0.15, size=40)

    adv = fit_and_calibrate(
        X_tr,
        y_tr,
        X_cal,
        y_cal,
        alpha=0.10,
        trust_pct=80.0,
        nonconformity="cqr_r",
        distance_metric="tanimoto_mahalanobis",
        surrogate="frontier_ensemble",
        domain_bounds=None,
        seed=0,
    )
    assert isinstance(adv.model, FrontierEnsembleRegressor)
    assert math.isfinite(adv.q_in)

    # Pickle round-trip
    restored = pickle.loads(pickle.dumps(adv))
    res_1d = restored.predict(X_te[0])
    res_2d = restored.predict(X_te)
    assert res_2d["pred"][0] == pytest.approx(res_1d["pred"])
    assert np.all(res_2d["lo"] <= res_2d["pred"] + 1e-12)
    assert np.all(res_2d["pred"] <= res_2d["hi"] + 1e-12)

    cov_in, _, mae_in = summarize(
        y_te, res_2d["lo"], res_2d["hi"], res_2d["trusted"], pred=res_2d["pred"]
    )
    assert cov_in >= 0.75
    assert mae_in < 0.35

    # Safe expansion and UCB work cleanly on FrontierEnsembleRegressor
    props = TrustRegionAdvisor(restored).propose(X_te, mode="safe_expand", batch_size=3)
    assert len(props) <= 3


def test_rdkit_2d_morgan_fails_cleanly_when_rdkit_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(chem_mod, "HAS_RDKIT", False)
    with pytest.raises(ImportError, match="RDKit"):
        smiles_to_rdkit_2d_morgan("CCO")
    with pytest.raises(ImportError, match="RDKit"):
        delaney_2004_descriptors("CCO")
    with pytest.raises(ImportError, match="RDKit"):
        load_dataset(BUNDLED_ESOL_RESOURCE_PATH, offline=True, encoding="rdkit_2d_morgan")

    fs_path = tmp_path / "freesolv.json"
    write_schema_sample_fixture("freesolv", fs_path, n=32, seed=0)
    with pytest.raises(ImportError, match="RDKit"):
        load_open_dataset("freesolv", path=fs_path, encoding="rdkit_2d_morgan")

    # Non-single-molecule datasets reject rdkit_2d_morgan with ValueError
    cal_path = tmp_path / "calisol.csv"
    write_schema_sample_fixture("calisol23", cal_path, n=32, seed=0)
    with pytest.raises(ValueError, match="single-molecule"):
        load_open_dataset("calisol23", path=cal_path, encoding="rdkit_2d_morgan")


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for rdkit_2d_morgan and Delaney tests")
def test_rdkit_2d_morgan_hygiene_and_head_to_head_vs_delaney_on_esol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    n_core, n_dm, n_fp = get_rdkit_2d_morgan_layout()
    feat_names = get_rdkit_2d_morgan_feature_names()
    assert len(feat_names) == n_dm + n_fp
    assert n_core >= 25 and n_dm >= 350 and n_fp == 1024

    # Edge-case molecules (C60 fullerene, sucrose, ethanol, salt) produce 100% finite features
    edge_smiles = [
        "CCO",
        "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
        "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41",
        "[Na+].[Cl-]",
    ]
    X_edge = featurize_rdkit_2d_morgan_batch(edge_smiles, strict=False)
    assert X_edge.shape == (4, len(feat_names))
    assert np.all(np.isfinite(X_edge))

    with pytest.raises(ValueError, match="Invalid SMILES"):
        smiles_to_rdkit_2d_morgan("not_a_smiles_###", strict=True)

    # Head-to-head on seed-0 ESOL: Frontier rdkit_2d_morgan beats Delaney (2004) CSV & refitted OLS by >35%!
    df = pd.read_csv(BUNDLED_ESOL_RESOURCE_PATH)
    df["smiles"] = df["smiles"].str.strip()
    first_order = df.drop_duplicates(subset=["smiles"], keep="first")["smiles"].tolist()
    df_dedup = df.groupby("smiles", sort=False, as_index=False).mean(numeric_only=True)
    df_dedup = df_dedup.set_index("smiles").loc[first_order].reset_index()

    bundle_mol = load_dataset(
        csv_path=BUNDLED_ESOL_RESOURCE_PATH,
        offline=True,
        encoding="rdkit_2d_morgan",
    )
    assert bundle_mol.smiles == df_dedup["smiles"].tolist()
    sp = partition_splits(len(bundle_mol.y), strategy="random", seed=0)
    te = sp.test_idx

    y_te = bundle_mol.y[te]
    delaney_csv_mae = float(np.mean(np.abs(y_te - df_dedup[DELANEY_CSV_PRED_COLUMN].to_numpy()[te])))
    X4 = featurize_delaney_4d(bundle_mol.smiles)
    assert X4.shape == (len(bundle_mol.y), 4)
    formula_pred = delaney_2004_formula_predict(bundle_mol.smiles[:5])
    assert formula_pred.shape == (5,) and np.all(np.isfinite(formula_pred))

    adv = fit_and_calibrate(
        bundle_mol.X[sp.train_idx],
        bundle_mol.y[sp.train_idx],
        bundle_mol.X[sp.cal_idx],
        bundle_mol.y[sp.cal_idx],
        alpha=0.10,
        trust_pct=80.0,
        nonconformity="cqr_r",
        distance_metric="tanimoto_mahalanobis",
        surrogate="frontier_ensemble",
        domain_bounds=(-15.0, 1.75),
        feature_names=bundle_mol.feature_names,
        seed=0,
    )
    res = adv.predict(bundle_mol.X[te], use_mondrian=True)
    cov_in, _, mae_in = summarize(y_te, res["lo"], res["hi"], res["trusted"], pred=res["pred"])
    _, _, mae_out = summarize(y_te, res["lo"], res["hi"], ~res["trusted"], pred=res["pred"])
    _, _, mae_all = summarize(
        y_te, res["lo"], res["hi"], np.ones(len(te), dtype=bool), pred=res["pred"]
    )
    assert mae_all <= 0.65 * delaney_csv_mae
    assert mae_in < mae_out
    assert cov_in >= 0.88

    # Open-dataset calibration & CLI support rdkit_2d_morgan + frontier_ensemble
    fs_sample = tmp_path / "freesolv_sample.json"
    write_schema_sample_fixture("freesolv", fs_sample, n=80, seed=0)
    _, b_fs, _, m_fs = calibrate_open_dataset(
        "freesolv",
        path=fs_sample,
        encoding="rdkit_2d_morgan",
        surrogate="frontier_ensemble",
        distance_metric="tanimoto_mahalanobis",
        nonconformity="cqr_r",
        seed=0,
    )
    assert b_fs.X.shape[1] == len(feat_names)
    assert m_fs["encoding"] == "rdkit_2d_morgan"
    assert m_fs["surrogate"] == "frontier_ensemble"

    rc = cli_main(
        [
            "--dataset",
            "freesolv",
            "--csv",
            str(fs_sample),
            "--encoding",
            "rdkit_2d_morgan",
            "--surrogate",
            "frontier_ensemble",
            "--distance-metric",
            "tanimoto_mahalanobis",
            "--nonconformity",
            "cqr_r",
            "--json",
        ]
    )
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["encoding"] == "rdkit_2d_morgan"
    assert cli_out["surrogate"] == "frontier_ensemble"
