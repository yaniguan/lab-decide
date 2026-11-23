"""Safe atomic dataset download, schema/NaN validation, SMILES deduplication, and 4-way splits."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from lab_decide.chemistry import (
    ESOL_FEATURES,
    ESOL_TARGET,
    compute_scaffold_groups,
    normalize_smiles_str,
)

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ESOL_SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"

DEFAULT_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "delaney-processed.csv")
)


@dataclass(frozen=True)
class DatasetBundle:
    """Validated tabular feature matrix, target vector, SMILES provenance, and cleaning counts."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    smiles: list[str]
    compound_ids: list[str]
    raw_row_count: int
    dropped_nonfinite_count: int
    deduplicated_count: int


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint 4-way indices separating training, candidate acquisition pool, calibration, and test."""

    train_idx: np.ndarray
    pool_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


def compute_file_sha256(path: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file_atomic(
    url: str,
    dest_path: str,
    *,
    expected_sha256: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Download `url` to `dest_path + '.part'`, verify optional SHA-256, and atomically rename."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    part_path = f"{dest_path}.part.{os.getpid()}"
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        digest = hasher.hexdigest()
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}"
            )
        os.replace(part_path, dest_path)
        return digest
    except Exception:
        if os.path.exists(part_path):
            try:
                os.unlink(part_path)
            except OSError:
                pass
        raise


def ensure_esol_csv(
    path: str = DEFAULT_DATA_PATH,
    *,
    url: str = ESOL_URL,
    expected_sha256: str | None = ESOL_SHA256,
    timeout: float = 30.0,
    offline: bool = False,
) -> str:
    """Ensure ESOL CSV exists locally with verified SHA-256; unlink corrupt partial caches."""
    if os.path.exists(path):
        if expected_sha256 is not None:
            actual = compute_file_sha256(path)
            if actual.lower() != expected_sha256.lower():
                try:
                    os.unlink(path)
                except OSError:
                    pass
                if offline:
                    raise RuntimeError(
                        f"Cached file {path} failed SHA-256 check and was removed; "
                        "re-run without --offline to fetch a clean copy."
                    )
            else:
                return path
        else:
            return path

    if offline:
        raise FileNotFoundError(
            f"Dataset {path} is not cached locally and --offline mode is enabled."
        )

    download_file_atomic(url, path, expected_sha256=expected_sha256, timeout=timeout)
    return path


def make_synthetic_solubility_dataframe(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Create a deterministic offline solubility dataset with realistic 6D descriptors and OOD tail."""
    rng = np.random.default_rng(seed)
    is_ood_tail = rng.random(n) < 0.18
    min_deg = rng.choice([1.0, 1.0, 1.0, 1.0, 2.0], size=n)
    rings = np.where(is_ood_tail, rng.integers(4, 7, size=n), rng.integers(0, 3, size=n)).astype(float)
    mw = np.where(
        is_ood_tail,
        380.0 + 45.0 * rings + rng.uniform(20.0, 160.0, size=n),
        90.0 + 40.0 * rings + rng.uniform(10.0, 140.0, size=n),
    )
    hbd = rng.integers(0, 5, size=n).astype(float)
    rot = np.where(is_ood_tail, rng.integers(6, 13, size=n), rng.integers(0, 6, size=n)).astype(float)
    tpsa = 18.0 * hbd + rng.uniform(0.0, 75.0, size=n)

    # Physical trend + nonlinear polycyclic packing penalty & heteroscedastic OOD variance
    packing_shift = np.where(is_ood_tail, -1.35 * np.sin(mw / 50.0), 0.0)
    noise_scale = np.where(is_ood_tail, 1.15, 0.22)
    y = (
        0.85
        - 0.012 * mw
        - 0.35 * rings
        + 0.014 * tpsa
        - 0.05 * rot
        + packing_shift
        + rng.normal(0.0, noise_scale)
    )
    y = np.clip(y, -11.5, 1.55)

    smiles = []
    for i in range(n):
        r = int(rings[i])
        chain = "C" * max(1, int(mw[i] // 45))
        ring_part = "c1ccccc1" * min(r, 2)
        oh = "O" * min(int(hbd[i]), 2)
        smiles.append(f"{chain}{ring_part}{oh}_{i}" if not ring_part else f"{chain}{ring_part}{oh}")
    # Ensure unique synthetic SMILES strings per compound index via alkyl tail length + branch
    unique_smiles = [f"{'C' * (i % 7 + 1)}{'c1ccccc1' if rings[i] > 0 else 'CCO'}{'O' * (i // 7)}" for i in range(n)]

    return pd.DataFrame(
        {
            "Compound ID": [f"SYN-{i:04d}" for i in range(n)],
            "Minimum Degree": min_deg,
            "Molecular Weight": np.round(mw, 3),
            "Number of H-Bond Donors": hbd,
            "Number of Rings": rings,
            "Number of Rotatable Bonds": rot,
            "Polar Surface Area": np.round(tpsa, 2),
            ESOL_TARGET: np.round(y, 4),
            "smiles": unique_smiles,
        }
    )


def validate_and_clean_dataframe(
    df: pd.DataFrame,
    *,
    features: Sequence[str] = ESOL_FEATURES,
    target: str = ESOL_TARGET,
    deduplicate_smiles: bool = True,
    canonicalize_smiles: bool = False,
    min_rows: int = 4,
) -> DatasetBundle:
    """Validate schema, filter non-finite rows, and deduplicate replicate SMILES."""
    feature_list = list(features)
    if not feature_list:
        raise ValueError("features list must be non-empty")

    required = feature_list + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}; available columns: {list(df.columns)}"
        )

    raw_count = len(df)
    work = df.copy()

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    num_mat = work[required].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(num_mat).all(axis=1)
    dropped_nonfinite = int((~finite_mask).sum())
    work = work.loc[finite_mask].reset_index(drop=True)

    dedup_count = 0
    if "smiles" in work.columns:
        cleaned_smiles = [
            normalize_smiles_str(str(s), canonicalize=canonicalize_smiles)
            for s in work["smiles"].tolist()
        ]
        work["smiles"] = cleaned_smiles
        if deduplicate_smiles:
            before = len(work)
            agg_spec: dict[str, str] = {col: "mean" for col in required}
            if "Compound ID" in work.columns:
                agg_spec["Compound ID"] = "first"
            # Preserve first-seen deterministic row order across pandas versions
            first_order = work.drop_duplicates(subset=["smiles"], keep="first")["smiles"].tolist()
            grouped = work.groupby("smiles", sort=False, as_index=False).agg(agg_spec)
            grouped = grouped.set_index("smiles").loc[first_order].reset_index()
            dedup_count = before - len(grouped)
            work = grouped

    if len(work) < min_rows:
        raise ValueError(
            f"Dataset contains only {len(work)} valid finite rows after cleaning; "
            f"at least {min_rows} rows are required."
        )

    X = work[feature_list].to_numpy(dtype=np.float64)
    y = work[target].to_numpy(dtype=np.float64)
    smiles_out = (
        work["smiles"].astype(str).tolist()
        if "smiles" in work.columns
        else [f"row-{i}" for i in range(len(work))]
    )
    comp_ids = (
        work["Compound ID"].astype(str).tolist()
        if "Compound ID" in work.columns
        else [f"cand-{i:04d}" for i in range(len(work))]
    )
    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_list,
        target_name=target,
        smiles=smiles_out,
        compound_ids=comp_ids,
        raw_row_count=raw_count,
        dropped_nonfinite_count=dropped_nonfinite,
        deduplicated_count=dedup_count,
    )


def load_dataset(
    csv_path: str | None = None,
    *,
    features: Sequence[str] = ESOL_FEATURES,
    target: str = ESOL_TARGET,
    deduplicate_smiles: bool = True,
    canonicalize_smiles: bool = False,
    offline: bool = False,
    min_rows: int = 4,
) -> DatasetBundle:
    """Load ESOL (or a custom CSV / synthetic offline fallback) into a validated DatasetBundle."""
    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        if offline and not os.path.exists(DEFAULT_DATA_PATH):
            df = make_synthetic_solubility_dataframe(n=240, seed=0)
        else:
            resolved = ensure_esol_csv(DEFAULT_DATA_PATH, offline=offline)
            df = pd.read_csv(resolved)

    return validate_and_clean_dataframe(
        df,
        features=features,
        target=target,
        deduplicate_smiles=deduplicate_smiles,
        canonicalize_smiles=canonicalize_smiles,
        min_rows=min_rows,
    )


def load(
    csv_path: str | None = None,
    *,
    offline: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible loader returning `(X, y)` with schema check & SMILES deduplication."""
    bundle = load_dataset(csv_path=csv_path, offline=offline)
    return bundle.X, bundle.y


def partition_splits(
    n_samples: int,
    *,
    smiles: Sequence[str] | None = None,
    strategy: str = "random",
    seed: int = 0,
    fracs: tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15),
) -> SplitIndices:
    """Partition dataset into 4 strictly disjoint splits: `(train, pool, calibration, test)`.

    Supports `'random'` and `'scaffold'` (Bemis-Murcko scaffold group-disjoint splitting).
    """
    if n_samples < 8:
        raise ValueError(f"Need at least 8 samples for 4-way train/pool/cal/test split, got {n_samples}")
    if not np.isclose(sum(fracs), 1.0, atol=1e-6) or any(f <= 0 for f in fracs):
        raise ValueError(f"Split fractions must be positive and sum to 1.0, got {fracs}")

    f_tr, f_pool, f_cal, f_te = fracs
    idx = np.arange(n_samples, dtype=int)

    if strategy == "random":
        tr_idx, rem1 = train_test_split(idx, test_size=(1.0 - f_tr), random_state=seed)
        rel_pool = f_pool / (f_pool + f_cal + f_te)
        pool_idx, rem2 = train_test_split(rem1, train_size=rel_pool, random_state=seed)
        rel_te = f_te / (f_cal + f_te)
        cal_idx, te_idx = train_test_split(rem2, test_size=rel_te, random_state=seed)
        return SplitIndices(
            train_idx=np.sort(tr_idx),
            pool_idx=np.sort(pool_idx),
            cal_idx=np.sort(cal_idx),
            test_idx=np.sort(te_idx),
        )

    if strategy == "scaffold":
        if smiles is None or len(smiles) != n_samples:
            raise ValueError("smiles sequence of length n_samples is required when strategy='scaffold'")
        groups = compute_scaffold_groups(smiles)
        unique_groups = np.unique(groups)
        if unique_groups.size < 4:
            raise ValueError(
                f"Scaffold split requires at least 4 distinct scaffolds, found {unique_groups.size}"
            )

        gss1 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - f_tr), random_state=seed)
        tr_pos, rem1_pos = next(gss1.split(idx, groups=groups))
        tr_idx = idx[tr_pos]
        rem1 = idx[rem1_pos]

        rel_pool = f_pool / (f_pool + f_cal + f_te)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_pool, random_state=seed + 1)
        pool_pos, rem2_pos = next(gss2.split(rem1, groups=groups[rem1]))
        pool_idx = rem1[pool_pos]
        rem2 = rem1[rem2_pos]

        rel_te = f_te / (f_cal + f_te)
        gss3 = GroupShuffleSplit(n_splits=1, test_size=rel_te, random_state=seed + 2)
        cal_pos, te_pos = next(gss3.split(rem2, groups=groups[rem2]))
        cal_idx = rem2[cal_pos]
        te_idx = rem2[te_pos]

        return SplitIndices(
            train_idx=np.sort(tr_idx),
            pool_idx=np.sort(pool_idx),
            cal_idx=np.sort(cal_idx),
            test_idx=np.sort(te_idx),
        )

    raise ValueError(f"Unknown split strategy {strategy!r}; expected 'random' or 'scaffold'")
