"""Safe atomic dataset download, schema/NaN validation, SMILES deduplication, and balanced 4-way splits."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import lab_decide.chemistry as _chem_mod
from lab_decide.chemistry import (
    ESOL_FEATURES,
    ESOL_TARGET,
    compute_scaffold_groups,
    featurize_rdkit_2d_morgan_batch,
    get_rdkit_2d_morgan_feature_names,
    normalize_smiles_str,
)

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ESOL_SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"

DEFAULT_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "delaney-processed.csv")
)
BUNDLED_ESOL_RESOURCE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "delaney-processed.csv")
)


@dataclass(frozen=True)
class DatasetBundle:
    """Validated tabular feature matrix, target vector, SMILES provenance, group keys, and data source."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    smiles: list[str]
    compound_ids: list[str]
    raw_row_count: int
    dropped_nonfinite_count: int
    deduplicated_count: int
    group_keys: list[str] | None = None
    group_kind: str = "murcko_scaffold"
    data_source: str = "real_csv"


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint 4-way indices separating training, candidate acquisition pool, calibration, and test."""

    train_idx: np.ndarray
    pool_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray
    n_groups_total: int | None = None
    group_counts: dict[str, int] | None = None
    achieved_fractions: dict[str, float] | None = None


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
            f"Dataset {path} is not cached locally and --offline mode is enabled "
            "(pass --allow-synthetic to run on the deterministic synthetic fixture)."
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

    unique_smiles = [
        f"{'C' * (i % 7 + 1)}{'c1ccccc1' if rings[i] > 0 else 'CCO'}{'O' * (i // 7)}"
        for i in range(n)
    ]

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
    data_source: str = "real_csv",
    group_column: str | None = None,
    group_kind: str = "murcko_scaffold",
) -> DatasetBundle:
    """Validate schema, filter non-finite rows, deduplicate replicate SMILES, and attach group keys."""
    if data_source not in ("real_csv", "synthetic_fixture"):
        raise ValueError(f"data_source must be 'real_csv' or 'synthetic_fixture', got {data_source!r}")
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
            if group_column is not None and group_column in work.columns:
                agg_spec[group_column] = "first"
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

    group_keys_out: list[str] | None = None
    if group_column is not None and group_column in work.columns:
        group_keys_out = [str(v).strip() for v in work[group_column].tolist()]
    elif "smiles" in work.columns and _chem_mod.HAS_RDKIT:
        try:
            group_keys_out = compute_scaffold_groups(smiles_out).tolist()
        except ValueError:
            group_keys_out = None

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
        group_keys=group_keys_out,
        group_kind=group_kind,
        data_source=data_source,
    )


def load_dataset(
    csv_path: str | None = None,
    *,
    features: Sequence[str] = ESOL_FEATURES,
    target: str = ESOL_TARGET,
    deduplicate_smiles: bool = True,
    canonicalize_smiles: bool = False,
    offline: bool = False,
    allow_synthetic: bool = False,
    min_rows: int = 4,
    encoding: str = "descriptors",
) -> DatasetBundle:
    """Load ESOL (or custom CSV; synthetic fixture strictly opt-in via `allow_synthetic=True`)."""
    if encoding not in ("descriptors", "rdkit_2d_morgan"):
        raise ValueError(
            f"Unknown encoding {encoding!r} for load_dataset(); expected 'descriptors' or 'rdkit_2d_morgan'"
        )
    if encoding == "rdkit_2d_morgan" and not _chem_mod.HAS_RDKIT:
        raise ImportError(
            "RDKit is required for encoding='rdkit_2d_morgan'; install `lab-decide[chem]` (`pip install rdkit`)."
        )

    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        data_source = "real_csv"
    else:
        if offline and not os.path.exists(DEFAULT_DATA_PATH):
            if allow_synthetic:
                df = make_synthetic_solubility_dataframe(n=240, seed=0)
                data_source = "synthetic_fixture"
            else:
                ensure_esol_csv(DEFAULT_DATA_PATH, offline=True)
                raise AssertionError("unreachable")
        else:
            resolved = ensure_esol_csv(DEFAULT_DATA_PATH, offline=offline)
            df = pd.read_csv(resolved)
            data_source = "real_csv"

    if encoding == "rdkit_2d_morgan" and "smiles" not in df.columns:
        raise ValueError(
            f"Dataset is missing 'smiles' column required for encoding='rdkit_2d_morgan'; columns: {list(df.columns)}"
        )

    bundle = validate_and_clean_dataframe(
        df,
        features=features,
        target=target,
        deduplicate_smiles=deduplicate_smiles,
        canonicalize_smiles=canonicalize_smiles,
        min_rows=min_rows,
        data_source=data_source,
    )
    if encoding == "rdkit_2d_morgan":
        X_mol = featurize_rdkit_2d_morgan_batch(bundle.smiles, strict=False)
        if list(features) == list(ESOL_FEATURES) and bundle.X.shape[1] == len(ESOL_FEATURES):
            X_mol[:, : len(ESOL_FEATURES)] = bundle.X
        return replace(
            bundle,
            X=X_mol,
            feature_names=list(get_rdkit_2d_morgan_feature_names()),
        )
    return bundle


def load(
    csv_path: str | None = None,
    *,
    offline: bool = False,
    allow_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible loader returning `(X, y)` with schema check & SMILES deduplication."""
    bundle = load_dataset(csv_path=csv_path, offline=offline, allow_synthetic=allow_synthetic)
    return bundle.X, bundle.y


def _balanced_group_partition(
    n_samples: int,
    groups: Sequence[str],
    fracs: tuple[float, float, float, float],
    seed: int,
    *,
    strategy_name: str,
) -> SplitIndices:
    """Deterministic balanced 4-way group-disjoint bin packing (`train, pool, calibration, test`).

    Sorts unique groups by descending row count, uses `seed` ONLY to shuffle groups of equal
    row count (and tie-break equal deficits), places the largest groups into `train` first up
    to the requested training row target, and greedily assigns remaining groups to maximize
    remaining row deficit `target_rows[b] - current_rows[b]` while guaranteeing every split
    receives at least one group.
    """
    if len(groups) != n_samples:
        raise ValueError(
            f"groups sequence of length n_samples ({n_samples}) is required for strategy={strategy_name!r}, "
            f"got length {len(groups)}"
        )

    members_by_key: dict[str, list[int]] = {}
    for idx, raw_g in enumerate(groups):
        key = str(raw_g).strip() if raw_g is not None else ""
        if not key:
            raise ValueError(f"Empty group key at row {idx} for strategy={strategy_name!r}")
        if key.startswith("fallback_r"):
            raise ValueError(
                f"Regex fallback group key {key!r} is forbidden in strategy={strategy_name!r}"
            )
        members_by_key.setdefault(key, []).append(idx)

    unique_keys = sorted(members_by_key.keys())
    n_groups = len(unique_keys)
    if n_groups < 4:
        raise ValueError(
            f"{strategy_name.capitalize()} split requires at least 4 distinct groups, found {n_groups}"
        )

    # Group keys by exact integer member count; shuffle ONLY within equal-size tiers using `seed`
    by_size: dict[int, list[str]] = {}
    for k in unique_keys:
        by_size.setdefault(len(members_by_key[k]), []).append(k)

    rng = np.random.default_rng(seed)
    ordered_keys: list[str] = []
    for sz in sorted(by_size.keys(), reverse=True):
        tier = by_size[sz]
        if len(tier) > 1:
            perm = rng.permutation(len(tier))
            ordered_keys.extend(tier[int(p)] for p in perm)
        else:
            ordered_keys.append(tier[0])

    targets = [float(f) * float(n_samples) for f in fracs]
    bucket_rows = [0, 0, 0, 0]
    bucket_groups = [0, 0, 0, 0]
    bucket_members: list[list[int]] = [[], [], [], []]

    filling_train_head = True
    for pos, g_key in enumerate(ordered_keys):
        members = members_by_key[g_key]
        sz = len(members)
        rem_groups = n_groups - pos
        empty_buckets = [b for b in range(4) if bucket_groups[b] == 0]

        if rem_groups == len(empty_buckets):
            # Reserve final groups so zero splits are ever left empty
            cand_buckets = empty_buckets
            filling_train_head = False
        elif filling_train_head:
            empty_non_train = sum(1 for b in (1, 2, 3) if bucket_groups[b] == 0)
            # Place largest head groups into train first while doing so stays closer to train target
            if pos == 0 or (
                rem_groups > empty_non_train
                and bucket_rows[0] < targets[0]
                and abs((bucket_rows[0] + sz) - targets[0]) <= abs(bucket_rows[0] - targets[0])
            ):
                chosen = 0
                bucket_rows[chosen] += sz
                bucket_groups[chosen] += 1
                bucket_members[chosen].extend(members)
                continue
            filling_train_head = False
            cand_buckets = [0, 1, 2, 3]
        else:
            cand_buckets = [0, 1, 2, 3]

        # Greedy L2-optimal deficit packing: pick bucket maximizing (target_rows - current_rows)
        deficits = [targets[b] - float(bucket_rows[b]) for b in cand_buckets]
        max_def = max(deficits)
        tied = [cand_buckets[i] for i, d_val in enumerate(deficits) if abs(d_val - max_def) <= 1e-9]
        chosen = tied[int(rng.integers(0, len(tied)))] if len(tied) > 1 else tied[0]

        bucket_rows[chosen] += sz
        bucket_groups[chosen] += 1
        bucket_members[chosen].extend(members)

    names = ("train", "pool", "calibration", "test")
    group_counts = {names[b]: int(bucket_groups[b]) for b in range(4)}
    achieved_fracs = {names[b]: float(bucket_rows[b]) / float(n_samples) for b in range(4)}

    return SplitIndices(
        train_idx=np.sort(np.asarray(bucket_members[0], dtype=int)),
        pool_idx=np.sort(np.asarray(bucket_members[1], dtype=int)),
        cal_idx=np.sort(np.asarray(bucket_members[2], dtype=int)),
        test_idx=np.sort(np.asarray(bucket_members[3], dtype=int)),
        n_groups_total=n_groups,
        group_counts=group_counts,
        achieved_fractions=achieved_fracs,
    )


def partition_splits(
    n_samples: int,
    *,
    smiles: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    strategy: str = "random",
    seed: int = 0,
    fracs: tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15),
) -> SplitIndices:
    """Partition dataset into 4 strictly disjoint splits: `(train, pool, calibration, test)`.

    Supports:
    - `'random'`: stratified index random split.
    - `'scaffold'`: exact RDKit Bemis-Murcko scaffold balanced group-packing (raises `ImportError`
      if RDKit is missing and `ValueError` on non-SMILES composite strings).
    - `'group'`: balanced group-packing over explicit `groups` (solute, DOI, aryl halide / substrate,
      or Bemis-Murcko scaffold).
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
        tr_s = np.sort(tr_idx)
        pool_s = np.sort(pool_idx)
        cal_s = np.sort(cal_idx)
        te_s = np.sort(te_idx)
        return SplitIndices(
            train_idx=tr_s,
            pool_idx=pool_s,
            cal_idx=cal_s,
            test_idx=te_s,
            n_groups_total=None,
            group_counts=None,
            achieved_fractions={
                "train": float(tr_s.size) / float(n_samples),
                "pool": float(pool_s.size) / float(n_samples),
                "calibration": float(cal_s.size) / float(n_samples),
                "test": float(te_s.size) / float(n_samples),
            },
        )

    if strategy == "scaffold":
        if smiles is None or len(smiles) != n_samples:
            raise ValueError("smiles sequence of length n_samples is required when strategy='scaffold'")
        scaf_groups = compute_scaffold_groups(smiles).tolist()
        return _balanced_group_partition(
            n_samples, scaf_groups, fracs, seed, strategy_name="scaffold"
        )

    if strategy == "group":
        eff_groups: Sequence[str] | None = groups
        if eff_groups is None and smiles is not None:
            eff_groups = compute_scaffold_groups(smiles).tolist()
        if eff_groups is None or len(eff_groups) != n_samples:
            raise ValueError(
                "groups sequence of length n_samples is required when strategy='group' "
                "(ensure RDKit is installed for single-molecule scaffold groups or supply explicit group_keys)."
            )
        return _balanced_group_partition(
            n_samples, eff_groups, fracs, seed, strategy_name="group"
        )

    raise ValueError(
        f"Unknown split strategy {strategy!r}; expected 'random', 'scaffold', or 'group'"
    )
