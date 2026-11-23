"""CALiSol-23 publication-DOI split loader & kernel support gate adapter (aligned with labbench-posttrain)."""
from __future__ import annotations

import csv
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

from lab_decide.conformal import ConformalTrustModel, fit_and_calibrate, summarize

# Exact publication-DOI holdouts shared with sibling repo labbench-posttrain/bioenvs/calisol.py
CALISOL_VAL_DOIS: frozenset[str] = frozenset(
    {
        "10.1021/je0498863",
        "10.1149/1.1403730",
        "10.1149/2.0271802jes",
        "10.5012/bkcs.2011.32.10.3682",
        "10.1149/1.1872737",
    }
)

CALISOL_TEST_DOIS: frozenset[str] = frozenset(
    {
        "10.1149/1.1914757",
        "10.1149/1.1513556",
        "10.1016/S0378-7753(03)00153-8",
        "10.1016/S0013-4686(99)00431-4",
        "10.1016/S0378-7753(01)00697-8",
        "10.1149/1.1466857",
        "10.1007/s10008-002-0300-9",
    }
)

COMMON_SALTS = ("LiPF6", "LiTFSI", "LiBF4", "LiClO4", "LiFSI")
COMMON_SOLVENTS = ("EC", "PC", "DMC", "EMC", "DEC", "DME")


def assign_calisol_doi_split(doi: str) -> str:
    d = (doi or "").strip()
    if d in CALISOL_VAL_DOIS:
        return "cal"
    if d in CALISOL_TEST_DOIS:
        return "test"
    return "train"


def formulation_kernel_support(
    query_rec: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
    *,
    support_scale: float = 0.28,
) -> tuple[float, float]:
    """Compute `(support_score, d_min)` matching labbench-posttrain's InverseDesignSupportTemplate."""
    if not train_records:
        return 0.0, 10.0
    q_salt = str(query_rec["salt"])
    q_c = max(float(query_rec["c"]), 1e-3)
    q_t = float(query_rec["T_K"])
    q_solv: Mapping[str, float] = query_rec["solvents"]

    d_min = float("inf")
    for ref in train_records:
        salt_pen = 0.0 if q_salt == ref["salt"] else 1.25
        r_solv: Mapping[str, float] = ref["solvents"]
        all_s = set(q_solv) | set(r_solv)
        tv = 0.5 * sum(abs(float(q_solv.get(s, 0.0)) - float(r_solv.get(s, 0.0))) for s in all_s)
        c_diff = abs(math.log(q_c) - math.log(max(float(ref["c"]), 1e-3))) / 0.85
        t_diff = max(0.0, abs(q_t - float(ref["T_K"])) - 5.0) / 25.0
        d = salt_pen + tv + c_diff + t_diff
        if d < d_min:
            d_min = d
    return math.exp(-d_min / support_scale), d_min


def make_synthetic_calisol_records(n_per_split: tuple[int, int, int] = (140, 50, 50), seed: int = 0) -> list[dict[str, Any]]:
    """Generate deterministic electrolyte conductivity records stratified by disjoint CALiSol DOIs."""
    rng = np.random.default_rng(seed)
    train_dois = [f"10.1149/train.{i:03d}" for i in range(6)]
    val_dois = sorted(CALISOL_VAL_DOIS)
    test_dois = sorted(CALISOL_TEST_DOIS)

    records: list[dict[str, Any]] = []
    for split_name, count, doi_pool, ood_prob in [
        ("train", n_per_split[0], train_dois, 0.0),
        ("cal", n_per_split[1], val_dois, 0.20),
        ("test", n_per_split[2], test_dois, 0.25),
    ]:
        for i in range(count):
            is_ood = rng.random() < ood_prob
            salt = "LiClO4" if is_ood else str(rng.choice(["LiPF6", "LiTFSI", "LiBF4"]))
            t_k = float(rng.uniform(335.0, 360.0) if is_ood else rng.uniform(265.0, 320.0))
            conc = float(rng.uniform(2.2, 3.5) if is_ood else rng.uniform(0.5, 1.8))
            ec_frac = float(rng.uniform(0.2, 0.7))
            solv = {"EC": round(ec_frac, 4), "DMC": round(1.0 - ec_frac, 4)}
            # Arrhenius-like peak around c ~ 1.0 M and higher temperature
            base_k = 11.5 * math.exp(-((conc - 1.05) ** 2) / 0.9) * math.exp(0.012 * (t_k - 298.15))
            noise = rng.normal(0.0, 2.2 if is_ood else 0.55)
            k_val = max(0.05, base_k + noise)
            records.append(
                {
                    "doi": str(doi_pool[i % len(doi_pool)]),
                    "split": split_name,
                    "salt": salt,
                    "T_K": round(t_k, 2),
                    "c": round(conc, 4),
                    "k_mS_cm": round(k_val, 5),
                    "solvents": solv,
                }
            )
    return records


def load_calisol_csv_records(csv_path: str) -> list[dict[str, Any]]:
    """Parse real `calisol23.csv` into sanitized physical records with DOI split labels."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CALiSol-23 CSV not found: {csv_path}")

    records: list[dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or len(reader.fieldnames) < 8:
            raise ValueError(f"Malformed CALiSol-23 header in {csv_path}")
        solvent_cols = list(reader.fieldnames[7:])
        for row in reader:
            try:
                k = float(row["k"])
                t_k = float(row["T"])
                conc = float(row["c"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(k) and k >= 1e-4 and math.isfinite(conc) and conc >= 1e-3 and math.isfinite(t_k) and t_k > 0.0):
                continue
            raw_solv: dict[str, float] = {}
            bad = False
            for sc in solvent_cols:
                v = float(row.get(sc) or 0.0)
                if not math.isfinite(v) or v < -1e-4:
                    bad = True
                    break
                if v > 1e-5:
                    raw_solv[sc.strip()] = v
            if bad or not raw_solv:
                continue
            tot = sum(raw_solv.values())
            if not (0.94 <= tot <= 1.06):
                continue
            norm_solv = {s: round(v / tot, 6) for s, v in sorted(raw_solv.items())}
            doi = (row.get("doi") or "").strip()
            records.append(
                {
                    "doi": doi,
                    "split": assign_calisol_doi_split(doi),
                    "salt": (row.get("salt") or "").strip(),
                    "T_K": t_k,
                    "c": conc,
                    "k_mS_cm": k,
                    "solvents": norm_solv,
                }
            )
    if not records:
        raise ValueError(f"No valid electrolyte rows parsed from {csv_path}")
    return records


def featurize_calisol_records(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Featurize electrolyte records into tabular descriptors predicting `log1p(k_mS_cm)`."""
    feat_names = ["T_K", "c_mol", "log_c"] + [f"salt_{s}" for s in COMMON_SALTS] + [f"solv_{s}" for s in COMMON_SOLVENTS]
    X = np.zeros((len(records), len(feat_names)), dtype=np.float64)
    y = np.zeros(len(records), dtype=np.float64)
    for i, rec in enumerate(records):
        c_val = max(float(rec["c"]), 1e-3)
        X[i, 0] = float(rec["T_K"])
        X[i, 1] = c_val
        X[i, 2] = math.log(c_val)
        salt = str(rec["salt"])
        for j, s_name in enumerate(COMMON_SALTS):
            if salt == s_name:
                X[i, 3 + j] = 1.0
        solv = rec["solvents"]
        base = 3 + len(COMMON_SALTS)
        for j, sv_name in enumerate(COMMON_SOLVENTS):
            X[i, base + j] = float(solv.get(sv_name, 0.0))
        y[i] = math.log1p(float(rec["k_mS_cm"]))
    return X, y, feat_names


def evaluate_calisol_conformal(
    csv_path: str | None = None,
    *,
    alpha: float = 0.10,
    trust_pct: float = 80.0,
    seed: int = 0,
    max_records_per_split: int = 600,
) -> dict[str, Any]:
    """Evaluate Mondrian conformal gating vs CALiSol-23 formulation kernel support across DOI splits."""
    if csv_path and os.path.exists(csv_path):
        records = load_calisol_csv_records(csv_path)
        source = "calisol23_csv"
    else:
        records = make_synthetic_calisol_records(seed=seed)
        source = "synthetic_calisol_doi_fixture"

    rng = np.random.default_rng(seed)
    sampled: list[dict[str, Any]] = []
    for sp in ("train", "cal", "test"):
        sub = [r for r in records if r["split"] == sp]
        if len(sub) > max_records_per_split:
            idx = np.sort(rng.choice(len(sub), size=max_records_per_split, replace=False))
            sub = [sub[int(i)] for i in idx]
        sampled.extend(sub)

    X, y, feat_names = featurize_calisol_records(sampled)
    splits = np.array([r["split"] for r in sampled])
    tr_mask = splits == "train"
    cal_mask = splits == "cal"
    te_mask = splits == "test"

    advisor = fit_and_calibrate(
        X[tr_mask],
        y[tr_mask],
        X[cal_mask],
        y[cal_mask],
        alpha=alpha,
        trust_pct=trust_pct,
        seed=seed,
        domain_bounds=(0.0, 6.0),
        feature_names=feat_names,
    )
    res = advisor.predict(X[te_mask], use_mondrian=True)
    yt = y[te_mask]
    inside = np.asarray(res["trusted"], dtype=bool)
    c_in, w_in, e_in = summarize(yt, res["lo"], res["hi"], inside, pred=res["pred"])
    c_out, _, e_out = summarize(yt, res["lo"], res["hi"], ~inside, pred=res["pred"])

    tr_recs = [r for r in sampled if r["split"] == "train"]
    te_recs = [r for r in sampled if r["split"] == "test"]
    kernel_supports = np.asarray(
        [formulation_kernel_support(r, tr_recs)[0] for r in te_recs],
        dtype=np.float64,
    )
    # Gate agreement between kNN trust region (inside) and kernel support >= 20th percentile
    k_gate = kernel_supports >= np.percentile(kernel_supports, 100.0 - trust_pct)
    gate_agreement = float(np.mean(inside == k_gate))

    return {
        "source": source,
        "n_train": int(np.sum(tr_mask)),
        "n_cal": int(np.sum(cal_mask)),
        "n_test": int(np.sum(te_mask)),
        "answered_fraction": float(np.mean(inside)),
        "cov_in": c_in,
        "cov_out": c_out,
        "mae_in": e_in,
        "mae_out": e_out,
        "width_in": w_in,
        "knn_vs_kernel_support_agreement": gate_agreement,
    }
