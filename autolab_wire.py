"""Auto-Lab 15-repository wire node for lab-decide (Layer L6 Calibrated Conformal Decision Gate).

Implements `autolab-wire/v1` packet processing backed by a frozen `ConformalTrustModel` fitted on
SHA-256-pinned ESOL (`seed=0`, 4-way split `561/224/169/169`):
- Rich `rdkit_2d_morgan` molecular featurization + semiparametric `FrontierEnsembleRegressor` point
  surrogate paired with Mondrian split-conformal calibration over the canonical 6D ESOL kNN
  applicability domain (`effective_n_calib = 169`, failing closed when RDKit or the SHA-verified
  ESOL CSV is unavailable, or when `candidate_smiles` cannot be parsed),
- Exact parity with `ConformalTrustModel.predict` on in-domain/out-of-domain gating and intervals,
- Packet `knn_distance` honored ONLY when `distance_source == "upstream"`,
- Genuine `(smiles/descriptors, measured_y)` / residual calibration updates from SHA-256 deduplicated
  `certified_ledger_entries` (never widening `effective_threshold` by counting entries or from
  high-residual measurements).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID = "lab-decide"
LAYER = "L6"
SCHEMA_VERSION = "autolab-wire/v1"
REPO_DIR = Path(__file__).resolve().parent

_SRC_DIR = str(REPO_DIR / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _load_mondrian_drift_conformal() -> Any:
    """Load local `mondrian_drift_conformal.py` under a unique module key to avoid cross-repo collisions."""
    mod_name = "_lab_decide_mondrian_drift_conformal"
    existing = sys.modules.get(mod_name)
    if existing is not None:
        return existing
    mod_path = REPO_DIR / "community_extensions" / "mondrian_drift_conformal.py"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


conformal_order_rank = _load_mondrian_drift_conformal().conformal_order_rank
import lab_decide.chemistry as _chem_mod  # noqa: E402
from lab_decide.chemistry import (  # noqa: E402
    ESOL_FEATURES,
    HAS_RDKIT,
    smiles_to_descriptors,
    smiles_to_rdkit_2d_morgan,
)
from lab_decide.conformal import (  # noqa: E402
    ConformalTrustModel,
    _fit_point_surrogate,
    conformal_quantile,
    conformal_threshold,
    fit_and_calibrate,
    weighted_conformal_quantile,
)
from lab_decide.data import (  # noqa: E402
    ESOL_SHA256,
    compute_file_sha256,
    load_dataset,
    partition_splits,
)

DEFAULT_CANDIDATE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
DEFAULT_WIRE_ENCODING = "rdkit_2d_morgan"
PHYS_MIN_LOGS = -15.0
PHYS_MAX_LOGS = 1.75
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_DEFAULT_ESOL_CANDIDATE_PATHS: tuple[Path, ...] = (
    REPO_DIR / "data" / "delaney-processed.csv",
    REPO_DIR / "src" / "lab_decide" / "resources" / "delaney-processed.csv",
)


@dataclass(frozen=True)
class _FrozenEsolState:
    advisor: ConformalTrustModel
    fallback_6d_model: Any
    model_encoding: str
    surrogate_name: str
    csv_path: str
    X_cal: np.ndarray
    y_cal: np.ndarray
    d_cal: np.ndarray
    pred_cal: np.ndarray
    scores_cal: np.ndarray


def _sha256_lf_normalized_matches(path: Path, expected_hex: str) -> tuple[bool, str]:
    """Verify SHA-256 digest, normalizing CRLF -> LF if Windows git autocrlf altered line endings."""
    digest = compute_file_sha256(str(path))
    if digest.lower() == expected_hex.lower():
        return True, digest
    raw = path.read_bytes()
    if b"\r\n" in raw:
        import hashlib

        lf_digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        if lf_digest.lower() == expected_hex.lower():
            return True, lf_digest
    return False, digest


def _resolve_verified_esol_csv(explicit_path: str | None = None) -> str:
    """Resolve a local ESOL CSV verified against `ESOL_SHA256` without performing network I/O."""
    env_override = os.environ.get("LAB_DECIDE_ESOL_CSV", "").strip()
    target_explicit = explicit_path.strip() if isinstance(explicit_path, str) and explicit_path.strip() else env_override

    if target_explicit:
        p = Path(target_explicit)
        if not p.is_file():
            raise FileNotFoundError(
                f"ESOL calibration CSV is not cached at {p}; cannot certify L6 conformal gate offline."
            )
        ok, digest = _sha256_lf_normalized_matches(p, ESOL_SHA256)
        if not ok:
            raise RuntimeError(
                f"ESOL calibration CSV at {p} failed SHA-256 verification (expected {ESOL_SHA256}, got {digest})."
            )
        return str(p.resolve())

    for cand in _DEFAULT_ESOL_CANDIDATE_PATHS:
        if cand.is_file():
            ok, _ = _sha256_lf_normalized_matches(cand, ESOL_SHA256)
            if ok:
                return str(cand.resolve())

    searched = ", ".join(str(p) for p in _DEFAULT_ESOL_CANDIDATE_PATHS)
    raise FileNotFoundError(
        f"ESOL calibration CSV is not cached locally (checked: {searched}) with SHA-256 {ESOL_SHA256}."
    )


@lru_cache(maxsize=8)
def _load_frozen_esol_state(
    resolved_csv_path: str,
    model_encoding: str = DEFAULT_WIRE_ENCODING,
) -> _FrozenEsolState:
    """Fit and cache the canonical seed-0 ESOL `ConformalTrustModel` on SHA-pinned CSV."""
    if model_encoding not in ("rdkit_2d_morgan", "descriptors"):
        raise ValueError(
            f"Unsupported model_encoding {model_encoding!r}; expected 'rdkit_2d_morgan' or 'descriptors'"
        )
    ok, digest = _sha256_lf_normalized_matches(Path(resolved_csv_path), ESOL_SHA256)
    if not ok:
        raise RuntimeError(
            f"ESOL CSV {resolved_csv_path} failed SHA-256 check (expected {ESOL_SHA256}, got {digest})."
        )
    bundle_6d = load_dataset(csv_path=resolved_csv_path, offline=True, encoding="descriptors")
    splits = partition_splits(
        len(bundle_6d.y),
        strategy="random",
        seed=0,
        fracs=(0.50, 0.20, 0.15, 0.15),
    )
    X_tr_6d = bundle_6d.X[splits.train_idx]
    y_tr = bundle_6d.y[splits.train_idx]
    y_cal = bundle_6d.y[splits.cal_idx]

    ref_6d_gbdt = _fit_point_surrogate(
        X_tr_6d, y_tr, surrogate="gbdt", loss="huber", seed=0
    )

    if model_encoding == "rdkit_2d_morgan":
        bundle_mol = load_dataset(
            csv_path=resolved_csv_path, offline=True, encoding="rdkit_2d_morgan"
        )
        X_tr = bundle_mol.X[splits.train_idx]
        X_cal = bundle_mol.X[splits.cal_idx]
        surrogate_name = "frontier_ensemble"
        advisor = fit_and_calibrate(
            X_tr,
            y_tr,
            X_cal,
            y_cal,
            alpha=0.10,
            k=5,
            trust_pct=80.0,
            seed=0,
            loss="huber",
            nonconformity="normalized",
            distance_weighting="importance",
            distance_metric="euclidean",
            surrogate=surrogate_name,
            domain_bounds=(PHYS_MIN_LOGS, PHYS_MAX_LOGS),
            feature_names=bundle_mol.feature_names,
            gate_n_features=len(ESOL_FEATURES),
            gate_reference_model=ref_6d_gbdt,
        )
    else:
        X_cal = bundle_6d.X[splits.cal_idx]
        surrogate_name = "gbdt"
        advisor = fit_and_calibrate(
            X_tr_6d,
            y_tr,
            X_cal,
            y_cal,
            alpha=0.10,
            k=5,
            trust_pct=80.0,
            seed=0,
            loss="huber",
            nonconformity="normalized",
            distance_weighting="importance",
            domain_bounds=(PHYS_MIN_LOGS, PHYS_MAX_LOGS),
            feature_names=bundle_6d.feature_names,
        )

    d_cal = advisor.distance(X_cal)
    pred_cal = np.asarray(advisor.model.predict(X_cal), dtype=np.float64)
    scores_cal = np.abs(y_cal - pred_cal) / (1.0 + advisor.gamma * d_cal)
    return _FrozenEsolState(
        advisor=advisor,
        fallback_6d_model=ref_6d_gbdt,
        model_encoding=model_encoding,
        surrogate_name=surrogate_name,
        csv_path=resolved_csv_path,
        X_cal=X_cal,
        y_cal=y_cal,
        d_cal=d_cal,
        pred_cal=pred_cal,
        scores_cal=scores_cal,
    )


def get_cached_esol_model(
    esol_csv_path: str | None = None,
    *,
    model_encoding: str = DEFAULT_WIRE_ENCODING,
) -> ConformalTrustModel:
    """Public accessor returning the cached seed-0 SHA-pinned ESOL `ConformalTrustModel`."""
    resolved = _resolve_verified_esol_csv(esol_csv_path)
    return _load_frozen_esol_state(resolved, model_encoding=model_encoding).advisor


def _rdkit_available() -> bool:
    return bool(HAS_RDKIT and _chem_mod.HAS_RDKIT)


def _extract_certified_entry_keys(entry: Any) -> tuple[str, str | None] | None:
    """Validate a `lab-attrib` ledger entry and return `(primary_dedup_key, optional_run_key)`."""
    if not isinstance(entry, Mapping):
        return None
    if entry.get("admit_to_memory") is False:
        return None

    for num_key in ("calibration_residual", "measured_y", "measured_log_s", "observed_y"):
        if num_key in entry and entry[num_key] is not None:
            if isinstance(entry[num_key], bool):
                return None
            try:
                val = float(entry[num_key])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(val):
                return None

    for w_key in ("likelihood_ratio_weight", "importance_weight"):
        if w_key in entry and entry[w_key] is not None:
            if isinstance(entry[w_key], bool):
                return None
            try:
                w_val = float(entry[w_key])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(w_val) or w_val <= 0.0:
                return None

    run_key: str | None = None
    for id_field in ("run_id", "entry_id", "ledger_id"):
        raw_id = entry.get(id_field)
        if isinstance(raw_id, str) and raw_id.strip():
            run_key = f"id:{raw_id.strip()}"
            break

    raw_sha = entry.get("ledger_entry_sha256")
    if "ledger_entry_sha256" in entry and raw_sha is not None:
        if not isinstance(raw_sha, str) or not _SHA256_HEX_RE.fullmatch(raw_sha):
            return None
        return (f"sha256:{raw_sha.lower()}", run_key)

    if entry.get("admit_to_memory") is True and run_key is not None:
        return (run_key, None)

    return None


def _deduplicate_certified_entries(
    entries: list[Any],
) -> tuple[list[Mapping[str, Any]], int, int]:
    """Filter and deduplicate certified ledger entries in deterministic first-seen order."""
    deduped: list[Mapping[str, Any]] = []
    seen_keys: set[str] = set()
    dup_count = 0
    invalid_count = 0

    for entry in entries:
        keys = _extract_certified_entry_keys(entry)
        if keys is None:
            invalid_count += 1
            continue
        primary_key, secondary_key = keys
        if primary_key in seen_keys or (secondary_key is not None and secondary_key in seen_keys):
            dup_count += 1
            continue
        seen_keys.add(primary_key)
        if secondary_key is not None:
            seen_keys.add(secondary_key)
        deduped.append(entry)

    return deduped, dup_count, invalid_count


def _featurize_entry_vectors(
    entry: Mapping[str, Any],
    model_encoding: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return `(full_model_vec, gate_6d_vec)` for a ledger entry strictly via RDKit or explicit dict."""
    smi = entry.get("candidate_smiles")
    if isinstance(smi, str) and smi.strip() and _rdkit_available():
        try:
            if model_encoding == "rdkit_2d_morgan":
                mol_vec = smiles_to_rdkit_2d_morgan(smi, strict=True)
                return mol_vec, mol_vec[: len(ESOL_FEATURES)]
            d = smiles_to_descriptors(smi)
            vec6 = np.asarray([float(d[f]) for f in ESOL_FEATURES], dtype=np.float64)
            return vec6, vec6
        except (ValueError, ImportError):
            pass

    desc_map = entry.get("candidate_descriptors")
    if isinstance(desc_map, Mapping) and all(f in desc_map for f in ESOL_FEATURES):
        try:
            vec6 = np.asarray([float(desc_map[f]) for f in ESOL_FEATURES], dtype=np.float64)
            if np.all(np.isfinite(vec6)):
                return (vec6 if model_encoding == "descriptors" else None), vec6
        except (TypeError, ValueError):
            return None, None

    return None, None


def _extract_entry_calibration_observation(
    entry: Mapping[str, Any],
    state: _FrozenEsolState,
) -> tuple[float, float, float, bool] | None:
    """Return `(d_entry, norm_score, lr_weight, has_descriptors)` for a measured ledger entry.

    Returns `None` when the entry carries neither a measured target (`measured_y` / `measured_log_s`
    / `observed_y`) nor an explicit `calibration_residual`.
    """
    advisor = state.advisor
    x_full, x_gate6 = _featurize_entry_vectors(entry, state.model_encoding)
    has_desc = x_gate6 is not None
    if x_full is not None:
        d_entry = float(advisor.distance(x_full)[0])
        pred_entry = float(advisor.model.predict(x_full.reshape(1, -1))[0])
    elif x_gate6 is not None:
        d_entry = float(advisor.distance(x_gate6)[0])
        pred_entry = float(state.fallback_6d_model.predict(x_gate6.reshape(1, -1))[0])
    elif (
        entry.get("distance_source") == "upstream"
        and "knn_distance" in entry
        and entry["knn_distance"] is not None
    ):
        d_entry = float(entry["knn_distance"])
        pred_entry = float("nan")
    else:
        d_entry = 0.0
        pred_entry = float("nan")

    measured_val: float | None = None
    for y_key in ("measured_y", "measured_log_s", "observed_y"):
        if y_key in entry and entry[y_key] is not None:
            measured_val = float(entry[y_key])
            break

    if measured_val is not None and has_desc and math.isfinite(pred_entry):
        raw_res = abs(measured_val - pred_entry)
    elif "calibration_residual" in entry and entry["calibration_residual"] is not None:
        raw_res = abs(float(entry["calibration_residual"]))
    else:
        return None

    norm_score = raw_res / (1.0 + advisor.gamma * max(d_entry, 0.0))

    weight = 1.0
    for w_key in ("likelihood_ratio_weight", "importance_weight"):
        if w_key in entry and entry[w_key] is not None:
            weight = float(entry[w_key])
            break
    return d_entry, norm_score, weight, has_desc


def _emit_fail_closed_abstain(
    out: dict[str, Any],
    *,
    candidate_smiles: str,
    distance_source: str,
    reason_code: str,
    detail: str,
    model_encoding: str = DEFAULT_WIRE_ENCODING,
    knn_distance: float | None = None,
    effective_threshold: float | None = None,
    effective_n_calib: int = 0,
    duplicate_ignored: int = 0,
    invalid_ignored: int = 0,
) -> dict[str, Any]:
    out["halted"] = True
    out["halt_reason"] = "L6_ABSTAIN"
    out["decision"] = {
        "candidate_smiles": candidate_smiles,
        "in_trust_region": False,
        "abstain": True,
        "abstain_reason": reason_code,
        "model_encoding": model_encoding,
        "surrogate": "frontier_ensemble" if model_encoding == "rdkit_2d_morgan" else "gbdt",
        "distance_source": distance_source,
        "knn_distance": round(float(knn_distance), 6) if (knn_distance is not None and math.isfinite(knn_distance)) else None,
        "effective_threshold": round(float(effective_threshold), 6) if (effective_threshold is not None and math.isfinite(effective_threshold)) else None,
        "effective_n_calib": int(effective_n_calib),
        "predicted_log_s": None,
        "conformal_interval": [PHYS_MIN_LOGS, PHYS_MAX_LOGS],
        "nonconformity": "normalized",
        "duplicate_ledger_entries_ignored": duplicate_ignored,
        "invalid_ledger_entries_ignored": invalid_ignored,
    }
    summary = f"Conformal decision gate abstained ({reason_code}: {detail})."
    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "ABSTAIN",
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "out_of_domain_always_abstains",
            "first_principles": (
                "Layer L6 conformal gating fails closed whenever RDKit descriptor parsing, "
                "SHA-256 verified ESOL calibration data, finite-sample conformal rank, or "
                "kNN applicability-domain membership cannot be certified."
            ),
        }
    )
    return out


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate and score an `autolab-wire/v1` packet with the frozen SHA-pinned ESOL model."""
    out: dict[str, Any] = copy.deepcopy(dict(packet)) if packet is not None else {}
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if bool(out["halted"]):
        return out

    raw_smiles = out.get("candidate_smiles", DEFAULT_CANDIDATE_SMILES)
    candidate_smiles = raw_smiles if isinstance(raw_smiles, str) else str(raw_smiles or "")
    out.setdefault("candidate_smiles", candidate_smiles)

    raw_certified = out.get("certified_ledger_entries", [])
    certified_entries = list(raw_certified) if isinstance(raw_certified, (list, tuple)) else []
    valid_certified, duplicate_ignored, invalid_ignored = _deduplicate_certified_entries(certified_entries)

    req_dist_source = out.get("distance_source")
    use_upstream_dist = (
        req_dist_source == "upstream"
        and "knn_distance" in out
        and out["knn_distance"] is not None
    )
    distance_source = "upstream" if use_upstream_dist else "fitted_esol_model"

    req_encoding = out.get("model_encoding", DEFAULT_WIRE_ENCODING)
    model_encoding = str(req_encoding) if isinstance(req_encoding, str) else DEFAULT_WIRE_ENCODING
    if model_encoding not in ("rdkit_2d_morgan", "descriptors"):
        return _emit_fail_closed_abstain(
            out,
            candidate_smiles=candidate_smiles,
            distance_source=distance_source,
            reason_code="invalid_model_encoding",
            detail=f"Unsupported model_encoding={req_encoding!r}",
            model_encoding=DEFAULT_WIRE_ENCODING,
            duplicate_ignored=duplicate_ignored,
            invalid_ignored=invalid_ignored,
        )

    # 1. Fail closed if RDKit is missing or candidate_smiles does not parse as a valid molecule
    if not _rdkit_available():
        return _emit_fail_closed_abstain(
            out,
            candidate_smiles=candidate_smiles,
            distance_source=distance_source,
            reason_code="rdkit_unavailable",
            detail="RDKit is required to compute molecular descriptors for L6 conformal gating",
            model_encoding=model_encoding,
            duplicate_ignored=duplicate_ignored,
            invalid_ignored=invalid_ignored,
        )

    try:
        if not isinstance(raw_smiles, str) or not raw_smiles.strip():
            raise ValueError("candidate_smiles must be a non-empty SMILES string")
        if model_encoding == "rdkit_2d_morgan":
            x_cand = smiles_to_rdkit_2d_morgan(candidate_smiles, strict=True)
        else:
            cand_desc = smiles_to_descriptors(candidate_smiles)
            x_cand = np.asarray([float(cand_desc[f]) for f in ESOL_FEATURES], dtype=np.float64)
    except (ValueError, ImportError) as exc:
        return _emit_fail_closed_abstain(
            out,
            candidate_smiles=candidate_smiles,
            distance_source=distance_source,
            reason_code="invalid_candidate_smiles",
            detail=str(exc),
            model_encoding=model_encoding,
            duplicate_ignored=duplicate_ignored,
            invalid_ignored=invalid_ignored,
        )

    # 2. Load SHA-256 verified ESOL frozen model offline; fail closed if cache is missing/corrupt
    explicit_csv = out.get("esol_csv_path")
    try:
        resolved_csv = _resolve_verified_esol_csv(
            str(explicit_csv) if explicit_csv is not None else None
        )
        state = _load_frozen_esol_state(resolved_csv, model_encoding=model_encoding)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return _emit_fail_closed_abstain(
            out,
            candidate_smiles=candidate_smiles,
            distance_source=distance_source,
            reason_code="esol_cache_unavailable",
            detail=str(exc),
            model_encoding=model_encoding,
            duplicate_ignored=duplicate_ignored,
            invalid_ignored=invalid_ignored,
        )

    advisor = state.advisor

    try:
        alpha = float(out.get("alpha", advisor.alpha))
        test_weight = float(out.get("test_weight", 1.0))
    except (TypeError, ValueError) as exc:
        return _emit_fail_closed_abstain(
            out,
            candidate_smiles=candidate_smiles,
            distance_source=distance_source,
            reason_code="invalid_alpha_or_weight",
            detail=str(exc),
            model_encoding=model_encoding,
            effective_threshold=advisor.threshold,
            effective_n_calib=int(state.scores_cal.size),
            duplicate_ignored=duplicate_ignored,
            invalid_ignored=invalid_ignored,
        )

    if use_upstream_dist:
        try:
            raw_d = out["knn_distance"]
            if isinstance(raw_d, bool):
                raise ValueError("knn_distance must be numeric, not bool")
            knn_distance = float(raw_d)
        except (TypeError, ValueError) as exc:
            return _emit_fail_closed_abstain(
                out,
                candidate_smiles=candidate_smiles,
                distance_source=distance_source,
                reason_code="invalid_upstream_distance",
                detail=str(exc),
                model_encoding=model_encoding,
                effective_threshold=advisor.threshold,
                effective_n_calib=int(state.scores_cal.size),
                duplicate_ignored=duplicate_ignored,
                invalid_ignored=invalid_ignored,
            )
    else:
        knn_distance = float(advisor.distance(x_cand)[0])

    # Extract genuine calibration observations from deduplicated certified ledger entries
    measured_obs: list[tuple[float, float, float, bool]] = []
    unmeasured_ignored = 0
    for entry in valid_certified:
        obs = _extract_entry_calibration_observation(entry, state)
        if obs is None:
            unmeasured_ignored += 1
        else:
            measured_obs.append(obs)

    alpha_valid = math.isfinite(alpha) and (0.0 < alpha < 1.0)
    weight_valid = math.isfinite(test_weight) and test_weight > 0.0

    # Fast-path exact parity with ConformalTrustModel.predict when using unaltered frozen ESOL state
    if (
        not measured_obs
        and not use_upstream_dist
        and alpha_valid
        and math.isclose(alpha, advisor.alpha, rel_tol=0.0, abs_tol=1e-12)
        and weight_valid
        and math.isclose(test_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and "n_calib_override" not in out
    ):
        pred_1d = advisor.predict(x_cand, use_mondrian=True)
        abstain = not bool(pred_1d["trusted"])
        low = float(pred_1d["lo"])
        high = float(pred_1d["hi"])
        point_pred = float(pred_1d["pred"])
        effective_threshold = float(advisor.threshold)
        effective_n_calib = int(state.scores_cal.size)
        rank_k = conformal_order_rank(effective_n_calib, alpha)
        rank_ok = True
    else:
        # Support optional n_calib_override in tests exercising small-sample rank abstention
        base_n = int(out.get("n_calib_override", state.scores_cal.size))
        base_n = max(0, min(base_n, int(state.scores_cal.size)))
        d_base = state.d_cal[:base_n]
        s_base = state.scores_cal[:base_n]

        obs_d = np.asarray([o[0] for o in measured_obs], dtype=np.float64)
        obs_s = np.asarray([o[1] for o in measured_obs], dtype=np.float64)
        obs_w = np.asarray([o[2] for o in measured_obs], dtype=np.float64)

        all_d = np.concatenate([d_base, obs_d]) if measured_obs else d_base
        all_s = np.concatenate([s_base, obs_s]) if measured_obs else s_base
        all_w = (
            np.concatenate([np.ones(base_n, dtype=np.float64), obs_w])
            if measured_obs
            else np.ones(base_n, dtype=np.float64)
        )
        effective_n_calib = int(all_s.size)

        # Threshold recomputation: NEVER count entries; entries whose nonconformity score exceeds
        # the current inside quantile `advisor.q_in` MUST NOT widen the trust region!
        eligible_thr_d = [
            o[0] for o in measured_obs if o[3] and (o[1] <= advisor.q_in)
        ]
        if base_n == 0:
            effective_threshold = float("inf")
        elif eligible_thr_d:
            d_for_thr = np.concatenate([d_base, np.asarray(eligible_thr_d, dtype=np.float64)])
            effective_threshold = conformal_threshold(d_for_thr, trust_pct=advisor.trust_pct)
        else:
            effective_threshold = (
                float(advisor.threshold)
                if base_n == int(state.scores_cal.size)
                else conformal_threshold(d_base, trust_pct=advisor.trust_pct)
            )

        if effective_n_calib > 0 and alpha_valid and weight_valid:
            rank_k = conformal_order_rank(effective_n_calib, alpha)
            use_unit_w = bool(np.all(all_w == 1.0) and test_weight == 1.0)

            def _q_eval(sub_s: np.ndarray, sub_w: np.ndarray) -> float:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if use_unit_w:
                        return conformal_quantile(sub_s, alpha=alpha)
                    return weighted_conformal_quantile(
                        sub_s, sub_w, alpha=alpha, test_weight=test_weight
                    )

            q_marg = _q_eval(all_s, all_w)
            in_mask = all_d <= effective_threshold
            if int(np.sum(in_mask)) > 0:
                q_in_raw = _q_eval(all_s[in_mask], all_w[in_mask])
                if math.isfinite(q_in_raw):
                    q_in_eff = q_in_raw
                elif math.isfinite(q_marg):
                    q_in_eff = float(max(q_marg, float(np.max(all_s[in_mask]))))
                else:
                    q_in_eff = float("inf")
            else:
                q_in_eff = q_marg if math.isfinite(q_marg) else float("inf")

            out_mask = ~in_mask
            if int(np.sum(out_mask)) > 0:
                q_out_cand = _q_eval(all_s[out_mask], all_w[out_mask])
                q_out_eff = q_out_cand if math.isfinite(q_out_cand) else q_marg
            else:
                q_out_eff = q_marg
        else:
            rank_k = max(1, effective_n_calib + 1)
            q_marg = q_in_eff = q_out_eff = float("inf")

        const_viol = bool(advisor._constant_invariant_violated(x_cand.reshape(1, -1))[0])
        domain_ok = (
            math.isfinite(knn_distance)
            and math.isfinite(effective_threshold)
            and knn_distance >= 0.0
            and effective_threshold > 0.0
            and (knn_distance <= effective_threshold)
            and (not const_viol)
        )
        q_used = q_in_eff if domain_ok else q_out_eff
        rank_ok = (
            alpha_valid
            and weight_valid
            and effective_n_calib > 0
            and (rank_k <= effective_n_calib)
            and math.isfinite(q_used)
        )
        abstain = not (rank_ok and domain_ok)

        raw_pred = float(advisor.model.predict(x_cand.reshape(1, -1))[0])
        if math.isfinite(q_used) and math.isfinite(knn_distance) and knn_distance >= 0.0:
            half = max(q_used, 0.0) * (1.0 + advisor.gamma * knn_distance)
            lo_c = max(PHYS_MIN_LOGS, min(PHYS_MAX_LOGS, raw_pred - half))
            hi_c = max(lo_c, min(PHYS_MAX_LOGS, raw_pred + half))
            low = float(lo_c)
            high = float(hi_c)
            point_pred = float(max(low, min(high, max(PHYS_MIN_LOGS, min(PHYS_MAX_LOGS, raw_pred)))))
        else:
            low = PHYS_MIN_LOGS
            high = PHYS_MAX_LOGS
            point_pred = float(max(PHYS_MIN_LOGS, min(PHYS_MAX_LOGS, raw_pred)))

    if abstain:
        out["halted"] = True
        out["halt_reason"] = "L6_ABSTAIN"

    out["decision"] = {
        "candidate_smiles": candidate_smiles,
        "in_trust_region": not abstain,
        "abstain": abstain,
        "model_encoding": state.model_encoding,
        "surrogate": state.surrogate_name,
        "distance_source": distance_source,
        "knn_distance": knn_distance,
        "effective_threshold": effective_threshold if math.isfinite(effective_threshold) else None,
        "effective_n_calib": effective_n_calib,
        "predicted_log_s": point_pred,
        "conformal_interval": [low, high],
        "nonconformity": advisor.nonconformity,
        "certified_calibration_added": len(measured_obs),
        "unmeasured_ledger_entries_ignored": unmeasured_ignored,
        "duplicate_ledger_entries_ignored": duplicate_ignored,
        "invalid_ledger_entries_ignored": invalid_ignored,
    }

    if abstain:
        thr_str = f"{effective_threshold:.4f}" if math.isfinite(effective_threshold) else "inf"
        detail = (
            f"rank_k={rank_k} > effective_n_calib={effective_n_calib} or insufficient weighted mass"
            if not rank_ok
            else f"knn_distance={knn_distance:.4f} > effective_threshold={thr_str}"
        )
        summary = (
            f"Conformal decision gate abstained ({detail}; "
            f"encoding={state.model_encoding}, "
            f"distance_source={distance_source}, "
            f"certified_recalibration_count={len(measured_obs)}, "
            f"duplicates_ignored={duplicate_ignored})."
        )
    else:
        summary = (
            f"Conformal decision gate passed (knn_distance={knn_distance:.4f} <= "
            f"effective_threshold={effective_threshold:.4f}, encoding={state.model_encoding}, "
            f"distance_source={distance_source}, "
            f"rank={rank_k}/{effective_n_calib}, "
            f"interval=[{low:.4f}, {high:.4f}], "
            f"certified_recalibration_count={len(measured_obs)}, "
            f"duplicates_ignored={duplicate_ignored})."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "ABSTAIN" if abstain else "PASS",
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": "out_of_domain_always_abstains",
            "first_principles": (
                "Distribution-free Mondrian normalized split-conformal coverage at level (1 - alpha) "
                "scores candidate SMILES with the SHA-256-pinned ESOL ConformalTrustModel "
                "(rdkit_2d_morgan + frontier_ensemble surrogate over the canonical 6D ESOL kNN domain), "
                "enforcing exact rational finite-sample rank ceil((n + 1) * (1 - alpha)) <= n without "
                "clamping and kNN applicability-domain membership d_kNN(x) <= tau; certified ledger "
                "observations recompute nonconformity quantiles and never widen tau on unmeasured or "
                "high-residual entries."
            ),
        }
    )
    return out


if __name__ == "__main__":
    print(json.dumps(process_packet({}), indent=2))
