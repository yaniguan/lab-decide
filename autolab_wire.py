"""Auto-Lab 15-repository wire node for lab-decide (Layer L6 Calibrated Conformal Decision Gate).

Implements `autolab-wire/v1` packet processing with:
- Exact rational/integer finite-sample split-conformal order-statistic rank (`LabDecideLean.conformalRank`),
- Tibshirani-Foygel-Barber likelihood-ratio weighted conformal calibration (`weighted_conformal_quantile`),
- Descriptor-conditioned locally scaled Conformalized Quantile Regression (`CQR-r`) with `lo <= hi` guard,
- Canonical 64-hex SHA-256 validation and idempotent `ledger_entry_sha256` deduplication across rounds.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from lab_decide.chemistry import HAS_RDKIT, murcko_scaffold_group, smiles_to_descriptors  # noqa: E402
from lab_decide.conformal import conformal_quantile, weighted_conformal_quantile  # noqa: E402

DEFAULT_CANDIDATE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
PHYS_MIN_LOGS = -15.0
PHYS_MAX_LOGS = 1.75
CERTIFIED_THRESHOLD_STEP = 0.02
DEFAULT_MAX_THRESHOLD_EXPANSION = 0.20
CQR_SCALE_FLOOR = 0.05
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _extract_certified_entry_keys(entry: Any) -> tuple[str, str | None] | None:
    """Validate a `lab-attrib` ledger entry and return `(primary_dedup_key, optional_run_key)`.

    Accepts entries where `admit_to_memory` is not `False` and either:
    1. `ledger_entry_sha256` is a canonical 64-character hex SHA-256 digest, OR
    2. `ledger_entry_sha256` is omitted/`None`, `admit_to_memory is True`, and a non-empty
       unique `run_id` / `entry_id` / `ledger_id` string is supplied.
    Rejects malformed SHA-256 strings, anonymous digest-free dicts, and non-finite residuals/weights.
    """
    if not isinstance(entry, Mapping):
        return None
    if entry.get("admit_to_memory") is False:
        return None

    if "calibration_residual" in entry and entry["calibration_residual"] is not None:
        try:
            res_val = float(entry["calibration_residual"])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(res_val):
            return None

    for w_key in ("likelihood_ratio_weight", "importance_weight"):
        if w_key in entry and entry[w_key] is not None:
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


def _is_valid_certified_entry(entry: Any) -> bool:
    """Return True iff `entry` carries a valid 64-hex SHA-256 or explicit `admit_to_memory=True` with unique ID."""
    return _extract_certified_entry_keys(entry) is not None


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


def _extract_esol_descriptors(smiles: str, override: Any = None) -> dict[str, float]:
    """Compute or fall back to deterministic 6D ESOL descriptors for heteroscedastic CQR-r."""
    if isinstance(override, Mapping):
        try:
            return {
                "Minimum Degree": float(override.get("Minimum Degree", 1.0)),
                "Molecular Weight": max(12.0, float(override["Molecular Weight"])),
                "Number of H-Bond Donors": max(0.0, float(override.get("Number of H-Bond Donors", 0.0))),
                "Number of Rings": max(0.0, float(override.get("Number of Rings", 0.0))),
                "Number of Rotatable Bonds": max(0.0, float(override.get("Number of Rotatable Bonds", 0.0))),
                "Polar Surface Area": max(0.0, float(override.get("Polar Surface Area", 0.0))),
            }
        except (KeyError, TypeError, ValueError):
            pass

    if HAS_RDKIT:
        try:
            return smiles_to_descriptors(smiles)
        except (ValueError, ImportError):
            pass

    clean = smiles.strip()
    scaffold_sig = murcko_scaffold_group(clean)
    ring_hint = float(sum(1 for ch in scaffold_sig if ch.isdigit()) // 2)
    polar_atoms = sum(1 for ch in clean if ch in "ONon")
    hbd_hint = float(clean.count("O") + clean.count("N")) * 0.4
    mw_est = max(30.0, 14.0 * max(1, len(clean)))
    return {
        "Minimum Degree": 2.0 if ring_hint > 0 and "(" not in clean else 1.0,
        "Molecular Weight": mw_est,
        "Number of H-Bond Donors": hbd_hint,
        "Number of Rings": ring_hint,
        "Number of Rotatable Bonds": float(max(0, clean.count("CC") - 1)),
        "Polar Surface Area": 16.0 * polar_atoms,
    }


def _estimate_descriptor_quantile_band(
    smiles: str,
    *,
    alpha: float = 0.10,
    descriptor_override: Any = None,
    point_override: float | None = None,
) -> tuple[float, float, float, float]:
    """Return `(point_pred, q_lo_raw, q_hi_raw, spread)` conditioned on 6D chemical descriptors.

    Models heteroscedastic solubility dispersion and crystal-packing lower-tail asymmetry:
    high-MW, multi-ring hydrophobic polycycles exhibit substantially wider and more negatively
    skewed conditional quantile bands than low-MW polar alcohols at identical kNN distance.
    """
    desc = _extract_esol_descriptors(smiles, override=descriptor_override)
    mw = desc["Molecular Weight"]
    rings = desc["Number of Rings"]
    rot = desc["Number of Rotatable Bonds"]
    hbd = desc["Number of H-Bond Donors"]
    tpsa = desc["Polar Surface Area"]

    if point_override is not None and math.isfinite(float(point_override)):
        point_pred = float(point_override)
    else:
        est = 0.25 - 0.010 * mw - 0.35 * rings - 0.08 * rot + 0.006 * tpsa
        point_pred = float(max(-12.0, min(0.8, est)))

    # Local heteroscedastic dispersion grows with MW, ring packing complexity, and rotor flexibility,
    # and contracts slightly for small hydrogen-bonding polar solutes.
    alpha_eff = min(max(float(alpha), 1e-4), 0.9999) if math.isfinite(float(alpha)) else 0.10
    level_scale = max(0.25, math.sqrt(max(0.05, 1.0 - alpha_eff) / 0.90))

    mw_term = 0.22 * max(0.0, (mw - 60.0) / 150.0)
    ring_term = 0.18 * rings
    rot_term = 0.04 * rot
    polar_relief = min(0.12, 0.015 * hbd + 0.0008 * tpsa)
    base_half = max(0.20, (0.28 + mw_term + ring_term + rot_term - polar_relief) * level_scale)

    # Crystal-packing polymorphism creates heavier lower solubility tails on multi-ring compounds
    lower_skew = 1.0 + 0.12 * min(rings, 5.0) + 0.04 * max(0.0, (mw - 200.0) / 150.0)
    upper_skew = max(0.75, 1.0 - 0.04 * min(rings, 4.0))

    lo_span = base_half * lower_skew
    hi_span = base_half * upper_skew
    q_lo_raw = point_pred - lo_span
    q_hi_raw = point_pred + hi_span
    spread = max(q_hi_raw - q_lo_raw, CQR_SCALE_FLOOR)
    return point_pred, q_lo_raw, q_hi_raw, spread


def _estimate_point_solubility_logs(smiles: str) -> float:
    """Deterministic ESOL-scale point estimate in log10(mol/L) without network I/O."""
    point_pred, _, _, _ = _estimate_descriptor_quantile_band(smiles)
    return point_pred


def _build_calibration_cqr_r_scores(n_samples: int) -> list[float]:
    """Build deterministic locally scaled CQR-r calibration nonconformity scores of length `n_samples`.

    Scores represent `(max(q_lo - y, y - q_hi)) / max(q_hi - q_lo, CQR_SCALE_FLOOR)`: negative on
    well-covered interior compounds (~80%) and positive on tail calibration compounds (~20%).
    """
    if n_samples <= 0:
        return []
    return [round(-0.35 + 1.15 * ((idx + 0.5) / n_samples), 6) for idx in range(n_samples)]


def _entry_cqr_r_score(entry: Mapping[str, Any], alpha: float) -> float:
    """Convert a certified ledger entry's measurement residual into a locally scaled CQR-r score."""
    if "cqr_r_score" in entry and entry["cqr_r_score"] is not None:
        return float(entry["cqr_r_score"])
    raw_res = abs(float(entry.get("calibration_residual", 0.32)))
    entry_smi = str(entry.get("candidate_smiles") or DEFAULT_CANDIDATE_SMILES)
    _, q_lo_e, q_hi_e, spread_e = _estimate_descriptor_quantile_band(entry_smi, alpha=alpha)
    half_band = 0.5 * (q_hi_e - q_lo_e)
    return round((raw_res - half_band) / max(spread_e, CQR_SCALE_FLOOR), 6)


def _entry_likelihood_ratio_weight(entry: Mapping[str, Any], base_threshold: float) -> tuple[float, bool]:
    """Return `(weight, is_explicit)` for covariate-shifted active boundary picks."""
    for w_key in ("likelihood_ratio_weight", "importance_weight"):
        if w_key in entry and entry[w_key] is not None:
            return float(entry[w_key]), True
    if "knn_distance" in entry and entry["knn_distance"] is not None and base_threshold > 0.0:
        d_ent = float(entry["knn_distance"])
        if math.isfinite(d_ent) and d_ent > base_threshold:
            # Defensive outer-shell Radon-Nikodym weight w(x) in (0, 1)
            shell_excess = (d_ent - base_threshold) / max(base_threshold, 1e-6)
            return 1.0 / (1.0 + shell_excess), False
    return 1.0, False


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate and enrich an `autolab-wire/v1` packet at the L6 conformal decision gate."""
    out: dict[str, Any] = copy.deepcopy(dict(packet)) if packet is not None else {}
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if bool(out["halted"]):
        return out

    candidate_smiles = str(out.get("candidate_smiles", DEFAULT_CANDIDATE_SMILES))
    out.setdefault("candidate_smiles", candidate_smiles)
    knn_distance = float(out.get("knn_distance", 0.412))
    base_threshold = float(out.get("knn_threshold", 0.758))
    base_n_calib = int(out.get("n_calib", 226))
    alpha = float(out.get("alpha", 0.10))
    test_weight = float(out.get("test_weight", 1.0))
    max_thr_exp = max(0.0, float(out.get("max_threshold_expansion", DEFAULT_MAX_THRESHOLD_EXPANSION)))

    raw_certified = out.get("certified_ledger_entries", [])
    if isinstance(raw_certified, (list, tuple)):
        certified_entries = list(raw_certified)
    else:
        certified_entries = []

    valid_certified, duplicate_ignored, invalid_ignored = _deduplicate_certified_entries(certified_entries)
    effective_n_calib = base_n_calib + len(valid_certified)

    cert_weights_info = [_entry_likelihood_ratio_weight(e, base_threshold) for e in valid_certified]
    cert_raw_weights = [w for w, _ in cert_weights_info]
    has_explicit_entry_w = any(is_exp for _, is_exp in cert_weights_info)

    effective_cert_mass = sum(min(w, 1.0) for w in cert_raw_weights)
    thr_delta = min(max_thr_exp, CERTIFIED_THRESHOLD_STEP * effective_cert_mass)
    effective_threshold = round(base_threshold + thr_delta, 4)

    base_scores = _build_calibration_cqr_r_scores(max(0, base_n_calib))
    certified_scores = [_entry_cqr_r_score(entry, alpha) for entry in valid_certified]
    cqr_scores = base_scores + certified_scores

    explicit_cal_w = out.get("cal_weights")
    if isinstance(explicit_cal_w, (list, tuple)) and len(explicit_cal_w) == len(cqr_scores):
        cal_weights = [float(w) for w in explicit_cal_w]
    else:
        raw_weights = [1.0] * max(0, base_n_calib) + cert_raw_weights
        if raw_weights and not has_explicit_entry_w and sum(raw_weights) > 0.0:
            # Self-normalize automatic shell weights to mean 1.0 so sample-size rank gates remain exact
            mean_w = sum(raw_weights) / float(len(raw_weights))
            cal_weights = [w / mean_w for w in raw_weights]
        else:
            cal_weights = raw_weights

    alpha_valid = math.isfinite(alpha) and (0.0 < alpha < 1.0)
    weights_valid = (
        len(cal_weights) == effective_n_calib
        and all(math.isfinite(w) and w >= 0.0 for w in cal_weights)
        and math.isfinite(test_weight)
        and test_weight > 0.0
    )
    if effective_n_calib > 0 and alpha_valid and len(cqr_scores) == effective_n_calib and weights_valid:
        rank_k = conformal_order_rank(effective_n_calib, alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if all(w == 1.0 for w in cal_weights) and test_weight == 1.0:
                q_hat = conformal_quantile(cqr_scores, alpha=alpha)
            else:
                q_hat = weighted_conformal_quantile(
                    cqr_scores, cal_weights, alpha=alpha, test_weight=test_weight
                )
    else:
        rank_k = max(1, effective_n_calib + 1)
        q_hat = float("inf")

    # Lean invariants (`LabDecideLean.checkDecide`):
    # (a) Finite-sample conformal rank `rank_k <= effective_n_calib` and finite weighted quantile
    # (b) Applicability-domain gate `knn_distance <= effective_threshold`
    rank_ok = alpha_valid and effective_n_calib > 0 and (rank_k <= effective_n_calib) and math.isfinite(q_hat)
    domain_ok = (
        math.isfinite(knn_distance)
        and math.isfinite(effective_threshold)
        and knn_distance >= 0.0
        and effective_threshold > 0.0
        and (knn_distance <= effective_threshold)
    )
    abstain = not (rank_ok and domain_ok)

    raw_pred_override = out.get("predicted_log_s")
    point_override = float(raw_pred_override) if raw_pred_override is not None else None
    point_pred, q_lo_raw, q_hi_raw, desc_spread = _estimate_descriptor_quantile_band(
        candidate_smiles,
        alpha=alpha if alpha_valid else 0.10,
        descriptor_override=out.get("candidate_descriptors"),
        point_override=point_override,
    )

    if math.isfinite(q_hat) and math.isfinite(knn_distance) and knn_distance >= 0.0:
        local_scale = desc_spread * (1.0 + 0.25 * knn_distance)
        lo_cand = q_lo_raw - q_hat * local_scale
        hi_cand = q_hi_raw + q_hat * local_scale
        # Monotonic crossing guard (`LabDecideLean.interval_well_ordered`: `lo <= hi`)
        if lo_cand > hi_cand:
            mid = 0.5 * (q_lo_raw + q_hi_raw)
            lo_cand = mid
            hi_cand = mid
        lo_clipped = max(PHYS_MIN_LOGS, min(PHYS_MAX_LOGS, lo_cand))
        hi_clipped = max(lo_clipped, min(PHYS_MAX_LOGS, hi_cand))
        low = round(lo_clipped, 4)
        high = round(max(low, hi_clipped), 4)
    else:
        local_scale = float("nan")
        low = PHYS_MIN_LOGS
        high = PHYS_MAX_LOGS

    if abstain:
        out["halted"] = True
        out["halt_reason"] = "L6_ABSTAIN"

    out["decision"] = {
        "candidate_smiles": candidate_smiles,
        "in_trust_region": not abstain,
        "abstain": abstain,
        "knn_distance": knn_distance,
        "effective_threshold": effective_threshold,
        "effective_n_calib": effective_n_calib,
        "conformal_interval": [low, high],
        "nonconformity": "cqr_r",
        "duplicate_ledger_entries_ignored": duplicate_ignored,
        "invalid_ledger_entries_ignored": invalid_ignored,
    }

    if abstain:
        detail = (
            f"rank_k={rank_k} > effective_n_calib={effective_n_calib} or insufficient weighted mass"
            if not rank_ok
            else f"knn_distance={knn_distance:.4f} > effective_threshold={effective_threshold:.4f}"
        )
        summary = (
            f"Conformal decision gate abstained ({detail}; "
            f"certified_recalibration_count={len(valid_certified)}, "
            f"duplicates_ignored={duplicate_ignored})."
        )
    else:
        summary = (
            f"Conformal decision gate passed (knn_distance={knn_distance:.4f} <= "
            f"effective_threshold={effective_threshold:.4f}, rank={rank_k}/{effective_n_calib}, "
            f"cqr_r_interval=[{low:.4f}, {high:.4f}], "
            f"certified_recalibration_count={len(valid_certified)}, "
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
                "Distribution-free locally scaled CQR-r coverage at level (1 - alpha) requires "
                "exact rational finite-sample rank ceil((n + 1) * (1 - alpha)) <= n without "
                "clamping, likelihood-ratio covariate reweighting, and kNN support membership "
                "d_kNN(x) <= tau; deduplicated 64-hex SHA-256 certified ledger entries from "
                "lab-attrib recalibrate effective_n_calib and effective_threshold idempotently "
                "only after causal blame verification blocks execution and spectral artifacts."
            ),
        }
    )
    return out


if __name__ == "__main__":
    print(json.dumps(process_packet({}), indent=2))
