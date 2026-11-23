"""Typed DecisionEnvelope JSON contract between lab-ask-why (L6), lab-decide (L6), labshift (L5), and lab-rehearsal (L4)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from lab_decide.chemistry import ESOL_FEATURES, featurize_smiles_batch
from lab_decide.conformal import ConformalTrustModel

DECISION_VALUES = ("ANSWER", "ABSTAIN_ACQUIRE")


def _json_safe_float(v: float) -> float | None:
    return round(float(v), 6) if math.isfinite(float(v)) else None


@dataclass(frozen=True)
class DecisionEnvelope:
    """Per-candidate conformal decision record for L6 -> L5 -> L4 campaign handoff."""

    candidate_id: str
    prediction: float
    interval_lo: float
    interval_hi: float
    knn_distance: float
    trust_threshold: float
    trusted: bool
    decision: str
    confidence_alpha: float
    source_claim: str | None = None
    smiles: str | None = None
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.decision not in DECISION_VALUES:
            raise ValueError(f"decision {self.decision!r} must be one of {DECISION_VALUES}")
        if not (0.0 < float(self.confidence_alpha) < 1.0):
            raise ValueError(f"confidence_alpha must be in (0, 1), got {self.confidence_alpha}")
        object.__setattr__(self, "prediction", round(float(self.prediction), 6))
        if math.isfinite(float(self.interval_lo)):
            object.__setattr__(self, "interval_lo", round(float(self.interval_lo), 6))
        if math.isfinite(float(self.interval_hi)):
            object.__setattr__(self, "interval_hi", round(float(self.interval_hi), 6))
        object.__setattr__(self, "knn_distance", round(float(self.knn_distance), 6))
        if math.isfinite(float(self.trust_threshold)):
            object.__setattr__(self, "trust_threshold", round(float(self.trust_threshold), 6))
        object.__setattr__(self, "confidence_alpha", round(float(self.confidence_alpha), 6))

    @property
    def interval_width(self) -> float:
        return float(self.interval_hi - self.interval_lo)

    @property
    def distance_ratio(self) -> float:
        if self.trust_threshold <= 0.0 or not math.isfinite(self.trust_threshold):
            return 0.0 if self.knn_distance == 0.0 else float("inf")
        return float(self.knn_distance / self.trust_threshold)

    def to_dict(self) -> dict[str, Any]:
        lo_val = _json_safe_float(self.interval_lo)
        hi_val = _json_safe_float(self.interval_hi)
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "prediction": round(float(self.prediction), 6),
            "interval_bounds": [lo_val, hi_val],
            "interval_90": [lo_val, hi_val] if math.isclose(self.confidence_alpha, 0.10) else [lo_val, hi_val],
            "interval_width": _json_safe_float(self.interval_width),
            "knn_distance": round(float(self.knn_distance), 6),
            "trust_threshold": _json_safe_float(self.trust_threshold),
            "distance_ratio": _json_safe_float(self.distance_ratio),
            "trusted": bool(self.trusted),
            "decision": self.decision,
            "confidence_alpha": round(float(self.confidence_alpha), 6),
            "source_claim": self.source_claim,
            "smiles": self.smiles,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionEnvelope":
        bounds = data.get("interval_bounds") or data.get("interval_90") or [None, None]
        lo = float("-inf") if bounds[0] is None else float(bounds[0])
        hi = float("inf") if bounds[1] is None else float(bounds[1])
        thr_raw = data.get("trust_threshold")
        thr = float("inf") if thr_raw is None else float(thr_raw)
        return cls(
            schema_version=str(data.get("schema_version", "1.0.0")),
            candidate_id=str(data["candidate_id"]),
            prediction=float(data["prediction"]),
            interval_lo=lo,
            interval_hi=hi,
            knn_distance=float(data["knn_distance"]),
            trust_threshold=thr,
            trusted=bool(data["trusted"]),
            decision=str(data["decision"]),
            confidence_alpha=float(data["confidence_alpha"]),
            source_claim=data.get("source_claim"),
            smiles=data.get("smiles"),
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        nullable_num = {"type": ["number", "null"]}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/yaniguan/lab-decide/schemas/decision-envelope-v1.json",
            "title": "DecisionEnvelope",
            "type": "object",
            "required": [
                "schema_version",
                "candidate_id",
                "prediction",
                "interval_bounds",
                "interval_90",
                "interval_width",
                "knn_distance",
                "trust_threshold",
                "distance_ratio",
                "trusted",
                "decision",
                "confidence_alpha",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "candidate_id": {"type": "string", "minLength": 1},
                "prediction": {"type": "number"},
                "interval_bounds": {"type": "array", "items": nullable_num, "minItems": 2, "maxItems": 2},
                "interval_90": {"type": "array", "items": nullable_num, "minItems": 2, "maxItems": 2},
                "interval_width": nullable_num,
                "knn_distance": {"type": "number", "minimum": 0.0},
                "trust_threshold": nullable_num,
                "distance_ratio": nullable_num,
                "trusted": {"type": "boolean"},
                "decision": {"type": "string", "enum": list(DECISION_VALUES)},
                "confidence_alpha": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "source_claim": {"type": ["string", "null"]},
                "smiles": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        }


DECISION_ENVELOPE_JSON_SCHEMA = DecisionEnvelope.json_schema()


def build_decision_envelopes(
    advisor: ConformalTrustModel,
    X: np.ndarray,
    *,
    candidate_ids: Sequence[str] | None = None,
    smiles: Sequence[str | None] | None = None,
    source_claims: Sequence[str | None] | None = None,
) -> list[DecisionEnvelope]:
    """Evaluate `X` with `advisor` and emit typed `DecisionEnvelope` objects."""
    X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = X_arr.shape[0]
    res = advisor.predict(X_arr, use_mondrian=True)
    ids = list(candidate_ids) if candidate_ids is not None else [f"cand-{i:04d}" for i in range(n)]
    smis = list(smiles) if smiles is not None else [None] * n
    claims = list(source_claims) if source_claims is not None else [None] * n

    envelopes: list[DecisionEnvelope] = []
    for i in range(n):
        envelopes.append(
            DecisionEnvelope(
                candidate_id=str(ids[i]),
                prediction=float(res["pred"][i]),
                interval_lo=float(res["lo"][i]),
                interval_hi=float(res["hi"][i]),
                knn_distance=float(res["distance"][i]),
                trust_threshold=float(advisor.threshold),
                trusted=bool(res["trusted"][i]),
                decision=str(res["decision"][i]),
                confidence_alpha=float(advisor.alpha),
                source_claim=claims[i],
                smiles=smis[i],
            )
        )
    return envelopes


def evaluate_epistemic_probes(
    advisor: ConformalTrustModel,
    probe_records: Sequence[Mapping[str, Any]],
) -> list[DecisionEnvelope]:
    """Consume `EpistemicProbeResult` dicts from `lab-ask-why` and emit `DecisionEnvelope` gates."""
    rows: list[np.ndarray] = []
    ids: list[str] = []
    claims: list[str | None] = []
    smis: list[str | None] = []

    for idx, rec in enumerate(probe_records):
        obs = rec.get("discriminating_observable") or {}
        desc = obs.get("candidate_descriptors") or {}
        cand_smi = obs.get("candidate_smiles")
        if all(f in desc for f in advisor.feature_names):
            vec = np.asarray([float(desc[f]) for f in advisor.feature_names], dtype=np.float64)
        elif cand_smi:
            vec = featurize_smiles_batch([str(cand_smi)], feature_names=advisor.feature_names)[0]
        else:
            raise ValueError(
                f"Probe record {idx} lacks both complete candidate_descriptors for "
                f"{advisor.feature_names} and candidate_smiles"
            )
        rows.append(vec)
        ids.append(str(obs.get("observable_name") or f"probe-{idx:04d}"))
        claims.append(str(rec.get("claim")) if rec.get("claim") else None)
        smis.append(str(cand_smi) if cand_smi else None)

    if not rows:
        return []
    return build_decision_envelopes(
        advisor,
        np.vstack(rows),
        candidate_ids=ids,
        smiles=smis,
        source_claims=claims,
    )


def export_decisions_jsonl(envelopes: Sequence[DecisionEnvelope], path: str) -> int:
    with open(path, mode="w", encoding="utf-8") as fh:
        for env in envelopes:
            fh.write(env.to_json() + "\n")
    return len(envelopes)


def load_decisions_jsonl(path: str) -> list[DecisionEnvelope]:
    with open(path, mode="r", encoding="utf-8") as fh:
        return [DecisionEnvelope.from_dict(json.loads(line)) for line in fh if line.strip()]
