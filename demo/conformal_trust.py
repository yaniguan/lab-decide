"""Mondrian split conformal prediction, kNN trust region, and pool-separated expansion on ESOL.

Evaluates Huber gradient boosting with finite-sample Mondrian conformal calibration,
importance-weighted kNN applicability-domain gating, diverse boundary acquisition
from a disjoint candidate pool (holding the test split untouched), and trust-region
constrained Bayesian optimization on ESOL (Delaney, 2004).

Run:
  python demo/conformal_trust.py
  python demo/conformal_trust.py --nonconformity cqr --split scaffold
  python demo/conformal_trust.py --dataset calisol23 --offline
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_decide.acquisition import (
    TrustRegionAdvisor,
    run_expansion_campaign,
    select_diverse_boundary_batch,
)
from lab_decide.chemistry import ESOL_FEATURES as FEATURES, ESOL_TARGET as TARGET
from lab_decide.cli import main as cli_main
from lab_decide.conformal import (
    ConformalTrustModel,
    conformal_quantile,
    conformal_threshold,
    fit_and_calibrate,
    summarize,
)
from lab_decide.data import (
    DEFAULT_DATA_PATH as DATA,
    ESOL_SHA256,
    ESOL_URL as URL,
    load,
    load_dataset,
    partition_splits,
)
from lab_decide.schema import DecisionEnvelope

SEED = 0
ALPHA = 0.10
K = 5
TRUST_PCT = 80
ROUNDS, PER_ROUND = 3, 20


def main(argv: Sequence[str] | None = None) -> int:
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
