"""Community research extensions for lab-decide (L6 conformal & BO advisor).

Modules:
- mondrian_drift_conformal: Scaffold-Mondrian Online Adaptive Conformal Calibrator (Idea #03).
- chem_rrt_frontier: Stepped Annular Chem-RRT Frontier Explorer (Idea #04).
"""

from __future__ import annotations

from .chem_rrt_frontier import (
    CandidateCompound,
    ChemRRTExplorer,
    FrontierActionRecord,
    TreeEdge,
    verify_frontier_action,
)
from .chem_rrt_frontier import (
    run_benchmark as run_chem_rrt_benchmark,
)
from .mondrian_drift_conformal import (
    ACIIntervalPrediction,
    ScaffoldMondrianACI,
    StaticSplitConformal,
    conformal_order_rank,
    finite_sample_conformal_quantile,
)
from .mondrian_drift_conformal import (
    run_benchmark as run_mondrian_aci_benchmark,
)

__all__ = [
    "ACIIntervalPrediction",
    "CandidateCompound",
    "ChemRRTExplorer",
    "FrontierActionRecord",
    "ScaffoldMondrianACI",
    "StaticSplitConformal",
    "TreeEdge",
    "conformal_order_rank",
    "finite_sample_conformal_quantile",
    "run_chem_rrt_benchmark",
    "run_mondrian_aci_benchmark",
    "verify_frontier_action",
]
