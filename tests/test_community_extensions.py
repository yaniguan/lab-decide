"""Unit and benchmark tests for lab-decide community_extensions (Ideas #03 and #04)."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_ce = importlib.import_module("community_extensions")
CandidateCompound = _ce.CandidateCompound
ChemRRTExplorer = _ce.ChemRRTExplorer
ScaffoldMondrianACI = _ce.ScaffoldMondrianACI
StaticSplitConformal = _ce.StaticSplitConformal
conformal_order_rank = _ce.conformal_order_rank
finite_sample_conformal_quantile = _ce.finite_sample_conformal_quantile
run_chem_rrt_benchmark = _ce.run_chem_rrt_benchmark
run_mondrian_aci_benchmark = _ce.run_mondrian_aci_benchmark
verify_frontier_action = _ce.verify_frontier_action


def test_finite_sample_conformal_rank_never_clamps_undersized() -> None:
    assert conformal_order_rank(5, 0.10) == 6
    assert math.isinf(finite_sample_conformal_quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.10))

    static_small = StaticSplitConformal([0.1, 0.2, 0.3, 0.4, 0.5], alpha_target=0.10)
    pred_small = static_small.predict(-3.0)
    assert pred_small.abstained is True

    # With n = 19 at alpha = 0.10 -> ceil(20 * 0.90) = 18 <= 19
    res_19 = [float(i) * 0.1 for i in range(1, 20)]
    assert conformal_order_rank(19, 0.10) == 18
    assert finite_sample_conformal_quantile(res_19, 0.10) == pytest.approx(
        1.8, abs=1e-9
    )


def test_mondrian_aci_step_rule_and_clipping() -> None:
    aci = ScaffoldMondrianACI(
        {"aryl": [0.01 * i for i in range(1, 101)]},
        alpha_target=0.10,
        gamma=0.05,
        alpha_bounds=(0.01, 0.50),
        rolling_window=100,
    )
    pred = aci.predict(-4.0, "aryl")
    assert pred.abstained is False
    assert pred.alpha_used == pytest.approx(0.10)

    # Miscoverage error (err_t = 1): alpha_{t+1} = 0.10 + 0.05 * (0.10 - 1.0) = 0.055
    next_alpha = aci.update(pred, y_true=1.5)
    assert next_alpha == pytest.approx(0.055, abs=1e-9)

    # Repeated escalating misses clip cleanly at 0.01 without going negative
    for k in range(1, 6):
        p = aci.predict(-12.0, "aryl")
        next_alpha = aci.update(p, y_true=-12.0 + 2.0 * (k + 3))
    assert next_alpha == pytest.approx(0.01, abs=1e-9)

    # Repeated hits move alpha upward and clip at 0.50
    aci_tight = ScaffoldMondrianACI(
        {"aryl": [0.5] * 100},
        alpha_target=0.10,
        gamma=0.50,
        alpha_bounds=(0.01, 0.50),
    )
    for _ in range(15):
        p = aci_tight.predict(-4.0, "aryl")
        next_alpha = aci_tight.update(p, y_true=-4.0)
    assert next_alpha == pytest.approx(0.50, abs=1e-9)


def test_mondrian_aci_benchmark_restores_coverage_under_drift() -> None:
    summary = run_mondrian_aci_benchmark(n_steps=300, alpha_target=0.10, seed=42)
    assert summary["n_steps"] == 300
    static_m = summary["static_split_conformal"]
    aci_m = summary["mondrian_aci"]
    assert isinstance(static_m, dict) and isinstance(aci_m, dict)

    # Static split conformal collapses well below 90% under temporal + scaffold drift
    assert static_m["empirical_coverage"] < 0.75
    # Mondrian ACI tracks the 90% target closely across all scaffolds
    assert 0.86 <= aci_m["empirical_coverage"] <= 0.94
    assert aci_m["abs_coverage_error"] < static_m["abs_coverage_error"]
    assert aci_m["worst_scaffold_coverage"] >= 0.84
    assert aci_m["winkler_efficiency_score"] < static_m["winkler_efficiency_score"]


def test_verify_frontier_action_mirrors_lean_guardrails() -> None:
    # Exploitation inside tau -> OK; outside tau -> rejected
    assert verify_frontier_action(
        mode="exploitation", nn_dist=0.65, tau_radius=0.80
    ) == (True, None)
    assert verify_frontier_action(
        mode="exploitation", nn_dist=0.95, tau_radius=0.80
    ) == (
        False,
        "exploitation_inside_tau",
    )

    # Valid annular stepping stone -> OK
    assert verify_frontier_action(
        mode="frontier_calibration",
        nn_dist=0.95,
        tau_radius=0.80,
        eta_step=0.25,
        max_eta=0.35,
        assay_vol_ul=8.0,
        micro_vol_cap_ul=10.0,
        is_direct_buy=True,
        used_as_point_prediction=False,
    ) == (True, None)

    # Deep OOD (> (1 + eta)*tau = 1.00) -> rejected
    assert verify_frontier_action(
        mode="frontier_calibration",
        nn_dist=1.25,
        tau_radius=0.80,
        eta_step=0.25,
    ) == (False, "within_stepping_stone_shell")

    # Micro-volume cap violated -> rejected
    assert verify_frontier_action(
        mode="frontier_calibration",
        nn_dist=0.95,
        tau_radius=0.80,
        eta_step=0.25,
        assay_vol_ul=15.0,
        micro_vol_cap_ul=10.0,
    ) == (False, "frontier_micro_volume_cap")

    # Synthesis-required compound forbidden as frontier probe -> rejected
    assert verify_frontier_action(
        mode="frontier_calibration",
        nn_dist=0.95,
        tau_radius=0.80,
        eta_step=0.25,
        is_direct_buy=False,
    ) == (False, "frontier_requires_direct_stock")


def test_chem_rrt_explorer_never_uses_unpurchasable_decoys() -> None:
    origin = CandidateCompound("S0", "core", (0.0, 0.0), 1, is_direct_buy=True)
    synth_decoy = CandidateCompound(
        "SYN_DECOY", "mid", (0.95, 0.0), 2, is_direct_buy=False
    )
    stock_step = CandidateCompound("BB_STEP", "mid", (0.98, 0.0), 2, is_direct_buy=True)
    target = CandidateCompound(
        "NOVEL_T", "macro", (1.65, 0.0), 4, is_direct_buy=False, is_novel_target=True
    )

    explorer = ChemRRTExplorer([origin], tau_radius=0.80, eta_step=0.30)
    unlocked = explorer.expand_toward_targets([synth_decoy, stock_step], [target])
    assert [t.compound_id for t in unlocked] == ["NOVEL_T"]
    assert "SYN_DECOY" not in explorer.support_ids
    assert "BB_STEP" in explorer.support_ids
    assert all(rec.guard_passed for rec in explorer.action_log)


def test_chem_rrt_benchmark_metrics() -> None:
    summary = run_chem_rrt_benchmark(seed=42)
    assert summary["n_novel_target_scaffolds"] == 12
    static_w = summary["static_wall"]
    leap_m = summary["unbounded_leap"]
    rrt_m = summary["chem_rrt_stepping"]
    assert (
        isinstance(static_w, dict)
        and isinstance(leap_m, dict)
        and isinstance(rrt_m, dict)
    )

    assert static_w["novel_scaffolds_unlocked"] == 0
    assert leap_m["uncalibrated_ood_leaps"] == 12
    assert rrt_m["novel_scaffolds_unlocked"] == 12
    assert rrt_m["uncalibrated_ood_leaps"] == 0
