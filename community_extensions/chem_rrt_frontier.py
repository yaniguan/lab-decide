"""Stepped Annular Chem-RRT Frontier Explorer (Idea #04).

Inspired by FutureHouse ether0 molecular design debates and active applicability-domain
expansion in self-driving laboratories. Grows a Rapidly-Exploring Random Tree (Chem-RRT)
of micro-volume (<= 10.0 uL), in-stock stepping-stone calibration probes strictly inside
annular frontier shells `tau_t < d(x, S_t) <= frontierOuterRadius(tau_t, eta)` toward high-ring
novel target scaffolds without ever taking an uncalibrated deep-OOD production leap or chaining
unassayed probes prior to wet-lab / `lab-attrib` certification.
Synchronized with `LabDecideLean.FrontierProbeSpec`, `checkFrontierAction`, and `checkCertifiedChain`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CandidateCompound:
    """Chemical compound node in standardized descriptor space."""

    compound_id: str
    scaffold_id: str
    coords: tuple[float, ...]
    ring_count: int
    is_direct_buy: bool
    is_novel_target: bool = False


@dataclass(frozen=True)
class TreeEdge:
    """Directed stepping-stone calibration edge in the Chem-RRT expansion tree."""

    parent_id: str
    child_id: str
    step_distance: float
    assay_vol_ul: float
    certified: bool = True


@dataclass(frozen=True)
class FrontierActionRecord:
    """Audited record of an exploitation or frontier-calibration experiment."""

    compound_id: str
    mode: str  # "exploitation" or "frontier_calibration"
    nn_dist: float
    tau_radius: float
    outer_shell_radius: float
    assay_vol_ul: float
    is_direct_buy: bool
    used_as_point_prediction: bool
    guard_passed: bool
    violation_rule: str | None
    certified: bool = True


def _to_exact_fraction(value: float | int | Fraction | Decimal) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Expected finite numeric value, got {value}")
        return Fraction(value)
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Expected finite numeric value, got {value}")
    return Fraction(str(fval))


def _frontier_outer_fraction(tau_f: Fraction, eta_f: Fraction) -> Fraction:
    """Exact outer radius matching `LabDecideLean.frontierOuterRadius` on permille grids."""
    tau_pm = tau_f * 1000
    eta_pm = eta_f * 1000
    if tau_pm.denominator == 1 and eta_pm.denominator == 1:
        outer_pm = int(tau_pm) + (int(tau_pm) * int(eta_pm)) // 1000
        return Fraction(outer_pm, 1000)
    return tau_f * (Fraction(1, 1) + eta_f)


def frontier_outer_radius(tau_radius: float, eta_step: float) -> float:
    """Compute annular shell outer radius aligned with `LabDecideLean.frontierOuterRadius`."""
    if tau_radius <= 0.0 or eta_step < 0.0:
        raise ValueError("tau_radius must be > 0 and eta_step >= 0")
    tau_f = _to_exact_fraction(tau_radius)
    eta_f = _to_exact_fraction(eta_step)
    return float(_frontier_outer_fraction(tau_f, eta_f))


def verify_frontier_action(
    *,
    mode: str,
    nn_dist: float,
    tau_radius: float,
    eta_step: float = 0.30,
    max_eta: float = 0.35,
    assay_vol_ul: float = 8.0,
    micro_vol_cap_ul: float = 10.0,
    is_direct_buy: bool = True,
    used_as_point_prediction: bool = False,
    parent_certified: bool = True,
) -> tuple[bool, str | None]:
    """Exact runtime mirror of `LabDecideLean.checkFrontierAction` and parent certification."""
    tau_f = _to_exact_fraction(tau_radius)
    dist_f = _to_exact_fraction(nn_dist)
    eta_f = _to_exact_fraction(eta_step)
    max_eta_f = _to_exact_fraction(max_eta)
    vol_f = _to_exact_fraction(assay_vol_ul)
    cap_f = _to_exact_fraction(micro_vol_cap_ul)

    if tau_f <= 0 or dist_f < 0:
        raise ValueError("tau_radius must be > 0 and nn_dist >= 0")

    if not parent_certified:
        return False, "parent_support_certified"

    outer_f = _frontier_outer_fraction(tau_f, eta_f)
    if mode == "exploitation":
        if dist_f <= tau_f:
            return True, None
        return False, "exploitation_inside_tau"

    if mode == "frontier_calibration":
        if eta_f > max_eta_f:
            return False, "bounded_annular_shell_width"
        if dist_f <= tau_f:
            return False, "strictly_outside_tau_shell"
        if dist_f > outer_f:
            return False, "within_stepping_stone_shell"
        if vol_f > cap_f:
            return False, "frontier_micro_volume_cap"
        if not is_direct_buy:
            return False, "frontier_requires_direct_stock"
        if used_as_point_prediction:
            return False, "frontier_calibration_only"
        return True, None

    raise ValueError(f"Unsupported campaign mode: {mode!r}")


def _nearest_support_node(
    query: CandidateCompound,
    support: Sequence[CandidateCompound],
    *,
    k: int = 1,
) -> tuple[CandidateCompound, float]:
    """Return `(nearest_support_node, kNN_mean_distance)` over certified `support`."""
    if not support:
        raise ValueError("support must be non-empty")
    q_arr = np.asarray(query.coords, dtype=np.float64)
    dists = [
        float(np.linalg.norm(q_arr - np.asarray(s_node.coords, dtype=np.float64)))
        for s_node in support
    ]
    best_idx = int(np.argmin(dists))
    k_eff = max(1, min(int(k), len(dists)))
    if k_eff == 1:
        return support[best_idx], dists[best_idx]
    mean_knn = float(np.mean(np.partition(np.asarray(dists, dtype=np.float64), k_eff - 1)[:k_eff]))
    return support[best_idx], mean_knn


def _evaluate_certification_verdict(verdict: Any) -> bool:
    """Validate boolean or `lab-attrib` ledger mapping certification result."""
    if isinstance(verdict, bool):
        return verdict
    if isinstance(verdict, Mapping):
        if verdict.get("admit_to_memory") is False:
            return False
        sha = verdict.get("ledger_entry_sha256")
        if sha is not None:
            return isinstance(sha, str) and bool(_SHA256_HEX_RE.match(sha))
        return verdict.get("admit_to_memory") is True
    return False


class ChemRRTExplorer:
    """Stepped annular frontier tree builder unlocking high-ring novel chemotypes safely."""

    def __init__(
        self,
        initial_support: Sequence[CandidateCompound],
        tau_radius: float = 0.80,
        eta_step: float = 0.30,
        max_eta: float = 0.35,
        micro_probe_vol_ul: float = 8.0,
        micro_vol_cap_ul: float = 10.0,
        production_vol_ul: float = 50.0,
        *,
        k_neighbors: int = 1,
    ) -> None:
        if not initial_support:
            raise ValueError("initial_support must be non-empty")
        if micro_probe_vol_ul > micro_vol_cap_ul:
            raise ValueError("micro_probe_vol_ul cannot exceed micro_vol_cap_ul")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.support: list[CandidateCompound] = list(initial_support)
        self.support_ids: set[str] = {c.compound_id for c in initial_support}
        self.pending_probes: dict[str, tuple[CandidateCompound, CandidateCompound, float]] = {}
        self.pending_probe_ids: set[str] = set()
        self.quarantined_probe_ids: set[str] = set()
        self.tau_radius = float(tau_radius)
        self.eta_step = float(eta_step)
        self.max_eta = float(max_eta)
        self.micro_probe_vol_ul = float(micro_probe_vol_ul)
        self.micro_vol_cap_ul = float(micro_vol_cap_ul)
        self.production_vol_ul = float(production_vol_ul)
        self.k_neighbors = int(k_neighbors)
        self.edges: list[TreeEdge] = []
        self.action_log: list[FrontierActionRecord] = []

    def propose_frontier_step(
        self,
        catalog_pool: Sequence[CandidateCompound],
        novel_targets: Sequence[CandidateCompound],
    ) -> CandidateCompound | None:
        """Select one annular stepping-stone probe anchored strictly to certified `self.support`.

        Places the selected probe into `self.pending_probes` (unassayed) WITHOUT promoting it
        to `self.support` until `certify_probe()` succeeds.
        """
        outer_shell = frontier_outer_radius(self.tau_radius, self.eta_step)
        tau_f = _to_exact_fraction(self.tau_radius)

        locked_targets = [
            t
            for t in novel_targets
            if _to_exact_fraction(_nearest_support_node(t, self.support, k=self.k_neighbors)[1]) > tau_f
        ]
        if not locked_targets:
            return None

        best_probe: CandidateCompound | None = None
        best_parent: CandidateCompound | None = None
        best_probe_dist = 0.0
        best_gain = 1e-9

        excluded_ids = self.support_ids | self.pending_probe_ids | self.quarantined_probe_ids
        for cand in catalog_pool:
            if cand.compound_id in excluded_ids or not cand.is_direct_buy:
                continue
            parent, dist_to_support = _nearest_support_node(cand, self.support, k=self.k_neighbors)
            ok, _ = verify_frontier_action(
                mode="frontier_calibration",
                nn_dist=dist_to_support,
                tau_radius=self.tau_radius,
                eta_step=self.eta_step,
                max_eta=self.max_eta,
                assay_vol_ul=self.micro_probe_vol_ul,
                micro_vol_cap_ul=self.micro_vol_cap_ul,
                is_direct_buy=cand.is_direct_buy,
                used_as_point_prediction=False,
                parent_certified=(parent.compound_id in self.support_ids),
            )
            if not ok:
                continue

            cand_arr = np.asarray(cand.coords, dtype=np.float64)
            max_progress = 0.0
            for target in locked_targets:
                _, cur_target_dist = _nearest_support_node(target, self.support, k=self.k_neighbors)
                hypothetical_support = [*self.support, cand]
                _, new_target_dist = _nearest_support_node(
                    target, hypothetical_support, k=self.k_neighbors
                )
                # Break kNN ties via direct Euclidean approach toward target
                euclid_progress = cur_target_dist - float(
                    np.linalg.norm(np.asarray(target.coords, dtype=np.float64) - cand_arr)
                )
                progress = max(cur_target_dist - new_target_dist, euclid_progress)
                max_progress = max(max_progress, progress)

            if max_progress > best_gain:
                best_gain = max_progress
                best_probe = cand
                best_parent = parent
                best_probe_dist = dist_to_support

        if best_probe is None or best_parent is None:
            return None

        ok, rule = verify_frontier_action(
            mode="frontier_calibration",
            nn_dist=best_probe_dist,
            tau_radius=self.tau_radius,
            eta_step=self.eta_step,
            max_eta=self.max_eta,
            assay_vol_ul=self.micro_probe_vol_ul,
            micro_vol_cap_ul=self.micro_vol_cap_ul,
            is_direct_buy=best_probe.is_direct_buy,
            used_as_point_prediction=False,
            parent_certified=(best_parent.compound_id in self.support_ids),
        )
        self.pending_probes[best_probe.compound_id] = (best_probe, best_parent, best_probe_dist)
        self.pending_probe_ids.add(best_probe.compound_id)
        self.action_log.append(
            FrontierActionRecord(
                compound_id=best_probe.compound_id,
                mode="frontier_calibration",
                nn_dist=best_probe_dist,
                tau_radius=self.tau_radius,
                outer_shell_radius=outer_shell,
                assay_vol_ul=self.micro_probe_vol_ul,
                is_direct_buy=best_probe.is_direct_buy,
                used_as_point_prediction=False,
                guard_passed=ok,
                violation_rule=rule,
                certified=False,
            )
        )
        return best_probe

    def certify_probe(
        self,
        compound_id: str,
        *,
        admit_to_memory: bool = True,
        ledger_entry_sha256: str | None = None,
    ) -> bool:
        """Promote a pending frontier probe into calibrated `self.support` after wet-lab certification."""
        pending = self.pending_probes.pop(compound_id, None)
        self.pending_probe_ids.discard(compound_id)
        if pending is None:
            return False
        probe, parent, step_dist = pending
        sha_ok = ledger_entry_sha256 is None or (
            isinstance(ledger_entry_sha256, str) and bool(_SHA256_HEX_RE.match(ledger_entry_sha256))
        )
        if not admit_to_memory or not sha_ok:
            self.quarantined_probe_ids.add(compound_id)
            return False

        self.support.append(probe)
        self.support_ids.add(compound_id)
        self.edges.append(
            TreeEdge(
                parent_id=parent.compound_id,
                child_id=probe.compound_id,
                step_distance=step_dist,
                assay_vol_ul=self.micro_probe_vol_ul,
                certified=True,
            )
        )
        # Mark matching pending action_log entry as certified
        for idx in range(len(self.action_log) - 1, -1, -1):
            rec = self.action_log[idx]
            if rec.compound_id == compound_id and rec.mode == "frontier_calibration" and not rec.certified:
                self.action_log[idx] = FrontierActionRecord(
                    compound_id=rec.compound_id,
                    mode=rec.mode,
                    nn_dist=rec.nn_dist,
                    tau_radius=rec.tau_radius,
                    outer_shell_radius=rec.outer_shell_radius,
                    assay_vol_ul=rec.assay_vol_ul,
                    is_direct_buy=rec.is_direct_buy,
                    used_as_point_prediction=rec.used_as_point_prediction,
                    guard_passed=rec.guard_passed,
                    violation_rule=rec.violation_rule,
                    certified=True,
                )
                break
        return True

    def exploit_certified_targets(
        self,
        novel_targets: Sequence[CandidateCompound],
    ) -> list[CandidateCompound]:
        """Run production exploitation strictly against wet-lab certified `self.support`."""
        outer_shell = frontier_outer_radius(self.tau_radius, self.eta_step)
        unlocked: list[CandidateCompound] = []
        for target in novel_targets:
            parent, dist = _nearest_support_node(target, self.support, k=self.k_neighbors)
            ok, rule = verify_frontier_action(
                mode="exploitation",
                nn_dist=dist,
                tau_radius=self.tau_radius,
                eta_step=self.eta_step,
                max_eta=self.max_eta,
                assay_vol_ul=self.production_vol_ul,
                micro_vol_cap_ul=self.micro_vol_cap_ul,
                is_direct_buy=target.is_direct_buy,
                used_as_point_prediction=True,
                parent_certified=(parent.compound_id in self.support_ids),
            )
            if ok:
                self.action_log.append(
                    FrontierActionRecord(
                        compound_id=target.compound_id,
                        mode="exploitation",
                        nn_dist=dist,
                        tau_radius=self.tau_radius,
                        outer_shell_radius=outer_shell,
                        assay_vol_ul=self.production_vol_ul,
                        is_direct_buy=target.is_direct_buy,
                        used_as_point_prediction=True,
                        guard_passed=True,
                        violation_rule=rule,
                        certified=True,
                    )
                )
                unlocked.append(target)
        return unlocked

    def expand_toward_targets(
        self,
        catalog_pool: Sequence[CandidateCompound],
        novel_targets: Sequence[CandidateCompound],
        max_probe_budget: int = 40,
        *,
        certify_probe_fn: Callable[[CandidateCompound], bool | Mapping[str, Any]] | None = None,
        require_certification: bool = False,
    ) -> list[CandidateCompound]:
        """Grow micro-scale in-stock RRT branches toward targets, promoting only certified probes."""
        probes_used = 0
        while probes_used < max_probe_budget:
            probe = self.propose_frontier_step(catalog_pool, novel_targets)
            if probe is None:
                break
            probes_used += 1

            if certify_probe_fn is not None:
                verdict = certify_probe_fn(probe)
                if isinstance(verdict, Mapping):
                    self.certify_probe(
                        probe.compound_id,
                        admit_to_memory=(verdict.get("admit_to_memory") is not False),
                        ledger_entry_sha256=verdict.get("ledger_entry_sha256"),
                    )
                else:
                    self.certify_probe(
                        probe.compound_id,
                        admit_to_memory=_evaluate_certification_verdict(verdict),
                    )
            elif not require_certification:
                # Simulated closed-loop wet-lab assay completion in standalone benchmark mode
                self.certify_probe(probe.compound_id, admit_to_memory=True)

        return self.exploit_certified_targets(novel_targets)


def run_benchmark(seed: int = 42) -> dict[str, object]:
    """Compare `static_wall`, `unbounded_leap`, and `chem_rrt_stepping` across 12 novel scaffolds."""
    rng = np.random.default_rng(seed)
    tau_radius = 0.80
    eta_step = 0.30

    # Initial calibrated low-ring training/calibration cluster around origin
    initial_support = [
        CandidateCompound(
            compound_id=f"S0_{i}",
            scaffold_id="monocyclic_core",
            coords=tuple(rng.normal(0.0, 0.22, size=3).tolist()),
            ring_count=1,
            is_direct_buy=True,
        )
        for i in range(12)
    ]
    initial_support.append(
        CandidateCompound("S0_origin", "monocyclic_core", (0.0, 0.0, 0.0), 1, True)
    )

    # 4 directional rays in 3D descriptor space, each leading to 3 novel high-ring targets
    # (12 total high-ring novel targets at radii ~2.1 to 2.9, well outside tau = 0.80)
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-0.7071, -0.7071, 0.0]),
    ]

    catalog_pool: list[CandidateCompound] = []
    novel_targets: list[CandidateCompound] = []

    for ray_idx, unit_dir in enumerate(directions):
        # Homologous series of in-stock building blocks spaced by ~0.20 (< shell width 0.24)
        for hop in range(1, 13):
            pos = unit_dir * (0.22 * hop + 0.45) + rng.normal(0.0, 0.01, size=3)
            catalog_pool.append(
                CandidateCompound(
                    compound_id=f"BB_ray{ray_idx}_hop{hop}",
                    scaffold_id=f"bridge_ring{min(4, 1 + hop // 4)}",
                    coords=tuple(pos.tolist()),
                    ring_count=min(4, 1 + hop // 4),
                    is_direct_buy=True,
                )
            )
        # Also add non-purchasable synthesis-only decoys in the shell (must NEVER be used as probes!)
        decoy_pos = unit_dir * 0.96 + np.array([0.02, -0.02, 0.01])
        catalog_pool.append(
            CandidateCompound(
                compound_id=f"SYNTH_ONLY_ray{ray_idx}",
                scaffold_id=f"unpurchasable_ray{ray_idx}",
                coords=tuple(decoy_pos.tolist()),
                ring_count=3,
                is_direct_buy=False,
            )
        )
        # 3 novel high-ring synthesis targets clustered around the end of each ray
        for t_sub in range(3):
            offset = rng.normal(0.0, 0.14, size=3)
            target_pos = unit_dir * (2.82 + 0.12 * t_sub) + offset
            novel_targets.append(
                CandidateCompound(
                    compound_id=f"TARGET_ray{ray_idx}_{t_sub}",
                    scaffold_id=f"novel_polycycle_{ray_idx}_{t_sub}",
                    coords=tuple(target_pos.tolist()),
                    ring_count=4 + (t_sub % 2),
                    is_direct_buy=False,
                    is_novel_target=True,
                )
            )

    # Policy 1: Static Wall (never probes outside S_0)
    static_unlocked = sum(
        1
        for t in novel_targets
        if _nearest_support_node(t, initial_support)[1] <= tau_radius
    )

    # Policy 2: Unbounded Leap (directly executes 50 uL production runs on all deep-OOD targets)
    unbounded_ood_leaps = 0
    unbounded_vol_wasted_ul = 0.0
    for target in novel_targets:
        _, dist = _nearest_support_node(target, initial_support)
        ok, _ = verify_frontier_action(
            mode="exploitation",
            nn_dist=dist,
            tau_radius=tau_radius,
            eta_step=eta_step,
            assay_vol_ul=50.0,
            is_direct_buy=target.is_direct_buy,
            used_as_point_prediction=True,
        )
        if not ok:
            unbounded_ood_leaps += 1
            unbounded_vol_wasted_ul += 50.0

    # Policy 3: Chem-RRT Stepping (micro-volume in-stock annular expansion)
    explorer = ChemRRTExplorer(
        initial_support=initial_support,
        tau_radius=tau_radius,
        eta_step=eta_step,
        micro_probe_vol_ul=8.0,
        micro_vol_cap_ul=10.0,
        production_vol_ul=50.0,
    )
    rrt_unlocked = explorer.expand_toward_targets(
        catalog_pool, novel_targets, max_probe_budget=24
    )
    rrt_ood_leaps = sum(1 for rec in explorer.action_log if not rec.guard_passed)
    probe_vol_ul = sum(
        rec.assay_vol_ul
        for rec in explorer.action_log
        if rec.mode == "frontier_calibration"
    )

    return {
        "idea_id": "idea_04_chem_rrt_frontier",
        "n_novel_target_scaffolds": len(novel_targets),
        "static_wall": {
            "novel_scaffolds_unlocked": static_unlocked,
            "uncalibrated_ood_leaps": 0,
            "calibration_probe_volume_ul": 0.0,
        },
        "unbounded_leap": {
            "novel_scaffolds_unlocked": 0,
            "uncalibrated_ood_leaps": unbounded_ood_leaps,
            "reagent_wasted_on_ood_ul": round(unbounded_vol_wasted_ul, 1),
        },
        "chem_rrt_stepping": {
            "novel_scaffolds_unlocked": len(rrt_unlocked),
            "uncalibrated_ood_leaps": rrt_ood_leaps,
            "stepping_stone_probes_used": len(explorer.edges),
            "calibration_probe_volume_ul": round(probe_vol_ul, 1),
        },
    }
