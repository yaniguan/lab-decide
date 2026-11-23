"""Stepped Annular Chem-RRT Frontier Explorer (Idea #04).

Inspired by FutureHouse ether0 molecular design debates and active applicability-domain
expansion in self-driving laboratories. Grows a Rapidly-Exploring Random Tree (Chem-RRT)
of micro-volume (<= 10.0 uL), in-stock stepping-stone calibration probes strictly inside
annular frontier shells `tau_t < d(x, S_t) <= (1 + eta) * tau_t` toward high-ring novel
target scaffolds without ever taking an uncalibrated deep-OOD production leap.
Synchronized with `LabDecideLean.FrontierProbeSpec` and `checkFrontierAction`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
) -> tuple[bool, str | None]:
    """Exact runtime mirror of `LabDecideLean.checkFrontierAction`."""
    if tau_radius <= 0.0 or nn_dist < 0.0:
        raise ValueError("tau_radius must be > 0 and nn_dist >= 0")

    outer_radius = tau_radius * (1.0 + eta_step)
    if mode == "exploitation":
        if nn_dist <= tau_radius + 1e-12:
            return True, None
        return False, "exploitation_inside_tau"

    if mode == "frontier_calibration":
        if eta_step > max_eta + 1e-12:
            return False, "bounded_annular_shell_width"
        if nn_dist > outer_radius + 1e-12:
            return False, "within_stepping_stone_shell"
        if assay_vol_ul > micro_vol_cap_ul + 1e-12:
            return False, "frontier_micro_volume_cap"
        if not is_direct_buy:
            return False, "frontier_requires_direct_stock"
        if used_as_point_prediction:
            return False, "frontier_calibration_only"
        return True, None

    raise ValueError(f"Unsupported campaign mode: {mode!r}")


def _nearest_support_node(
    query: CandidateCompound, support: Sequence[CandidateCompound]
) -> tuple[CandidateCompound, float]:
    q_arr = np.asarray(query.coords, dtype=np.float64)
    best_node = support[0]
    best_dist = float("inf")
    for s_node in support:
        dist = float(
            np.linalg.norm(q_arr - np.asarray(s_node.coords, dtype=np.float64))
        )
        if dist < best_dist:
            best_dist = dist
            best_node = s_node
    return best_node, best_dist


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
    ) -> None:
        if not initial_support:
            raise ValueError("initial_support must be non-empty")
        if micro_probe_vol_ul > micro_vol_cap_ul:
            raise ValueError("micro_probe_vol_ul cannot exceed micro_vol_cap_ul")
        self.support: list[CandidateCompound] = list(initial_support)
        self.support_ids: set[str] = {c.compound_id for c in initial_support}
        self.tau_radius = float(tau_radius)
        self.eta_step = float(eta_step)
        self.max_eta = float(max_eta)
        self.micro_probe_vol_ul = float(micro_probe_vol_ul)
        self.micro_vol_cap_ul = float(micro_vol_cap_ul)
        self.production_vol_ul = float(production_vol_ul)
        self.edges: list[TreeEdge] = []
        self.action_log: list[FrontierActionRecord] = []

    def expand_toward_targets(
        self,
        catalog_pool: Sequence[CandidateCompound],
        novel_targets: Sequence[CandidateCompound],
        max_probe_budget: int = 40,
    ) -> list[CandidateCompound]:
        """Grow micro-scale in-stock RRT branches toward targets, then exploit certified ones."""
        probes_used = 0
        outer_shell = self.tau_radius * (1.0 + self.eta_step)

        while probes_used < max_probe_budget:
            locked_targets = [
                t
                for t in novel_targets
                if _nearest_support_node(t, self.support)[1] > self.tau_radius + 1e-12
            ]
            if not locked_targets:
                break

            # Select in-stock stepping stone inside annular shell (tau, (1 + eta) * tau]
            # that maximizes distance reduction toward the closest locked high-ring target.
            best_probe: CandidateCompound | None = None
            best_parent: CandidateCompound | None = None
            best_probe_dist = 0.0
            best_gain = 1e-9

            for cand in catalog_pool:
                if cand.compound_id in self.support_ids or not cand.is_direct_buy:
                    continue
                parent, dist_to_support = _nearest_support_node(cand, self.support)
                if not (self.tau_radius < dist_to_support <= outer_shell + 1e-12):
                    continue

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
                )
                if not ok:
                    continue

                cand_arr = np.asarray(cand.coords, dtype=np.float64)
                max_progress = 0.0
                for target in locked_targets:
                    _, cur_target_dist = _nearest_support_node(target, self.support)
                    new_target_dist = float(
                        np.linalg.norm(
                            np.asarray(target.coords, dtype=np.float64) - cand_arr
                        )
                    )
                    progress = cur_target_dist - new_target_dist
                    max_progress = max(max_progress, progress)

                if max_progress > best_gain:
                    best_gain = max_progress
                    best_probe = cand
                    best_parent = parent
                    best_probe_dist = dist_to_support

            if best_probe is None or best_parent is None:
                break

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
            )
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
                )
            )
            self.support.append(best_probe)
            self.support_ids.add(best_probe.compound_id)
            self.edges.append(
                TreeEdge(
                    parent_id=best_parent.compound_id,
                    child_id=best_probe.compound_id,
                    step_distance=best_probe_dist,
                    assay_vol_ul=self.micro_probe_vol_ul,
                )
            )
            probes_used += 1

        unlocked: list[CandidateCompound] = []
        for target in novel_targets:
            _, dist = _nearest_support_node(target, self.support)
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
                    )
                )
                unlocked.append(target)

        return unlocked


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
        if _nearest_support_node(t, initial_support)[1] <= tau_radius + 1e-12
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
