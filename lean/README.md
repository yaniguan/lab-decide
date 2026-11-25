# `LabDecideLean` -- Formal L6 Conformal & Stepped Frontier Guardrails in Lean 4

Self-contained Lean 4 (`leanprover/lean4:v4.36.0`, zero external dependencies, `0 sorry`, `0 axiom`) formalization of the `L6` conformal prediction and active frontier expansion invariants for `lab-decide`, synchronized with the central 15-repository formal specification in [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean).

## Proved Theorems & Guardrails

| Component | Definition / Theorem | Guarantee |
| :--- | :--- | :--- |
| Conformal Gate | `checkDecide_ok_iff` | Exact soundness and completeness of `checkDecide` wrt `ConformalDecisionOK` |
| Applicability Domain | `out_of_domain_always_abstains` | Candidates with `knnDistPermille > tauDistPermille` never pass `checkDecide` |
| Finite-Sample Rank | `undersized_calib_always_abstains`, `conformalRank_pos`, `conformalRank_exact_lower`, `conformalRank_exact_upper` | Exact integer ceiling characterization without floating-point rank drift or undersized clamping |
| Physical Range | `physical_bounds_respected` | Accepted intervals obey `physMinMilli <= loMilli <= hiMilli <= physMaxMilli` |
| Strict Annular Shell | `interior_rejected_as_frontier_probe`, `frontier_calibration_strictly_annular`, `deep_ood_always_blocked` | Frontier stepping stones satisfy `tau < knnDistPermille <= frontierOuterRadius tau eta` |
| Stepping-Stone Safety | `accepted_outside_tau_is_bounded_probe` | Out-of-`tau` acceptances are micro-volume (`<= 10 uL`), direct-buy calibration probes |
| Certified Chaining | `unassayed_probe_blocks_chaining`, `unassayed_hop_blocks_chain`, `certified_hop_annular_bounds`, `certified_chain_annular_span`, `exploitation_after_chain_sound` | Unassayed probes can never be chained; certified `k`-hop branches obey telescoping annular bounds |

## Quickstart

```bash
cd lean
lake build
```
