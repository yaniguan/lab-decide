# `LabDecideLean` -- Formal L6 Conformal & Stepped Frontier Guardrails in Lean 4

Self-contained Lean 4 (`leanprover/lean4:v4.36.0`, zero external dependencies, `0 sorry`, `0 axiom`) formalization of the `L6` conformal prediction and active frontier expansion invariants for `lab-decide`, synchronized with the central 14-repository formal specification in [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean).

## Proved Theorems & Guardrails

| Component | Definition / Theorem | Guarantee |
| :--- | :--- | :--- |
| Conformal Gate | `checkDecide_ok_iff` | Exact soundness and completeness of `checkDecide` wrt `ConformalDecisionOK` |
| Applicability Domain | `out_of_domain_always_abstains` | Candidates with `knnDistPermille > tauDistPermille` never pass `checkDecide` |
| Finite-Sample Rank | `undersized_calib_always_abstains`, `conformalRank_pos` | Undersized calibration sets (`ceil((n+1)*(1-alpha)) > n`) abstain without clamping |
| Physical Range | `physical_bounds_respected` | Accepted intervals obey `physMinMilli <= loMilli <= hiMilli <= physMaxMilli` |
| Deep-OOD Block | `deep_ood_always_blocked` | Candidates beyond `frontierOuterRadius tau eta` are blocked in all modes |
| Stepping-Stone Safety | `accepted_outside_tau_is_bounded_probe` | Out-of-`tau` acceptances are micro-volume (`<= 10 uL`), direct-buy calibration probes |

## Quickstart

```bash
cd lean
lake build
```
