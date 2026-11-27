"""Verify that all theorems in lean/LabDecideLean.lean are audited by AuditAxioms.lean
and depend strictly on standard Lean 4 axioms ({propext, Quot.sound, Classical.choice})
with zero sorryAx or custom axioms.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_SRC_PATH = REPO_ROOT / "lean" / "LabDecideLean.lean"
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def verify_lean_axiom_output(ax_out: str, lean_src: str | None = None) -> list[str]:
    """Validate `#print axioms` output against all theorems in `LabDecideLean.lean`."""
    if lean_src is None:
        lean_src = LEAN_SRC_PATH.read_text(encoding="utf-8")
    thms = re.findall(
        r"^(?:@\[[^\]]+\]\s*)*theorem\s+([A-Za-z_][A-Za-z0-9_']*)",
        lean_src,
        flags=re.MULTILINE,
    )
    if len(thms) < 20:
        raise AssertionError(f"Expected >= 20 theorems in LabDecideLean.lean, found {len(thms)}")
    if "sorryAx" in ax_out:
        raise AssertionError(f"sorryAx detected in Lean axiom output:\n{ax_out}")

    for thm in thms:
        fqn = f"'LabDecideLean.{thm}'"
        if fqn not in ax_out:
            raise AssertionError(f"Theorem {fqn} missing from AuditAxioms.lean output")

    non_empty_lines = [ln.strip() for ln in ax_out.splitlines() if ln.strip()]
    if len(non_empty_lines) < len(thms):
        raise AssertionError(
            f"Expected >= {len(thms)} axiom report lines, got {len(non_empty_lines)}"
        )

    for ln in non_empty_lines:
        if ln.endswith("does not depend on any axioms"):
            continue
        m = re.search(r"depends on axioms:\s*\[([^\]]*)\]$", ln)
        if m is None:
            raise AssertionError(f"Unexpected line in Lean axiom output: {ln!r}")
        used = {tok.strip() for tok in m.group(1).split(",") if tok.strip()}
        extra = used - ALLOWED_AXIOMS
        if extra:
            raise AssertionError(f"Custom/disallowed axioms found in {ln!r}: {sorted(extra)}")

    return thms


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python scripts/check_lean_axioms.py <lean_axioms_output.txt>", file=sys.stderr)
        return 2
    ax_path = Path(args[0])
    ax_out = ax_path.read_text(encoding="utf-8")
    thms = verify_lean_axiom_output(ax_out)
    print(f"Verified {len(thms)} Lean theorems with zero sorryAx and zero custom axioms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
