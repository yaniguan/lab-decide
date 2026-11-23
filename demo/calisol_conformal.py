"""CALiSol-23 DOI-split conformal calibration vs formulation kernel support gate."""
from __future__ import annotations

import os
import sys
from typing import Sequence

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_decide.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    return cli_main(["--dataset", "calisol23", *args])


if __name__ == "__main__":
    raise SystemExit(main())
