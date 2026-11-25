"""On-demand fetcher, live URL verifier, and conformal calibration runner for open datasets.

Downloads registered `OpenCalibrationDataset` benchmarks into gitignored `data/` with atomic
partial-file replacement and SHA-256 verification.

Examples:
  python scripts/fetch_open_datasets.py --list
  python scripts/fetch_open_datasets.py --check-urls
  python scripts/fetch_open_datasets.py --dataset freesolv --calibrate
  python scripts/fetch_open_datasets.py --dataset aqsoldb,direct_arylation_shields --calibrate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lab_decide.datasets import (  # noqa: E402
    DEFAULT_OPEN_DATA_DIR,
    OPEN_CALIBRATION_DATASETS,
    VALID_DOMAINS,
    OpenCalibrationDataset,
    calibrate_open_dataset,
    fetch_open_dataset,
    get_open_dataset,
    list_open_datasets,
)


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    return obj


def verify_dataset_url_live(spec: OpenCalibrationDataset, *, timeout: float = 25.0) -> dict[str, Any]:
    """Verify `spec.url` responds with HTTP 2xx and a non-empty payload stream."""
    req = urllib.request.Request(
        spec.url,
        headers={
            "User-Agent": "lab-decide-url-check/0.2.0",
            "Range": "bytes=0-1023",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        sample = resp.read(1024)
    ok = (200 <= status < 300) and len(sample) > 0
    return {
        "dataset_id": spec.dataset_id,
        "url": spec.url,
        "http_status": status,
        "sample_bytes": len(sample),
        "ok": ok,
    }


def _resolve_requested_datasets(
    dataset_arg: Sequence[str] | None,
    *,
    fetch_all: bool,
    domain: str | None,
) -> list[OpenCalibrationDataset]:
    if fetch_all or not dataset_arg:
        return list_open_datasets(domain=domain)
    ids: list[str] = []
    for raw in dataset_arg:
        for token in raw.split(","):
            clean = token.strip()
            if clean:
                ids.append(clean)
    specs = [get_open_dataset(ds_id) for ds_id in ids]
    if domain is not None:
        specs = [s for s in specs if s.domain == domain]
    return specs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_open_datasets.py",
        description="Fetch, verify, and calibrate open molecular, ADMET, and HTE BO datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered OpenCalibrationDataset entries and exit.",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Verify live HTTP 200/206 reachability for selected dataset URLs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset ID(s) to fetch/check (comma-separated or repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all registered datasets (optionally filtered by --domain).",
    )
    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        default=None,
        help="Filter datasets by scientific domain.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_OPEN_DATA_DIR,
        help="Destination directory for cached downloads (defaults to gitignored data/).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run split Mondrian conformal calibration + trust-region BO after fetching.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap when calibrating.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Target miscoverage rate alpha in (0, 1).")
    parser.add_argument("--trust-pct", type=float, default=80.0, help="Calibration kNN distance percentile.")
    parser.add_argument(
        "--nonconformity",
        choices=("normalized", "cqr", "cqr_r", "absolute"),
        default="normalized",
        help="Conformal nonconformity score mode.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.list:
            specs = list_open_datasets(domain=args.domain)
            if args.json:
                print(json.dumps([s.to_dict() for s in specs], indent=2))
            else:
                header = (
                    f"{'dataset_id':<28}{'domain':<24}{'unit':<18}"
                    f"{'records':>9}  {'license':<13}{'comm_ok':<8}"
                )
                print(header)
                print("-" * len(header))
                for s in specs:
                    print(
                        f"{s.dataset_id:<28}{s.domain:<24}{s.target_unit:<18}"
                        f"{s.n_records:>9}  {s.spdx_license:<13}{str(s.commercial_use_ok):<8}"
                    )
            return 0

        if args.check_urls:
            specs = _resolve_requested_datasets(args.dataset, fetch_all=args.all, domain=args.domain)
            results = []
            all_ok = True
            for s in specs:
                res = verify_dataset_url_live(s, timeout=args.timeout)
                results.append(res)
                all_ok = all_ok and bool(res["ok"])
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for r in results:
                    status_lbl = "OK" if r["ok"] else "FAIL"
                    print(
                        f"[{status_lbl}] {r['dataset_id']:<28} HTTP {r['http_status']} "
                        f"({r['sample_bytes']} B) -> {r['url']}"
                    )
            return 0 if all_ok else 1

        if not args.dataset and not args.all:
            parser.print_help()
            return 2

        specs = _resolve_requested_datasets(args.dataset, fetch_all=args.all, domain=args.domain)
        outputs: list[dict[str, Any]] = []
        for s in specs:
            local_path = fetch_open_dataset(
                s.dataset_id,
                data_dir=args.data_dir,
                force=args.force,
                timeout=args.timeout,
            )
            entry: dict[str, Any] = {
                "dataset_id": s.dataset_id,
                "local_path": os.path.abspath(local_path),
                "bytes": os.path.getsize(local_path),
                "spdx_license": s.spdx_license,
            }
            if args.calibrate:
                _, _, _, metrics = calibrate_open_dataset(
                    s.dataset_id,
                    path=local_path,
                    alpha=args.alpha,
                    trust_pct=args.trust_pct,
                    nonconformity=args.nonconformity,
                    max_rows=args.max_rows,
                )
                entry["calibration"] = _json_clean(metrics)
            outputs.append(entry)

        if args.json:
            print(json.dumps(outputs, indent=2))
        else:
            for item in outputs:
                print(
                    f"Fetched {item['dataset_id']} ({item['bytes']} bytes, {item['spdx_license']}) "
                    f"-> {item['local_path']}"
                )
                cal = item.get("calibration")
                if cal:
                    print(
                        f"  Calibrated n={cal['n_clean']} | answered={cal['answered_fraction']:.3f} "
                        f"cov_in={cal['cov_in']:.3f} MAE_in={cal['mae_in']:.3f} "
                        f"width_in={cal['width_in']:.3f} ({cal['target_unit']})"
                    )
        return 0

    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
