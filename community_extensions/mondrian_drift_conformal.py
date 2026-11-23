"""Scaffold-Mondrian Online Adaptive Conformal Calibrator (ACI) (Idea #03).

Inspired by Gibbs & Candes (2021) Adaptive Conformal Inference (ACI) and
autonomous self-driving lab reagent-lot / temperature drift threads.
Maintains per-scaffold Mondrian residual windows and adapts each scaffold's
miscoverage parameter online via:
    alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - err_t), 0.01, 0.50)
while strictly honoring `LabDecideLean` finite-sample order-statistic ranks
without clamping when calibration support is undersized.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACIIntervalPrediction:
    """Single conformal interval decision emitted by Static or Mondrian-ACI calibrator."""

    scaffold_id: str
    point_pred: float
    lower: float
    upper: float
    alpha_used: float
    abstained: bool
    n_calib_bucket: int

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)


def conformal_order_rank(n_calib: int, alpha: float) -> int:
    """Exact finite-sample order-statistic rank ceil((n + 1) * (1 - alpha))."""
    if n_calib < 0:
        raise ValueError("n_calib must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly in (0, 1)")
    return math.ceil((n_calib + 1) * (1.0 - alpha) - 1e-12)


def finite_sample_conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """Exact finite-sample conformal quantile; returns +inf instead of clamping k > n."""
    n_cal = len(residuals)
    if n_cal == 0:
        return float("inf")
    rank = conformal_order_rank(n_cal, alpha)
    if rank > n_cal:
        return float("inf")
    sorted_res = np.sort(np.asarray(residuals, dtype=np.float64))
    return float(sorted_res[max(0, rank - 1)])


class StaticSplitConformal:
    """Baseline pooled split-conformal predictor frozen after pre-stream calibration."""

    def __init__(
        self,
        residuals: Sequence[float],
        alpha_target: float = 0.10,
        phys_bounds: tuple[float, float] = (-15.0, 1.75),
    ) -> None:
        if not (0.0 < alpha_target < 1.0):
            raise ValueError("alpha_target must lie in (0, 1)")
        self.alpha_target = float(alpha_target)
        self.phys_min, self.phys_max = float(phys_bounds[0]), float(phys_bounds[1])
        self._n_calib = len(residuals)
        self.q_hat = finite_sample_conformal_quantile(residuals, self.alpha_target)

    def predict(
        self, point_pred: float, scaffold_id: str = "global"
    ) -> ACIIntervalPrediction:
        if not math.isfinite(self.q_hat):
            return ACIIntervalPrediction(
                scaffold_id=scaffold_id,
                point_pred=point_pred,
                lower=self.phys_min,
                upper=self.phys_max,
                alpha_used=self.alpha_target,
                abstained=True,
                n_calib_bucket=self._n_calib,
            )
        lo = max(self.phys_min, min(self.phys_max, point_pred - self.q_hat))
        hi = max(lo, min(self.phys_max, point_pred + self.q_hat))
        return ACIIntervalPrediction(
            scaffold_id=scaffold_id,
            point_pred=point_pred,
            lower=lo,
            upper=hi,
            alpha_used=self.alpha_target,
            abstained=False,
            n_calib_bucket=self._n_calib,
        )


class ScaffoldMondrianACI:
    """Chemotype/scaffold-stratified online Adaptive Conformal Inference calibrator."""

    def __init__(
        self,
        initial_bucket_residuals: Mapping[str, Sequence[float]],
        alpha_target: float = 0.10,
        gamma: float = 0.04,
        alpha_bounds: tuple[float, float] = (0.01, 0.50),
        rolling_window: int = 100,
        phys_bounds: tuple[float, float] = (-15.0, 1.75),
    ) -> None:
        if not (0.0 < alpha_target < 1.0):
            raise ValueError("alpha_target must lie strictly in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not (0.0 < alpha_bounds[0] < alpha_bounds[1] < 1.0):
            raise ValueError("alpha_bounds must satisfy 0 < min < max < 1")
        if rolling_window < 1:
            raise ValueError("rolling_window must be >= 1")

        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.alpha_min, self.alpha_max = float(alpha_bounds[0]), float(alpha_bounds[1])
        self.rolling_window = int(rolling_window)
        self.phys_min, self.phys_max = float(phys_bounds[0]), float(phys_bounds[1])

        self._buckets: dict[str, deque[float]] = {}
        self._alpha_by_bucket: dict[str, float] = {}
        for scaf, res_seq in initial_bucket_residuals.items():
            cleaned = [abs(float(r)) for r in res_seq]
            self._buckets[scaf] = deque(
                cleaned[-self.rolling_window :], maxlen=self.rolling_window
            )
            self._alpha_by_bucket[scaf] = self.alpha_target

    def alpha_for_scaffold(self, scaffold_id: str) -> float:
        return self._alpha_by_bucket.get(scaffold_id, self.alpha_target)

    def predict(self, point_pred: float, scaffold_id: str) -> ACIIntervalPrediction:
        bucket = self._buckets.get(scaffold_id)
        alpha_cur = self._alpha_by_bucket.get(scaffold_id, self.alpha_target)
        n_cal = len(bucket) if bucket is not None else 0
        q_hat = (
            finite_sample_conformal_quantile(tuple(bucket), alpha_cur)
            if bucket is not None
            else float("inf")
        )
        if not math.isfinite(q_hat):
            return ACIIntervalPrediction(
                scaffold_id=scaffold_id,
                point_pred=point_pred,
                lower=self.phys_min,
                upper=self.phys_max,
                alpha_used=alpha_cur,
                abstained=True,
                n_calib_bucket=n_cal,
            )
        lo = max(self.phys_min, min(self.phys_max, point_pred - q_hat))
        hi = max(lo, min(self.phys_max, point_pred + q_hat))
        return ACIIntervalPrediction(
            scaffold_id=scaffold_id,
            point_pred=point_pred,
            lower=lo,
            upper=hi,
            alpha_used=alpha_cur,
            abstained=False,
            n_calib_bucket=n_cal,
        )

    def update(self, prediction: ACIIntervalPrediction, y_true: float) -> float:
        """Update scaffold alpha_t via Gibbs-Candes step and append latest residual."""
        scaf = prediction.scaffold_id
        if scaf not in self._buckets:
            self._buckets[scaf] = deque(maxlen=self.rolling_window)
            self._alpha_by_bucket[scaf] = self.alpha_target

        err_t = 0.0 if (prediction.lower <= y_true <= prediction.upper) else 1.0
        alpha_cur = self._alpha_by_bucket[scaf]
        alpha_next = alpha_cur + self.gamma * (self.alpha_target - err_t)
        alpha_clipped = max(self.alpha_min, min(self.alpha_max, alpha_next))
        self._alpha_by_bucket[scaf] = alpha_clipped
        self._buckets[scaf].append(abs(float(y_true - prediction.point_pred)))
        return alpha_clipped


def _winkler_interval_score(
    lower: float, upper: float, y_true: float, alpha_target: float
) -> float:
    """Proper scoring rule penalizing both interval width and miscoverage."""
    width = upper - lower
    if y_true < lower:
        return width + (2.0 / alpha_target) * (lower - y_true)
    if y_true > upper:
        return width + (2.0 / alpha_target) * (y_true - upper)
    return width


def run_benchmark(
    n_steps: int = 300,
    alpha_target: float = 0.10,
    gamma: float = 0.04,
    seed: int = 42,
) -> dict[str, object]:
    """Compare `static_split_conformal` vs `mondrian_aci` over a 300-step drifting stream."""
    if n_steps < 30:
        raise ValueError("n_steps must be >= 30")
    rng = np.random.default_rng(seed)

    scaffolds = ("aryl_amide", "spiro_heterocycle", "macrocycle_peptoid")
    base_sigma = {
        "aryl_amide": 0.28,
        "spiro_heterocycle": 0.42,
        "macrocycle_peptoid": 0.65,
    }

    # Pre-stream clean calibration pool (100 compounds per scaffold -> n >= 100 per bucket)
    init_bucket_res: dict[str, list[float]] = {}
    pooled_res: list[float] = []
    for scaf in scaffolds:
        draws = np.abs(rng.normal(0.0, base_sigma[scaf], size=100)).tolist()
        init_bucket_res[scaf] = draws
        pooled_res.extend(draws)

    static_cal = StaticSplitConformal(pooled_res, alpha_target=alpha_target)
    mondrian_aci = ScaffoldMondrianACI(
        init_bucket_res,
        alpha_target=alpha_target,
        gamma=gamma,
        alpha_bounds=(0.01, 0.50),
        rolling_window=100,
    )

    static_hits: list[float] = []
    static_widths: list[float] = []
    static_winkler: list[float] = []
    static_by_scaf: dict[str, list[float]] = {s: [] for s in scaffolds}

    aci_hits: list[float] = []
    aci_widths: list[float] = []
    aci_winkler: list[float] = []
    aci_by_scaf: dict[str, list[float]] = {s: [] for s in scaffolds}

    for step in range(n_steps):
        scaf = scaffolds[step % len(scaffolds)]
        point_pred = float(rng.uniform(-6.5, -1.2))

        # Temporal assay & reagent-lot drift schedule:
        # Step 0..79: mild warm-up drift; Step 80..179: DMSO water-uptake lot shift;
        # Step 180..299: turbidimetric detector drift + amplified macrocycle variance.
        drift_bias = 0.08 * math.sin(step / 28.0)
        sigma_mult = 1.0 + 0.25 * (step / n_steps)
        if step >= 80:
            drift_bias += 0.32 if scaf != "aryl_amide" else 0.14
            sigma_mult += 0.22
        if step >= 180:
            drift_bias += 0.28 if scaf == "macrocycle_peptoid" else 0.16
            sigma_mult += 0.25 if scaf == "macrocycle_peptoid" else 0.12

        noise = float(rng.normal(drift_bias, base_sigma[scaf] * sigma_mult))
        y_true = max(-14.5, min(1.5, point_pred + noise))

        p_static = static_cal.predict(point_pred, scaf)
        hit_s = 1.0 if (p_static.lower <= y_true <= p_static.upper) else 0.0
        static_hits.append(hit_s)
        static_widths.append(p_static.width)
        static_winkler.append(
            _winkler_interval_score(
                p_static.lower, p_static.upper, y_true, alpha_target
            )
        )
        static_by_scaf[scaf].append(hit_s)

        p_aci = mondrian_aci.predict(point_pred, scaf)
        hit_a = 1.0 if (p_aci.lower <= y_true <= p_aci.upper) else 0.0
        aci_hits.append(hit_a)
        aci_widths.append(p_aci.width)
        aci_winkler.append(
            _winkler_interval_score(p_aci.lower, p_aci.upper, y_true, alpha_target)
        )
        aci_by_scaf[scaf].append(hit_a)
        mondrian_aci.update(p_aci, y_true)

    target_cov = 1.0 - alpha_target
    static_cov = float(np.mean(static_hits))
    aci_cov = float(np.mean(aci_hits))

    return {
        "idea_id": "idea_03_mondrian_drift_conformal",
        "n_steps": n_steps,
        "target_coverage": round(target_cov, 4),
        "static_split_conformal": {
            "empirical_coverage": round(static_cov, 4),
            "abs_coverage_error": round(abs(static_cov - target_cov), 4),
            "worst_scaffold_coverage": round(
                min(float(np.mean(v)) for v in static_by_scaf.values()), 4
            ),
            "mean_interval_width": round(float(np.mean(static_widths)), 4),
            "winkler_efficiency_score": round(float(np.mean(static_winkler)), 4),
        },
        "mondrian_aci": {
            "empirical_coverage": round(aci_cov, 4),
            "abs_coverage_error": round(abs(aci_cov - target_cov), 4),
            "worst_scaffold_coverage": round(
                min(float(np.mean(v)) for v in aci_by_scaf.values()), 4
            ),
            "mean_interval_width": round(float(np.mean(aci_widths)), 4),
            "winkler_efficiency_score": round(float(np.mean(aci_winkler)), 4),
        },
    }
