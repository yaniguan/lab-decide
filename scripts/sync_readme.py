"""Synchronize README.md benchmark tables and demo output from committed JSON/txt artifacts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_mod(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_readme_text() -> str:
    reg_mod = _load_mod("_reg_mod", "scripts/benchmark_registry.py")
    camp_mod = _load_mod("_camp_mod", "scripts/benchmark_campaign.py")
    reg_payload = json.loads((REPO_ROOT / "results" / "registry_benchmark.json").read_text(encoding="utf-8"))
    camp_payload = json.loads((REPO_ROOT / "results" / "campaign_benchmark.json").read_text(encoding="utf-8"))

    reg_md = reg_mod.render_registry_markdown(reg_payload)
    camp_md = camp_mod.render_campaign_markdown(camp_payload)
    demo_txt = (REPO_ROOT / "demo" / "output.txt").read_text(encoding="utf-8").rstrip()

    h2h_r = reg_payload["esol_delaney_20seed_head_to_head"]["random"]
    h2h_s = reg_payload["esol_delaney_20seed_head_to_head"]["scaffold"]
    fr_m = h2h_r["models"]["frontier_rdkit_2d_morgan"]["metrics"]
    fs_m = h2h_s["models"]["frontier_rdkit_2d_morgan"]["metrics"]
    del_r = h2h_r["models"]["delaney_2004_csv_formula"]["metrics"]
    del_s = h2h_s["models"]["delaney_2004_csv_formula"]["metrics"]
    ols_s = h2h_s["models"]["delaney_2004_refit_4desc_ols"]["metrics"]
    esol20_r = reg_payload["esol_20seed"]["random"]
    esol20_s = reg_payload["esol_20seed"]["scaffold"]
    n_front_feats = int(h2h_r["models"]["frontier_rdkit_2d_morgan"]["n_features"])

    fmt = reg_mod._fmt_pm

    return f"""# lab-decide

An experiment advisor that *knows where it can be trusted*: calibrated answers inside its region, a clear "I don't know" outside, and a region that can grow as the lab adds data.

> **Auto-Lab Ecosystem**: Layer `L6 - Metacognitive & Epistemic Brain` in [auto-lab](https://github.com/yaniguan/auto-lab) | [Project Studio](https://yaniguan.github.io/auto-lab/)

## Where it fits

In the [Auto-Lab](https://github.com/yaniguan/auto-lab) agent ([yaniguan.github.io/auto-lab](https://yaniguan.github.io/auto-lab/) | [yaniguan.github.io](https://yaniguan.github.io/)), the brain proposes the next experiment. This repo is the decision layer under it: sequential active optimization (including trust-region conformal UCB/EI, Gaussian Process EI, LightGBM + CQR, and semiparametric RDKit 2D + Morgan stacked ensembles) to choose, and split/Mondrian conformal prediction to say how sure it is.

## System Profile: Object | Data | Model | Training | History | Unique Properties

Full architectural documentation, conformal calibration math, dataset diagnostics, commit lineage, and sibling-repository contracts are maintained in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

### Object
Epistemic experiment advisor and uncertainty-aware decision layer (`L6`) that delivers distribution-free calibrated prediction intervals inside its empirical applicability domain, emits explicit `"I don't know"` abstentions outside it, and recalibrates as active laboratory measurements expand the boundary.

### Data
- **Active Benchmark (`ESOL`, Delaney 2004)**: `1,128` raw (`1,123` deduplicated) compounds verified against `ESOL_SHA256` (`561` train / `224` pool / `169` calibration / `169` test), predicting `measured log solubility in mols per litre` ($\\log_{{10}} S \\in [-11.60, 1.58]$) under both the no-RDKit `6`-descriptor tabular mode (`encoding="descriptors"`) and the `{n_front_feats}`-dimensional `rdkit_2d_morgan` molecular representation (`RDKit` 2D physicochemical descriptors + intensive ratios + `167` MACCS keys + `1,024`-bit count-Morgan fingerprints), while strictly excluding the CSV's pre-fitted `ESOL predicted` column during training.
- **Open Calibration Dataset Registry (`src/lab_decide/datasets.py`)**: Typed `OpenCalibrationDataset` registry with explicit `objective_direction` (`maximize`, `minimize`, `calibration_only`), explicit domain `group_keys` (`murcko_scaffold`, `solute`, `doi`, `aryl_halide`, `substrate`, `ligand` with zero regex fallback groups), strict `data_source` provenance (`real_csv` vs opt-in `--allow-synthetic` `synthetic_fixture`), `rdkit_2d_morgan` molecular loaders across all `5` single-molecule benchmarks (`ESOL`, `AqSolDB`, `FreeSolv`, `Lipophilicity`, `Biogen ADME`), and multi-encoding HTE loaders (`descriptors`, `onehot`, `onehot_morgan`) across **`11` live-verified open-source molecular solubility, electrolyte solvation, ADMET, and high-throughput reaction optimization benchmarks**.
- **Curated Catalog**: Indexes open reaction/experiment optimization benchmarks (`EDBO`, `EDBO+`, `rxn_yields`, `Olympus`), `12` open-source BO & conformal libraries (`BoTorch`, `Ax`, `MAPIE`, `BayBE`, `TuRBO`, `SafeOpt`, `Summit`, `Gryffin`), and `12` foundational & 2024-2026 papers.

### Model
Semiparametric `FrontierEnsembleRegressor` (`surrogate="frontier_ensemble"`, out-of-fold stacked blend of core & full analytic SVD Ridge, multi-scale Universal-Kriging dual-space RBF/RQ + Tanimoto Kernel Ridge with closed-form Cholesky LOOCV, and residual/direct LightGBM) alongside Huber `GradientBoostingRegressor` / `HistGradientBoostingRegressor` / `LightGBM` surrogates, paired with finite-sample Mondrian Split Conformal Prediction ($\\alpha = 0.10$, exact rational order-statistic rank $\\lceil(n_{{\\text{{cal,in}}}}+1)(1-\\alpha)\\rceil$ aligned with `LabDecideLean.conformalRank`, returning $\\infty$ without clamping when $n < (1-\\alpha)/\\alpha$), centered residual-quantile `CQR-r` (`ResidualQuantileBandModel`), hybrid Tanimoto + descriptor $k$-NN + cross-family epistemic disagreement gating (`tanimoto_mahalanobis`, `ensemble_mahalanobis`, `mahalanobis`, `euclidean`), deterministic balanced group-packing splits (`--split scaffold` and `--split group`), and `R`-round sequential closed-loop optimization (`run_sequential_bo_campaign`).

### Training
CPU-only fitting, Mondrian conformal calibration, and active boundary expansion across a **4-way disjoint split** (`train 561 / pool 224 / calibration 169 / test 169` on `1,123` deduplicated ESOL compounds):
- **Frontier `rdkit_2d_morgan + frontier_ensemble` (`20` seeds `0..19`)**: On random splits, overall test MAE is **`{fmt(fr_m['mae_all'], 3)}`** (`RMSE {fmt(fr_m['rmse_all'], 3)}`, $R^2 = {fmt(fr_m['r2_all'], 3)}$), beating Delaney's (2004) 4-term linear formula (`MAE {fmt(del_r['mae_all'], 3)}`) by **`{h2h_r['mae_reduction_vs_delaney_csv_pct']:.1f}%`** and 6-descriptor GBDT by **`{h2h_r['mae_reduction_vs_gbdt_6desc_pct']:.1f}%`**, with inside coverage **`{fmt(fr_m['cov_in'], 3)}`** (`MAE_in {fmt(fr_m['mae_in'], 3)}` vs `MAE_out {fmt(fr_m['mae_out'], 3)}`, `width_in {fmt(fr_m['width_in'], 3)}`, `20/20` seeds `MAE_in < MAE_out`). On balanced Bemis-Murcko scaffold splits (max split-fraction error `{100.0 * float(esol20_s['max_achieved_fraction_abs_error']):.2f}` percentage points), overall MAE is **`{fmt(fs_m['mae_all'], 3)}`** vs **`{fmt(del_s['mae_all'], 3)}`** (Delaney 2004 CSV) and **`{fmt(ols_s['mae_all'], 3)}`** (out-of-sample 4-descriptor OLS).
- **No-RDKit `descriptors + gbdt` Baseline (`seed = 0`, `demo/output.txt`)**: Selective Mondrian conformal achieves `0.936` inside coverage (`MAE 0.640`, width `3.066`, `n = 141` answered) vs `1.217` MAE outside (`width 3.836`, `n = 28` abstained).

### History
Initialized in `41dae77` (`2026-11-18`) with `demo/conformal_trust.py`, reference output (`demo/output.txt`), Python 3.12 CI workflow (`.github/workflows/demo.yml`), and curated Bayesian optimization & conformal prediction resources (`2012`-`2026`). Hardened with Lean 4.36.0 formal proofs (`lean/LabDecideLean.lean`), balanced group-packing splits, semiparametric `rdkit_2d_morgan + frontier_ensemble` molecular QSAR & Delaney-2004 head-to-head benchmarks (`results/registry_benchmark.json`), and 10-seed sequential BO campaigns (`results/campaign_benchmark.json`).

### Unique Properties
- **Decoupled Interval Calibration & Abstention**: Separates finite-sample conformal interval calibration ($\\hat{{q}}$) from applicability-domain gating ($d(x) \\le \\hat{{\\tau}}$) without Gaussian homoscedastic assumptions.
- **Methodological Transparency**: Reports `data_source` (`real_csv` vs `synthetic_fixture`) on every output, benchmarks directly against both Delaney's (2004) in-sample formula and out-of-sample refitted 4-descriptor OLS across random and scaffold splits, distinguishes single-batch trust-region ranking from multi-round closed-loop campaigns, and marks every dataset/encoding configuration where `gate_informative = False`.
- **Ecosystem Interfaces**: Pairs at `L6` with [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why), aligns upstream with [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), [`lab-attrib`](https://github.com/yaniguan/lab-attrib), [`WignerFlow`](https://github.com/yaniguan/WignerFlow), and [`assayfold`](https://github.com/yaniguan/assayfold), and gates downstream campaigns in [`buy-or-make`](https://github.com/yaniguan/buy-or-make), [`labshift`](https://github.com/yaniguan/labshift), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), and [`lab-verify`](https://github.com/yaniguan/lab-verify).

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` exposes `process_packet(packet: dict) -> dict` (`REPO_ID = "lab-decide"`, `LAYER = "L6"`) for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) execution and learning loop:

- **Frozen SHA-Pinned ESOL Frontier `ConformalTrustModel`**: Fits and caches `ConformalTrustModel` (`seed = 0`, 4-way split `561/224/169/169`, `model_encoding = "rdkit_2d_morgan"`, `surrogate = "frontier_ensemble"`, `threshold = 0.6750`, `n_calib = 169`) from the SHA-256-verified ESOL CSV (`ESOL_SHA256`). Fails closed (`halted = True`, `halt_reason = "L6_ABSTAIN"`) when the verified ESOL CSV is missing or corrupt.
- **Exact RDKit Molecular Scoring**: Featurizes `candidate_smiles` via `smiles_to_rdkit_2d_morgan` (never falling back to string-length heuristics) and evaluates predictions, calibrated intervals, and canonical 6D ESOL kNN trust-region membership directly via `ConformalTrustModel.predict`. Missing RDKit or unparseable SMILES fail closed (`L6_ABSTAIN`), and out-of-domain molecules (such as C60 fullerene, cyclosporin A, and sucrose) abstain automatically.
- **Explicit Distance Provenance**: Uses the fitted ESOL model's kNN distance by default (`distance_source = "fitted_esol_model"`); honors packet `knn_distance` only when `distance_source == "upstream"`, recording `distance_source` inside `packet["decision"]`.
- **Residual-Gated Ledger Recalibration**: Deduplicates 64-hex SHA-256 `certified_ledger_entries` from [`lab-attrib`](https://github.com/yaniguan/lab-attrib), adds genuine `(smiles/descriptors, measured_y)` / residual calibration observations to recompute conformal quantiles and `conformal_threshold`, and never widens `effective_threshold` by counting entries or when an observation's nonconformity score exceeds `q_in`.

## Repository Layout

```text
lab-decide/
├── .github/workflows/
│   └── demo.yml                      # CI: Python+RDKit+LightGBM, no-RDKit job, and Lean 4.36.0 #print axioms job
├── community_extensions/             # Mondrian-ACI drift calibrator & stepped annular Chem-RRT explorer
├── demo/
│   ├── calisol_conformal.py          # CALiSol-23 publication-DOI conformal & kernel-support benchmark
│   ├── conformal_trust.py            # Mondrian conformal + 4-way split expansion + single-batch ranking demo
│   └── output.txt                    # Deterministic reference stdout capture on ESOL (CPU)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── lean/
│   ├── AuditAxioms.lean              # Compile-time #print axioms audit across all 23 theorems
│   └── LabDecideLean.lean            # Lean 4 formalization of finite-sample conformal & frontier guards
├── results/
│   ├── campaign_benchmark.json       # Committed 10-seed sequential BO campaign benchmark (real_csv)
│   └── registry_benchmark.json       # Committed 11-dataset + 20-seed Delaney head-to-head benchmark (real_csv)
├── scripts/
│   ├── benchmark_campaign.py         # Multi-round closed-loop BO benchmark (GP-EI, LightGBM+CQR, TrustRegionAdvisor)
│   ├── benchmark_registry.py         # 11-dataset multi-seed conformal, Delaney head-to-head & encoding ablations
│   ├── check_lean_axioms.py          # Lean 4 #print axioms verifier (zero sorryAx / custom axioms)
│   ├── fetch_open_datasets.py        # On-demand atomic downloader, live URL checker & calibrator
│   └── sync_readme.py                # Deterministic README table synchronizer & drift checker
├── src/lab_decide/                   # Installable package & SHA-pinned ESOL reference asset
├── tests/                            # Offline pytest suite (with RDKit & without RDKit)
├── autolab_wire.py                   # L6 Auto-Lab 15-repository wire node (process_packet)
├── .gitignore                        # Excludes downloaded data/ & build caches
├── LICENSE                           # MIT License (code)
├── pyproject.toml                    # Package metadata & `lab-decide` CLI console script
├── README.md                         # Overview, measured benchmarks, and curated BO & conformal catalog
└── requirements.txt                  # Bounded core dependencies (numpy, pandas, scikit-learn)
```

## Quickstart & CLI

```bash
pip install -r requirements.txt -e ".[dev,chem,bo]"
python demo/conformal_trust.py
lab-decide --encoding rdkit_2d_morgan --surrogate frontier_ensemble --distance-metric tanimoto_mahalanobis --nonconformity cqr_r
lab-decide --encoding rdkit_2d_morgan --surrogate frontier_ensemble --distance-metric tanimoto_mahalanobis --nonconformity cqr_r --split scaffold
lab-decide --dataset freesolv --encoding rdkit_2d_morgan --surrogate frontier_ensemble --distance-metric tanimoto_mahalanobis --nonconformity cqr_r
lab-decide --dataset bigsoldb_v2_2 --split group
lab-decide --dataset buchwald_hartwig_doyle --encoding onehot_morgan --distance-metric ensemble_mahalanobis --nonconformity cqr_r
lab-decide --dataset aqsoldb --offline --allow-synthetic --json
python scripts/fetch_open_datasets.py --all
python scripts/benchmark_registry.py --seeds 5 --print-markdown
python scripts/benchmark_campaign.py --seeds 10 --rounds 6 --batch-size 8 --print-markdown
lab-decide --export-schema decision_envelope.schema.json
```

Output (`demo/output.txt`, deterministic 6-descriptor no-RDKit compatible CPU capture):

```
{demo_txt}
```

## Benchmarks

{reg_md}

{camp_md}

## Datasets

### Integrated Open Calibration Registry (`OpenCalibrationDataset`)

All `11` datasets in `lab_decide.datasets.OPEN_CALIBRATION_DATASETS` carry live-verified HTTPS download URLs, SPDX open-license metadata (`commercial_use_ok = True`), explicit `objective_direction` and domain `group_keys`, leakage-free CSV/TSV/JSON/JSONL schema loaders, and on-demand atomic caching under gitignored `data/`:

| `dataset_id` | Domain | Target Unit | Direction | Group Key | Records | SPDX License | Source & Reference |
|---|---|---|---|---|---:|---|---|
| [`esol_delaney`](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | `molecular_solubility` | `log10(mol/L)` | `maximize` | `murcko_scaffold` | `1,128` | `MIT` | Delaney, *JCICS* 2004 ([`10.1021/ci034243x`](https://doi.org/10.1021/ci034243x)) / MoleculeNet |
| [`aqsoldb`](https://doi.org/10.7910/DVN/OVHAW8) | `molecular_solubility` | `log10(mol/L)` | `maximize` | `murcko_scaffold` | `9,982` | `CC0-1.0` | Sorkun et al., *Sci. Data* 2019 ([`10.1038/s41597-019-0151-1`](https://doi.org/10.1038/s41597-019-0151-1)) / Harvard Dataverse |
| [`bigsoldb_v2_2`](https://doi.org/10.5281/zenodo.22648301) | `molecular_solubility` | `log10(mol/L)` | `maximize` | `solute` | `125,752` | `CC-BY-4.0` | Krasnov et al., *Sci. Data* 2025 / Zenodo 2026 ([`10.1038/s41597-025-05559-8`](https://doi.org/10.1038/s41597-025-05559-8)) |
| [`calisol23`](https://doi.org/10.11583/DTU.24559960) | `electrolyte_solvation` | `log1p(mS/cm)` | `maximize` | `doi` | `13,825` | `CC-BY-4.0` | de Blasio et al., *Sci. Data* 2024 ([`10.1038/s41597-024-03575-8`](https://doi.org/10.11583/DTU.24559960)) |
| [`freesolv`](https://github.com/MobleyLab/FreeSolv) | `admet` | `kcal/mol` | `calibration_only` | `murcko_scaffold` | `642` | `CC-BY-4.0` | Mobley & Guthrie, *JCAMD* 2014 ([`10.1007/s10822-014-9747-x`](https://doi.org/10.1007/s10822-014-9747-x)) |
| [`lipophilicity_astrazeneca`](https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/) | `admet` | `logD7.4` | `calibration_only` | `murcko_scaffold` | `4,200` | `CC-BY-SA-3.0` | AstraZeneca ChEMBL3301361 ([`10.6019/CHEMBL3301361`](https://doi.org/10.6019/CHEMBL3301361)) / TDC & ADMET-AI |
| [`biogen_adme`](https://github.com/molecularinformatics/Computational-ADME) | `admet` | `log10(mL/min/kg)` | `minimize` | `murcko_scaffold` | `3,521` | `MIT` | Fang et al., *JCIM* 2023 ([`10.1021/acs.jcim.3c00160`](https://doi.org/10.1021/acs.jcim.3c00160)) / Biogen Research |
| [`buchwald_hartwig_doyle`](https://github.com/emdgroup/baybe) | `reaction_optimization` | `yield_pct` | `maximize` | `aryl_halide` | `4,599` | `Apache-2.0` | Ahneman et al., *Science* 2018 ([`10.1126/science.aar5169`](https://doi.org/10.1126/science.aar5169)) / BayBE HTE |
| [`suzuki_miyaura_perera`](https://github.com/rxn4chemistry/rxn_yields) | `reaction_optimization` | `yield_pct` | `maximize` | `substrate` | `5,760` | `MIT` | Perera et al., *Science* 2018 ([`10.1126/science.aap9112`](https://doi.org/10.1126/science.aap9112)) / `rxn_yields` |
| [`direct_arylation_shields`](https://github.com/b-shields/edbo) | `reaction_optimization` | `yield_pct` | `maximize` | `ligand` | `1,728` | `Apache-2.0` | Shields et al., *Nature* 2021 ([`10.1038/s41586-021-03213-y`](https://doi.org/10.1038/s41586-021-03213-y)) / EDBO & BayBE |
| [`bh_hte_ood_2026`](https://doi.org/10.5281/zenodo.19636649) | `reaction_optimization` | `yield_pct` | `maximize` | `aryl_halide` | `27,497` | `CC-BY-4.0` | Neves et al., *Nat. Comput. Sci.* 2026 ([`10.1038/s43588-026-01017-6`](https://doi.org/10.1038/s43588-026-01017-6)) / Schwaller & J&J |

### Additional Optimization Frameworks & Suites

| Suite | What | License |
|---|---|---|
| [EDBO reaction data](https://github.com/b-shields/edbo) | Reaction optimization data used to benchmark BO | MIT |
| [EDBO+](https://github.com/doyle-lab-ucla/edboplus) | Multi-objective reaction optimization | See repo |
| [Reaction yields (`rxn_yields`)](https://github.com/rxn4chemistry/rxn_yields) | High-throughput cross-coupling yield benchmarks | MIT |
| [Olympus](https://github.com/the-matter-lab/olympus) | Benchmark surfaces and experiment-planning datasets | MIT |

## Open-source repos

Stars as of 2026-11-18.

| Repo | What | Stars | License |
|---|---|---:|---|
| [meta-pytorch/botorch](https://github.com/meta-pytorch/botorch) | Bayesian optimization in PyTorch | 3,655 | MIT |
| [meta/Ax](https://github.com/meta/Ax) | Adaptive Experimentation Platform | 2,845 | MIT |
| [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | A scikit-learn-compatible library for estimating prediction intervals and controlling... | 1,690 | BSD-3-Clause |
| [valeman/awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction) | A professionally curated list of awesome Conformal Prediction videos, tutorials,... | 1,344 | no license file |
| [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction) | Lightweight, useful implementation of conformal prediction on real data. | 1,112 | MIT |
| [emdgroup/baybe](https://github.com/emdgroup/baybe) | Bayesian Optimization and Design of Experiments | 530 | Apache-2.0 |
| [uber-research/TuRBO](https://github.com/uber-research/TuRBO) | - | 239 | custom (see repo) |
| [b-shields/edbo](https://github.com/b-shields/edbo) | Experimental Design via Bayesian Optimization | 184 | MIT |
| [sustainable-processes/summit](https://github.com/sustainable-processes/summit) | Optimising chemical reactions using machine learning | 156 | MIT |
| [befelix/SafeOpt](https://github.com/befelix/SafeOpt) | Safe Bayesian Optimization | 154 | MIT |
| [the-matter-lab/olympus](https://github.com/the-matter-lab/olympus) | Olympus: a benchmarking framework for noisy optimization and experiment planning | 109 | MIT |
| [the-matter-lab/gryffin](https://github.com/the-matter-lab/gryffin) | - | 61 | Apache-2.0 |

## Papers

| Paper | Venue | Link |
|---|---|---|
| Practical Bayesian Optimization of Machine Learning Algorithms | NeurIPS 2012 | [arXiv:1206.2944](https://arxiv.org/abs/1206.2944) |
| BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization | NeurIPS 2020 | [arXiv:1910.06403](https://arxiv.org/abs/1910.06403) |
| Scalable Global Optimization via Local Bayesian Optimization | NeurIPS 2019 | [arXiv:1910.01739](https://arxiv.org/abs/1910.01739) |
| Safe Controller Optimization for Quadrotors with Gaussian Processes | ICRA 2016 | [arXiv:1509.01066](https://arxiv.org/abs/1509.01066) |
| A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | arXiv 2021 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) |
| Bayesian Optimization with Conformal Prediction Sets | AISTATS 2023 | [arXiv:2210.12496](https://arxiv.org/abs/2210.12496) |
| Bayesian Optimization with Formal Safety Guarantees via Online Conformal Prediction | IEEE J. Sel. Topics Signal Process. | [arXiv:2306.17815](https://arxiv.org/abs/2306.17815) |
| Olympus: a benchmarking framework for noisy optimization and experiment planning | MLST 2021 | [arXiv:2010.04153](https://arxiv.org/abs/2010.04153) |

## Conferences and workshops

- [NeurIPS (main track, Datasets and Benchmarks, workshops)](https://neurips.cc/)
- [ICML](https://icml.cc/)
- [COPA - Symposium on Conformal and Probabilistic Prediction](https://copa-conference.com/)
- [UAI - Uncertainty in Artificial Intelligence](https://www.auai.org/)
- [AI for Science workshop series (NeurIPS / ICML / ICLR)](https://ai4sciencecommunity.github.io/)

## Progress

Recent results, newest first. Date = first arXiv version; each line is a short summary of the abstract.

| Date | Paper | In one line |
|---|---|---|
| 2026-11-08 | [A Physical Testbed for Self-Driving Laboratories: Comparing Bayesian and Agentic Optimization in a Minimal Closed-Loop Color Formulation Task](https://arxiv.org/abs/2611.11690) | A physical self-driving lab testbed comparing Bayesian and agentic optimization on a closed-loop color task. |
| 2026-07-06 | [Safe Bayesian Optimization with Counterfactual Policies](https://arxiv.org/abs/2607.05620) | Safe BO with counterfactual policies: conformal intervals for an unobserved baseline, used as the safety constraint. |
| 2025-08-08 | [Benchmarking Self-Driving Labs](https://arxiv.org/abs/2508.06642) | Benchmarking Self-Driving Labs: review of acceleration factors across the literature (median 6). |
| 2024-11-26 | [Robust Bayesian Optimization via Localized Online Conformal Prediction](https://arxiv.org/abs/2411.17387) | Robust BO via localized online conformal prediction: recalibrates the surrogate where it is wrong. |

## Contributing

Found a broken link or a missing dataset, repo or paper? Open an issue or a pull request.

## License

Code in this repo: MIT. Listed datasets, repos and papers keep their own licenses.
"""


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    expected = build_readme_text()
    readme_path = REPO_ROOT / "README.md"
    if check_only:
        actual = readme_path.read_text(encoding="utf-8")
        if actual != expected:
            print("README.md is out of sync with benchmark JSONs / demo/output.txt", file=sys.stderr)
            return 1
        print("README.md is byte-for-byte synchronized.")
        return 0
    readme_path.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Synchronized {readme_path} ({len(expected.encode('utf-8'))} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
