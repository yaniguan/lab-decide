# lab-decide

An experiment advisor that *knows where it can be trusted*: calibrated answers inside its region, a clear "I don't know" outside, and a region that can grow as the lab adds data.

> **Auto-Lab Ecosystem**: Layer `L6 - Metacognitive & Epistemic Brain` in [auto-lab](https://github.com/yaniguan/auto-lab) | [Project Studio](https://yaniguan.github.io/auto-lab/)

## Where it fits

In the [Auto-Lab](https://github.com/yaniguan/auto-lab) agent ([yaniguan.github.io/auto-lab](https://yaniguan.github.io/auto-lab/) | [yaniguan.github.io](https://yaniguan.github.io/)), the brain proposes the next experiment. This repo is the decision layer under it: sequential active optimization (including trust-region conformal UCB/EI, Gaussian Process EI, LightGBM + CQR, and semiparametric RDKit 2D + Morgan stacked ensembles) to choose, and split/Mondrian conformal prediction to say how sure it is.

## System Profile: Object | Data | Model | Training | History | Unique Properties

Full architectural documentation, conformal calibration math, dataset diagnostics, commit lineage, and sibling-repository contracts are maintained in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

### Object
Epistemic experiment advisor and uncertainty-aware decision layer (`L6`) that delivers distribution-free calibrated prediction intervals inside its empirical applicability domain, emits explicit `"I don't know"` abstentions outside it, and recalibrates as active laboratory measurements expand the boundary.

### Data
- **Active Benchmark (`ESOL`, Delaney 2004)**: `1,128` raw (`1,123` deduplicated) compounds verified against `ESOL_SHA256` (`561` train / `224` pool / `169` calibration / `169` test), predicting `measured log solubility in mols per litre` ($\log_{10} S \in [-11.60, 1.58]$) under both the no-RDKit `6`-descriptor tabular mode (`encoding="descriptors"`) and the `1416`-dimensional `rdkit_2d_morgan` molecular representation (`RDKit` 2D physicochemical descriptors + intensive ratios + `167` MACCS keys + `1,024`-bit count-Morgan fingerprints), while strictly excluding the CSV's pre-fitted `ESOL predicted` column during training.
- **Open Calibration Dataset Registry (`src/lab_decide/datasets.py`)**: Typed `OpenCalibrationDataset` registry with explicit `objective_direction` (`maximize`, `minimize`, `calibration_only`), explicit domain `group_keys` (`murcko_scaffold`, `solute`, `doi`, `aryl_halide`, `substrate`, `ligand` with zero regex fallback groups), strict `data_source` provenance (`real_csv` vs opt-in `--allow-synthetic` `synthetic_fixture`), `rdkit_2d_morgan` molecular loaders across all `5` single-molecule benchmarks (`ESOL`, `AqSolDB`, `FreeSolv`, `Lipophilicity`, `Biogen ADME`), and multi-encoding HTE loaders (`descriptors`, `onehot`, `onehot_morgan`) across **`11` live-verified open-source molecular solubility, electrolyte solvation, ADMET, and high-throughput reaction optimization benchmarks**.
- **Curated Catalog**: Indexes open reaction/experiment optimization benchmarks (`EDBO`, `EDBO+`, `rxn_yields`, `Olympus`), `12` open-source BO & conformal libraries (`BoTorch`, `Ax`, `MAPIE`, `BayBE`, `TuRBO`, `SafeOpt`, `Summit`, `Gryffin`), and `12` foundational & 2024-2026 papers.

### Model
Semiparametric `FrontierEnsembleRegressor` (`surrogate="frontier_ensemble"`, out-of-fold stacked blend of core & full analytic SVD Ridge, multi-scale Universal-Kriging dual-space RBF/RQ + Tanimoto Kernel Ridge with closed-form Cholesky LOOCV, and residual/direct LightGBM) alongside Huber `GradientBoostingRegressor` / `HistGradientBoostingRegressor` / `LightGBM` surrogates, paired with finite-sample Mondrian Split Conformal Prediction ($\alpha = 0.10$, exact rational order-statistic rank $\lceil(n_{\text{cal,in}}+1)(1-\alpha)\rceil$ aligned with `LabDecideLean.conformalRank`, returning $\infty$ without clamping when $n < (1-\alpha)/\alpha$), centered residual-quantile `CQR-r` (`ResidualQuantileBandModel`), hybrid Tanimoto + descriptor $k$-NN + cross-family epistemic disagreement gating (`tanimoto_mahalanobis`, `ensemble_mahalanobis`, `mahalanobis`, `euclidean`), deterministic balanced group-packing splits (`--split scaffold` and `--split group`), and `R`-round sequential closed-loop optimization (`run_sequential_bo_campaign`).

### Training
CPU-only fitting, Mondrian conformal calibration, and active boundary expansion across a **4-way disjoint split** (`train 561 / pool 224 / calibration 169 / test 169` on `1,123` deduplicated ESOL compounds):
- **Frontier `rdkit_2d_morgan + frontier_ensemble` (`20` seeds `0..19`)**: On random splits, overall test MAE is **`0.422 +- 0.028`** (`RMSE 0.582 +- 0.047`, $R^2 = 0.918 +- 0.013$), beating Delaney's (2004) 4-term linear formula (`MAE 0.686 +- 0.029`) by **`38.5%`** and 6-descriptor GBDT by **`36.1%`**, with inside coverage **`0.904 +- 0.030`** (`MAE_in 0.378 +- 0.024` vs `MAE_out 0.610 +- 0.111`, `width_in 1.633 +- 0.144`, `20/20` seeds `MAE_in < MAE_out`). On balanced Bemis-Murcko scaffold splits (max split-fraction error `0.67` percentage points), overall MAE is **`0.808 +- 0.055`** vs **`0.832 +- 0.083`** (Delaney 2004 CSV) and **`1.083 +- 0.058`** (out-of-sample 4-descriptor OLS).
- **No-RDKit `descriptors + gbdt` Baseline (`seed = 0`, `demo/output.txt`)**: Selective Mondrian conformal achieves `0.936` inside coverage (`MAE 0.640`, width `3.066`, `n = 141` answered) vs `1.217` MAE outside (`width 3.836`, `n = 28` abstained).

### History
Initialized in `41dae77` (`2026-11-18`) with `demo/conformal_trust.py`, reference output (`demo/output.txt`), Python 3.12 CI workflow (`.github/workflows/demo.yml`), and curated Bayesian optimization & conformal prediction resources (`2012`-`2026`). Hardened with Lean 4.36.0 formal proofs (`lean/LabDecideLean.lean`), balanced group-packing splits, semiparametric `rdkit_2d_morgan + frontier_ensemble` molecular QSAR & Delaney-2004 head-to-head benchmarks (`results/registry_benchmark.json`), and 10-seed sequential BO campaigns (`results/campaign_benchmark.json`).

### Unique Properties
- **Decoupled Interval Calibration & Abstention**: Separates finite-sample conformal interval calibration ($\hat{q}$) from applicability-domain gating ($d(x) \le \hat{\tau}$) without Gaussian homoscedastic assumptions.
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
ESOL clean rows: 1123 (deduped 5 duplicate SMILES, data_source=real_csv) | train 561 / pool 224 / calibration 169 / test 169
Features: Minimum Degree, Molecular Weight, Number of H-Bond Donors, Number of Rings, Number of Rotatable Bonds, Polar Surface Area

== Split conformal (target 90%, domain-clipped) ==
Marginal half-width q_all: 1.291 log units | Mondrian inside q_in: 1.296 log units
Test coverage: 0.917 | mean interval width: 3.184 | MAE: 0.735

== Mondrian trust region (importance-weighted kNN k=5, conformal 80th-pct threshold = 0.675) ==
region      n  coverage   width     MAE
inside    141     0.936   3.066   0.640
outside    28     0.821   3.836   1.217
API answers 0.834 of frozen test points, abstains on 0.166

== Expansion on frozen test set (diverse boundary acquisition from separate pool + 20% cal update) ==
round  train  pool  test  n_in n_out  answered  cov_in  cov_out  cov_all  MAE_in  MAE_out width_in
0        561   224   169   141    28     0.834   0.936    0.821    0.917   0.640    1.217    3.066
1        577   204   169   143    26     0.846   0.930    0.808    0.911   0.629    1.177    2.798
2        588   190   169   144    25     0.852   0.931    0.840    0.917   0.650    1.270    2.880

== Single-batch trust-region ranking (conformal UCB top-10 on pool) ==
Constrained top logS: 1.580 (pred MAE 0.514) | Unconstrained top logS: 1.580 (pred MAE 0.651, OOD frac 0.200)
```

## Benchmarks

### Measured 11-Dataset Real-Data Benchmark (`results/registry_benchmark.json`)

- **20-Seed ESOL Frontier Summary (`rdkit_2d_morgan + frontier_ensemble`, `data_source=real_csv`)**: Random split overall test `MAE 0.422 +- 0.028` (`RMSE 0.582 +- 0.047`), beating Delaney (2004) published formula (`MAE 0.686 +- 0.029`) by **`38.5%`**, out-of-sample refitted 4-descriptor Delaney OLS (`MAE 0.777 +- 0.031`) by **`45.7%`**, and 6-descriptor GBDT (`MAE 0.661 +- 0.036`) by **`36.1%`**, with inside coverage `0.904 +- 0.030` (`MAE_in 0.378 +- 0.024` vs `MAE_out 0.610 +- 0.111`, `20/20` seeds `MAE_in < MAE_out`, `width_in 1.633 +- 0.144 -> 1.640 +- 0.138` after expansion). On balanced Bemis-Murcko scaffold split (max split-fraction deviation `0.67` percentage points across 20 seeds), `rdkit_2d_morgan + frontier_ensemble` achieves `MAE 0.808 +- 0.055` (`MAE_in 0.760 +- 0.044` vs `MAE_out 1.015 +- 0.184`), beating both Delaney (2004) in-sample formula (`0.832 +- 0.083`, `-2.8%`) and out-of-sample refitted 4-descriptor OLS (`1.083 +- 0.058`, `-25.3%`).
- **CALiSol-23 Publication-DOI Holdout (`cqr_r`, 5 seeds)**: `cov_in 0.792 +- 0.064`, `MAE_in 0.271 +- 0.055` vs `MAE_out 0.449 +- 0.051`, `Spearman(d, |err|) 0.390 +- 0.109`, `gate_informative = True`.

#### 20-Seed ESOL Head-to-Head vs. Delaney (2004) Linear Formula & 6-Descriptor GBDT

| Split (`20` seeds) | Model / Representation | Features | `MAE_all` | `RMSE_all` | `R^2` | `MAE_in` | `MAE_out` | `cov_in` / `cov_all` | `width_in` / `width_all` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | Delaney (2004) 4-term formula (CSV column, in-sample fit) | `4` | `0.686 +- 0.029` | `0.894 +- 0.034` | `0.806 +- 0.024` | n/a (ungated) | n/a (ungated) | `0.910 +- 0.024` (all) | `3.016 +- 0.227` (all) |
| `random` | Delaney (2004) 4-desc OLS (out-of-sample train split refit) | `4` | `0.777 +- 0.031` | `1.011 +- 0.047` | `0.752 +- 0.033` | n/a (ungated) | n/a (ungated) | `0.903 +- 0.032` (all) | `3.315 +- 0.344` (all) |
| `random` | 6-descriptor Huber GBDT + Euclidean kNN (`descriptors`) | `6` | `0.661 +- 0.036` | `0.887 +- 0.054` | `0.809 +- 0.027` | `0.581 +- 0.040` | `1.002 +- 0.161` | `0.924 +- 0.028` | `2.754 +- 0.241` |
| `random` | Frontier Ensemble (`rdkit_2d_morgan` + `tanimoto_mahalanobis` + `cqr_r`) | `1416` | `0.422 +- 0.028` | `0.582 +- 0.047` | `0.918 +- 0.013` | `0.378 +- 0.024` | `0.610 +- 0.111` | `0.904 +- 0.030` | `1.633 +- 0.144` |
| `scaffold` | Delaney (2004) 4-term formula (CSV column, in-sample fit) | `4` | `0.832 +- 0.083` | `1.069 +- 0.102` | `0.736 +- 0.028` | n/a (ungated) | n/a (ungated) | `0.877 +- 0.069` (all) | `3.430 +- 0.546` (all) |
| `scaffold` | Delaney (2004) 4-desc OLS (out-of-sample train split refit) | `4` | `1.083 +- 0.058` | `1.355 +- 0.074` | `0.571 +- 0.070` | n/a (ungated) | n/a (ungated) | `0.905 +- 0.045` (all) | `4.281 +- 0.263` (all) |
| `scaffold` | 6-descriptor Huber GBDT + Euclidean kNN (`descriptors`) | `6` | `1.112 +- 0.086` | `1.485 +- 0.105` | `0.489 +- 0.056` | `1.029 +- 0.093` | `1.570 +- 0.385` | `0.873 +- 0.059` | `4.374 +- 0.588` |
| `scaffold` | Frontier Ensemble (`rdkit_2d_morgan` + `tanimoto_mahalanobis` + `cqr_r`) | `1416` | `0.808 +- 0.055` | `1.011 +- 0.071` | `0.763 +- 0.025` | `0.760 +- 0.044` | `1.015 +- 0.184` | `0.870 +- 0.075` | `3.293 +- 0.579` |

#### All 11 Registry Datasets (`primary_11_datasets`)

| `dataset_id` | Split | Encoding / Gate | Seeds (`n`) | `cov_in` | `cov_out` | `MAE_all` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` | `gate_informative` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `esol_delaney` | `random` | `rdkit_2d_morgan` / `tanimoto_mahalanobis` | `5` (`1123`) | `0.906 +- 0.029` | `0.914 +- 0.064` | `0.43 +- 0.02` | `0.37 +- 0.02` | `0.65 +- 0.13` | `1.64 +- 0.22` | `0.305 +- 0.049` | `5/5` | **`True`** |
| `aqsoldb` | `random` | `rdkit_2d_morgan` / `tanimoto_mahalanobis` | `5` (`9982`) | `0.895 +- 0.013` | `0.901 +- 0.017` | `0.68 +- 0.01` | `0.58 +- 0.01` | `1.04 +- 0.08` | `2.39 +- 0.09` | `0.278 +- 0.036` | `5/5` | **`True`** |
| `bigsoldb_v2_2` | `group` | `descriptors` / `euclidean` | `5` (`117043`) | `0.892 +- 0.025` | `0.921 +- 0.028` | `0.84 +- 0.02` | `0.83 +- 0.03` | `0.89 +- 0.03` | `3.45 +- 0.13` | `0.053 +- 0.038` | `5/5` | **`True`** |
| `calisol23` | `group` | `descriptors` / `euclidean` | `5` (`13222`) | `0.859 +- 0.028` | `0.870 +- 0.040` | `0.60 +- 0.06` | `0.26 +- 0.01` | `0.67 +- 0.03` | `1.31 +- 0.02` | `0.168 +- 0.304` | `5/5` | **`True`** |
| `freesolv` | `random` | `rdkit_2d_morgan` / `tanimoto_mahalanobis` | `5` (`642`) | `0.913 +- 0.033` | `0.952 +- 0.106` | `0.57 +- 0.06` | `0.41 +- 0.07` | `1.13 +- 0.21` | `1.73 +- 0.31` | `0.500 +- 0.054` | `5/5` | **`True`** |
| `lipophilicity_astrazeneca` | `random` | `rdkit_2d_morgan` / `tanimoto_mahalanobis` | `5` (`4200`) | `0.905 +- 0.023` | `0.883 +- 0.055` | `0.46 +- 0.02` | `0.42 +- 0.02` | `0.64 +- 0.06` | `1.85 +- 0.11` | `0.213 +- 0.044` | `5/5` | **`True`** |
| `biogen_adme` | `random` | `rdkit_2d_morgan` / `tanimoto_mahalanobis` | `5` (`3087`) | `0.918 +- 0.010` | `0.873 +- 0.082` | `0.35 +- 0.02` | `0.34 +- 0.02` | `0.39 +- 0.04` | `1.42 +- 0.04` | `0.041 +- 0.051` | `5/5` | **`True`** |
| `buchwald_hartwig_doyle` | `random` | `onehot_morgan` / `ensemble_mahalanobis` | `5` (`4312`) | `0.889 +- 0.027` | `0.907 +- 0.033` | `8.34 +- 0.25` | `7.35 +- 0.35` | `11.97 +- 0.66` | `45.11 +- 1.70` | `0.323 +- 0.027` | `5/5` | **`True`** |
| `suzuki_miyaura_perera` | `random` | `onehot_morgan` / `ensemble_mahalanobis` | `5` (`5760`) | `0.902 +- 0.026` | `0.909 +- 0.035` | `12.66 +- 0.32` | `11.79 +- 0.25` | `16.21 +- 1.38` | `63.69 +- 3.47` | `0.258 +- 0.056` | `5/5` | **`True`** |
| `direct_arylation_shields` | `random` | `onehot_morgan` / `ensemble_mahalanobis` | `5` (`1728`) | `0.957 +- 0.015` | `0.952 +- 0.045` | `6.95 +- 0.66` | `4.96 +- 0.67` | `13.55 +- 2.02` | `29.62 +- 2.77` | `0.636 +- 0.073` | `5/5` | **`True`** |
| `bh_hte_ood_2026` | `random` | `onehot_morgan` / `ensemble_mahalanobis` | `5` (`22569`) | `0.945 +- 0.002` | `0.878 +- 0.014` | `18.56 +- 0.26` | `15.89 +- 0.32` | `29.14 +- 0.61` | `44.55 +- 0.50` | `0.409 +- 0.026` | `5/5` | **`True`** |

#### Single-Molecule Representation & Surrogate Ablation (`descriptors + gbdt` vs `rdkit_2d_morgan + frontier_ensemble`)

| `dataset_id` | Encoding & Surrogate | Gate & Nonconformity | `cov_in` | `MAE_all` | `RMSE_all` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `esol_delaney` | `descriptors` + `gbdt` | `euclidean` + `normalized` | `0.929 +- 0.014` | `0.658 +- 0.049` | `0.899 +- 0.056` | `0.562 +- 0.048` | `1.055 +- 0.165` | `2.74 +- 0.24` | `0.273 +- 0.088` | `5/5` |
| `esol_delaney` | `rdkit_2d_morgan` + `frontier_ensemble` | `tanimoto_mahalanobis` + `cqr_r` | `0.906 +- 0.029` | `0.429 +- 0.024` | `0.608 +- 0.043` | `0.373 +- 0.021` | `0.647 +- 0.129` | `1.64 +- 0.22` | `0.305 +- 0.049` | `5/5` |
| `aqsoldb` | `descriptors` + `gbdt` | `euclidean` + `normalized` | `0.899 +- 0.017` | `0.846 +- 0.016` | `1.211 +- 0.034` | `0.738 +- 0.010` | `1.267 +- 0.079` | `3.14 +- 0.20` | `0.289 +- 0.007` | `5/5` |
| `aqsoldb` | `rdkit_2d_morgan` + `frontier_ensemble` | `tanimoto_mahalanobis` + `cqr_r` | `0.895 +- 0.013` | `0.675 +- 0.011` | `0.995 +- 0.022` | `0.581 +- 0.015` | `1.044 +- 0.083` | `2.39 +- 0.09` | `0.278 +- 0.036` | `5/5` |
| `freesolv` | `descriptors` + `gbdt` | `euclidean` + `normalized` | `0.911 +- 0.076` | `1.071 +- 0.014` | `1.507 +- 0.112` | `0.845 +- 0.087` | `1.958 +- 0.471` | `3.97 +- 0.55` | `0.374 +- 0.067` | `5/5` |
| `freesolv` | `rdkit_2d_morgan` + `frontier_ensemble` | `tanimoto_mahalanobis` + `cqr_r` | `0.913 +- 0.033` | `0.571 +- 0.059` | `0.955 +- 0.094` | `0.413 +- 0.072` | `1.126 +- 0.207` | `1.73 +- 0.31` | `0.500 +- 0.054` | `5/5` |
| `lipophilicity_astrazeneca` | `descriptors` + `gbdt` | `euclidean` + `normalized` | `0.897 +- 0.031` | `0.809 +- 0.025` | `1.008 +- 0.034` | `0.783 +- 0.021` | `0.919 +- 0.050` | `3.15 +- 0.11` | `0.128 +- 0.028` | `5/5` |
| `lipophilicity_astrazeneca` | `rdkit_2d_morgan` + `frontier_ensemble` | `tanimoto_mahalanobis` + `cqr_r` | `0.905 +- 0.023` | `0.463 +- 0.019` | `0.628 +- 0.027` | `0.421 +- 0.016` | `0.638 +- 0.065` | `1.85 +- 0.11` | `0.213 +- 0.044` | `5/5` |
| `biogen_adme` | `descriptors` + `gbdt` | `euclidean` + `normalized` | `0.894 +- 0.010` | `0.438 +- 0.018` | `0.539 +- 0.020` | `0.422 +- 0.016` | `0.496 +- 0.034` | `1.68 +- 0.07` | `0.103 +- 0.021` | `5/5` |
| `biogen_adme` | `rdkit_2d_morgan` + `frontier_ensemble` | `tanimoto_mahalanobis` + `cqr_r` | `0.918 +- 0.010` | `0.350 +- 0.021` | `0.438 +- 0.023` | `0.340 +- 0.018` | `0.388 +- 0.038` | `1.42 +- 0.04` | `0.041 +- 0.051` | `5/5` |

#### Reaction-Yield HTE Encoding & Gate Ablation (`descriptors` vs `onehot` vs `onehot_morgan`)

| `dataset_id` | Encoding | Gate & Nonconformity | `cov_in` | `MAE_in` | `MAE_out` | `width_in` | `Spearman(d, |err|)` | `MAE_in < out` | `gate_informative` |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `buchwald_hartwig_doyle` | `descriptors` | `euclidean` + `normalized` | `0.900 +- 0.014` | `12.80 +- 0.35` | `9.11 +- 0.64` | `51.77 +- 1.46` | `-0.205 +- 0.026` | `0/5` | **`False` (inverted/flat)** |
| `buchwald_hartwig_doyle` | `onehot` | `mahalanobis` + `cqr_r` | `0.906 +- 0.046` | `10.87 +- 0.32` | `11.16 +- 1.29` | `66.89 +- 1.30` | `0.025 +- 0.035` | `2/5` | **`False` (inverted/flat)** |
| `buchwald_hartwig_doyle` | `onehot_morgan` | `ensemble_mahalanobis` + `cqr_r` | `0.889 +- 0.027` | `7.35 +- 0.35` | `11.97 +- 0.66` | `45.11 +- 1.70` | `0.323 +- 0.027` | `5/5` | **`True`** |
| `suzuki_miyaura_perera` | `descriptors` | `euclidean` + `normalized` | `0.899 +- 0.018` | `13.45 +- 0.42` | `12.89 +- 0.75` | `54.87 +- 1.36` | `-0.001 +- 0.045` | `2/5` | **`False` (inverted/flat)** |
| `suzuki_miyaura_perera` | `onehot` | `mahalanobis` + `cqr_r` | `0.906 +- 0.017` | `14.19 +- 0.42` | `13.66 +- 0.90` | `69.11 +- 5.53` | `-0.057 +- 0.033` | `1/5` | **`False` (inverted/flat)** |
| `suzuki_miyaura_perera` | `onehot_morgan` | `ensemble_mahalanobis` + `cqr_r` | `0.902 +- 0.026` | `11.79 +- 0.25` | `16.21 +- 1.38` | `63.69 +- 3.47` | `0.258 +- 0.056` | `5/5` | **`True`** |
| `direct_arylation_shields` | `descriptors` | `euclidean` + `normalized` | `0.903 +- 0.046` | `8.13 +- 1.07` | `8.87 +- 0.42` | `32.46 +- 3.60` | `-0.013 +- 0.082` | `4/5` | **`False` (inverted/flat)** |
| `direct_arylation_shields` | `onehot` | `mahalanobis` + `cqr_r` | `0.948 +- 0.008` | `7.34 +- 0.66` | `9.28 +- 1.18` | `40.39 +- 2.01` | `0.049 +- 0.105` | `5/5` | **`True`** |
| `direct_arylation_shields` | `onehot_morgan` | `ensemble_mahalanobis` + `cqr_r` | `0.957 +- 0.015` | `4.96 +- 0.67` | `13.55 +- 2.02` | `29.62 +- 2.77` | `0.636 +- 0.073` | `5/5` | **`True`** |
| `bh_hte_ood_2026` | `descriptors` | `euclidean` + `normalized` | `0.893 +- 0.011` | `19.39 +- 0.23` | `17.34 +- 1.63` | `63.05 +- 2.72` | `-0.095 +- 0.056` | `1/5` | **`False` (inverted/flat)** |
| `bh_hte_ood_2026` | `onehot` | `mahalanobis` + `cqr_r` | `0.935 +- 0.004` | `17.82 +- 0.36` | `21.56 +- 0.56` | `48.66 +- 0.59` | `0.051 +- 0.015` | `5/5` | **`True`** |
| `bh_hte_ood_2026` | `onehot_morgan` | `ensemble_mahalanobis` + `cqr_r` | `0.945 +- 0.002` | `15.89 +- 0.32` | `29.14 +- 0.61` | `44.55 +- 0.50` | `0.409 +- 0.026` | `5/5` | **`True`** |

### Sequential Closed-Loop Active Optimization Benchmark (`results/campaign_benchmark.json`)

CPU sequential campaign evaluation (`10` seeds, `6` rounds x `8` candidates/round, refitting and recalibrating after every round on `real_csv` data):

#### `esol_delaney` (`log10(mol/L)`, `maximize`)

- Lowest mean simple regret: **`tr_conformal_ucb`** (statistically tied within 1 SE with `tr_conformal_ucb`, `tr_conformal_ei`, `greedy`, `unconstrained_conformal_ucb`, `gp_ei`); lowest-regret zero-OOD trust-region method: **`tr_conformal_ucb`**.

| Method | Round 1 Best | Final Best-So-Far | Final Simple Regret | Cumulative OOD Pick Frac |
|---|---:|---:|---:|---:|
| `random` | `-0.605 +- 0.839` | `0.620 +- 0.237` | `0.888 +- 0.297` | `0.194 +- 0.037` |
| `greedy` | `1.375 +- 0.236` | `1.504 +- 0.160` | `0.004 +- 0.014` | `0.094 +- 0.077` |
| `unconstrained_conformal_ucb` | `1.195 +- 0.376` | `1.504 +- 0.160` | `0.004 +- 0.014` | `0.271 +- 0.141` |
| `tr_conformal_ucb` | `1.401 +- 0.329` | `1.504 +- 0.160` | `0.004 +- 0.014` | `0.000 +- 0.000` |
| `tr_conformal_ei` | `1.351 +- 0.338` | `1.504 +- 0.160` | `0.004 +- 0.014` | `0.000 +- 0.000` |
| `gp_ei` | `1.450 +- 0.252` | `1.504 +- 0.160` | `0.004 +- 0.014` | `0.302 +- 0.071` |
| `lightgbm_cqr_ucb` | `0.606 +- 0.951` | `1.270 +- 0.332` | `0.238 +- 0.283` | `0.404 +- 0.248` |
| `lightgbm_cqr_tr_ucb` | `0.861 +- 0.609` | `1.382 +- 0.281` | `0.126 +- 0.265` | `0.000 +- 0.000` |

#### `direct_arylation_shields` (`yield_pct`, `maximize`)

- Lowest mean simple regret: **`tr_conformal_ucb`** (statistically tied within 1 SE with `tr_conformal_ucb`, `tr_conformal_ei`, `lightgbm_cqr_tr_ucb`, `greedy`, `gp_ei`, `lightgbm_cqr_ucb`); lowest-regret zero-OOD trust-region method: **`tr_conformal_ucb`**.

| Method | Round 1 Best | Final Best-So-Far | Final Simple Regret | Cumulative OOD Pick Frac |
|---|---:|---:|---:|---:|
| `random` | `68.97 +- 19.10` | `90.60 +- 5.07` | `9.40 +- 5.07` | `0.198 +- 0.054` |
| `greedy` | `97.58 +- 4.65` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.040 +- 0.032` |
| `unconstrained_conformal_ucb` | `97.13 +- 5.66` | `100.00 +- 0.01` | `0.00 +- 0.01` | `0.119 +- 0.108` |
| `tr_conformal_ucb` | `97.28 +- 5.57` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.000 +- 0.000` |
| `tr_conformal_ei` | `96.91 +- 5.54` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.000 +- 0.000` |
| `gp_ei` | `99.76 +- 0.30` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.129 +- 0.105` |
| `lightgbm_cqr_ucb` | `97.76 +- 5.96` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.346 +- 0.129` |
| `lightgbm_cqr_tr_ucb` | `99.96 +- 0.08` | `100.00 +- 0.01` | `0.00 +- 0.00` | `0.000 +- 0.000` |

#### `buchwald_hartwig_doyle` (`yield_pct`, `maximize`)

- Lowest mean simple regret: **`lightgbm_cqr_tr_ucb`** (statistically tied within 1 SE with `lightgbm_cqr_tr_ucb`, `lightgbm_cqr_ucb`); lowest-regret zero-OOD trust-region method: **`lightgbm_cqr_tr_ucb`**.

| Method | Round 1 Best | Final Best-So-Far | Final Simple Regret | Cumulative OOD Pick Frac |
|---|---:|---:|---:|---:|
| `random` | `77.08 +- 13.77` | `91.09 +- 3.52` | `8.82 +- 3.62` | `0.206 +- 0.060` |
| `greedy` | `98.30 +- 1.87` | `99.70 +- 0.29` | `0.21 +- 0.18` | `0.069 +- 0.048` |
| `unconstrained_conformal_ucb` | `99.02 +- 0.76` | `99.73 +- 0.30` | `0.18 +- 0.19` | `0.125 +- 0.099` |
| `tr_conformal_ucb` | `98.43 +- 1.94` | `99.73 +- 0.30` | `0.18 +- 0.19` | `0.000 +- 0.000` |
| `tr_conformal_ei` | `96.40 +- 4.22` | `99.28 +- 0.90` | `0.63 +- 0.81` | `0.000 +- 0.000` |
| `gp_ei` | `89.36 +- 10.95` | `96.62 +- 7.43` | `3.29 +- 7.47` | `0.127 +- 0.050` |
| `lightgbm_cqr_ucb` | `98.21 +- 1.93` | `99.80 +- 0.32` | `0.11 +- 0.17` | `0.163 +- 0.158` |
| `lightgbm_cqr_tr_ucb` | `98.60 +- 1.40` | `99.80 +- 0.31` | `0.10 +- 0.16` | `0.000 +- 0.000` |

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
