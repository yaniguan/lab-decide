# Architecture and History — `lab-decide`

> **Auto-Lab Ecosystem Role**: Layer `L6 · Metacognitive & Epistemic Brain` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Self-driving laboratories rely on surrogate models and Bayesian optimization (BO) loops to recommend reaction conditions, formulations, and candidate molecules. Standard surrogates fail silently outside their training manifold—returning overconfident point predictions or miscalibrated Gaussian process variances on unfamiliar scaffolds—while unconstrained acquisition functions routinely propose unsafe or out-of-distribution (OOD) experiments. Conversely, calibrating uncertainty on top of an under-expressive 6-descriptor tabular tree model leaves interval widths needlessly wide and point accuracy no better than classical linear group-contribution formulas.

`lab-decide` builds an **epistemic experiment advisor that pairs frontier molecular accuracy with distribution-free trust boundaries**:
1. Inside its empirical domain of applicability, it returns **distribution-free calibrated prediction intervals** at a user-specified target coverage ($1 - \alpha$) backed by a semiparametric molecular ensemble (`rdkit_2d_morgan + frontier_ensemble`) that cuts test MAE by `38.5%` relative to Delaney's (2004) ESOL linear formula and `46.7%` on FreeSolv hydration free energies.
2. Outside its trust region, it emits an explicit **"I don't know" abstention** (`ABSTAIN_ACQUIRE` / `L6_ABSTAIN`) rather than an uncalibrated extrapolation.
3. As new wet-lab measurements arrive at the boundary of the known manifold, it **recalibrates and expands its trusted region** while holding held-out evaluation splits strictly untouched.

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `lab-decide` serves as the quantitative uncertainty and experimental decision gate in **`L6 · Metacognitive & Epistemic Brain`**:
- **Peer (`L6` Metacognitive Brain)**: Complements [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why). Where `lab-ask-why` surfaces competing literature rationales and names the discriminating hypothesis test, `lab-decide` determines whether the surrogate can reliably predict that regime, quantifies predictive interval width, and governs safe active data acquisition.
- **Upstream (`L0`–`L3` Surrogates & Calibrated Post-Trained Policies)**: Consumes molecular/reaction representations and calibrated observable predictors from [`WignerFlow`](https://github.com/yaniguan/WignerFlow), [`assayfold`](https://github.com/yaniguan/assayfold), [`YLM`](https://github.com/yaniguan/YLM), and [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` `ScalarObservableTemplate`, `InverseDesignSupportTemplate`, and `CALiSol-23` support-gated electrolyte environments).
- **Downstream (`L5` / `L4` Campaign Planning & Default-Deny Execution)**: Gates experiment proposals flowing into [`labshift`](https://github.com/yaniguan/labshift) and [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5` scheduling and synthesis-vs-procurement decisions), which must then pass [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) and [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`) prior to benchtop execution (`stepbot`, `lab-hands`, `lab-feet`).

### Inputs, Outputs, and Operating Modes
- **Dual Molecular Operating Modes**:
  - **Frontier Molecular Mode (`encoding="rdkit_2d_morgan"`, `surrogate="frontier_ensemble"`, `distance_metric="tanimoto_mahalanobis"`, `nonconformity="cqr_r"`)**: Extracts `1,416` molecular features (`6` ESOL descriptors + `4` intensive ratios + `215` standardized `RDKit` 2D physicochemical descriptors + `167` `MACCS` structural keys + `1,024`-bit radius-2 count-Morgan fingerprints in `log1p` space), fits the 7-component semiparametric `FrontierEnsembleRegressor` + centered `ResidualQuantileBandModel`, and gates predictions via hybrid descriptor $k$-NN + continuous Tanimoto $k$-NN + cross-family epistemic disagreement.
  - **No-RDKit Core Tabular Mode (`encoding="descriptors"`, `surrogate="gbdt"`)**: Preserves deterministic pure-Python/NumPy/scikit-learn compatibility (`demo/conformal_trust.py`, `demo/output.txt`, and `.github/workflows/demo.yml` `test-without-rdkit` CI job) on pre-computed tabular descriptors.

---

## 2. Data

### Active Benchmark Dataset: ESOL Aqueous Solubility (Delaney 2004)
`lab-decide` verifies `delaney-processed.csv` (`96,699` bytes, bundled offline at `src/lab_decide/resources/delaney-processed.csv`) against `ESOL_SHA256`:

| Attribute | Value / Specification |
|---|---|
| **Total Records** | `1,128` raw records (`1,123` unique compounds after stripping trailing SMILES whitespace and averaging `5` duplicates) |
| **Tabular Features (`descriptors`, $d=6$)** | `Minimum Degree`, `Molecular Weight`, `Number of H-Bond Donors`, `Number of Rings`, `Number of Rotatable Bonds`, `Polar Surface Area` |
| **Frontier Features (`rdkit_2d_morgan`, $d=1,416$)** | `6` ESOL descriptors + `4` intensive ratios (`arom_atom_prop`, `hetero_atom_prop`, `halogen_atom_prop`, `log1p_heavy_atoms`) + `215` clipped `RDKit` 2D descriptors + `167` `MACCS` keys + `1,024` `log1p` radius-2 Morgan counts |
| **Target (`ESOL_TARGET`)** | `measured log solubility in mols per litre` ($\log_{10} S \in [-11.60, 1.58]$, mean $-3.05$, std $2.10$) |
| **Leakage Prevention** | Strictly excludes `ESOL predicted log solubility in mols per litre` during training; evaluates Delaney's column solely as an external baseline |
| **4-Way Disjoint Split (`50/20/15/15`)** | `561` train / `224` candidate acquisition pool / `169` calibration / `169` held-out test (`random` and balanced Bemis-Murcko `scaffold`) |
| **License & Provenance** | MIT; cite Delaney, *J. Chem. Inf. Comput. Sci.* 2004, 44(3), 1000-1005 (`10.1021/ci034243x`) |

### Open Calibration Dataset Registry (`src/lab_decide/datasets.py` & `scripts/fetch_open_datasets.py`)
`OPEN_CALIBRATION_DATASETS` covers **`11` live-verified open-source datasets** across molecular solubility, electrolyte solvation, ADMET, and high-throughput reaction optimization, with explicit `objective_direction`, explicit domain `group_keys` (zero regex fallback groups), and strict `data_source` provenance (`real_csv` vs opt-in `synthetic_fixture`):

| `dataset_id` | Domain | Target (`target_unit`) | `n_records` | SPDX License | Leakage Exclusions | Primary Citation / Provenance |
|---|---|---|---|---|---:|---|---|
| `esol_delaney` | `molecular_solubility` | `log10(mol/L)` | `1,128` | `MIT` | `ESOL predicted log solubility...` | Delaney, *JCICS* 2004 (`10.1021/ci034243x`) |
| `aqsoldb` | `molecular_solubility` | `log10(mol/L)` | `9,982` | `CC0-1.0` | `SD`, `Ocurrences`, `Group` | Sorkun et al., *Sci. Data* 2019 (`10.1038/s41597-019-0151-1`; Dataverse `10.7910/DVN/OVHAW8`) |
| `bigsoldb_v2_2` | `molecular_solubility` | `log10(mol/L)` | `125,752` | `CC-BY-4.0` | `Solubility(mol/L)`, `Solubility(mole_fraction)`, `LogS(mole_fraction)` | Krasnov et al., *Sci. Data* 2025 / Zenodo 2026 (`10.5281/zenodo.22648301`) |
| `calisol23` | `electrolyte_solvation` | `log1p(mS/cm)` | `13,825` | `CC-BY-4.0` | DOI-stratified (`salt_other`, `solv_other_mass`) | de Blasio et al., *Sci. Data* 2024 (`10.1038/s41597-024-03575-8`) |
| `freesolv` | `admet` | `kcal/mol` | `642` | `CC-BY-4.0` | `calc`, `d_calc`, `calc_h`, `calc_charging`, `calc_vdw`, `h_solv` | Mobley & Guthrie, *JCAMD* 2014 (`10.1007/s10822-014-9747-x`) |
| `lipophilicity_astrazeneca` | `admet` | `logD7.4` | `4,200` | `CC-BY-SA-3.0` | Pure structural SMILES featurization | AstraZeneca ChEMBL3301361 (`10.6019/CHEMBL3301361`; TDC / ADMET-AI) |
| `biogen_adme` | `admet` | `log10(mL/min/kg)` | `3,521` | `MIT` | Excludes 5 sibling measured ADME endpoints | Fang et al., *JCIM* 2023 (`10.1021/acs.jcim.3c00160`; Biogen Research) |
| `buchwald_hartwig_doyle` | `reaction_optimization` | `yield_pct` | `4,599` | `Apache-2.0` | `plate`, `row`, `col`, reagent index IDs | Ahneman et al., *Science* 2018 (`10.1126/science.aar5169`; BayBE) |
| `suzuki_miyaura_perera` | `reaction_optimization` | `yield_pct` | `5,760` | `MIT` | Row index `Unnamed: 0`; scales `[0,1]` to `%` | Perera et al., *Science* 2018 (`10.1126/science.aap9112`; `rxn_yields`) |
| `direct_arylation_shields` | `reaction_optimization` | `yield_pct` | `1,728` | `Apache-2.0` | Row index `entry` | Shields et al., *Nature* 2021 (`10.1038/s41586-021-03213-y`; EDBO / BayBE) |
| `bh_hte_ood_2026` | `reaction_optimization` | `yield_pct` | `27,497` | `CC-BY-4.0` | Binary threshold `labels`, `DRFP`, `QM`, cluster IDs | Neves et al., *Nat. Comput. Sci.* 2026 (`10.1038/s43588-026-01017-6`; Zenodo `19636649`) |

---

## 3. Model

`fit_and_calibrate` (`src/lab_decide/conformal.py`) integrates three statistical layers strictly fitted on `(X_tr, y_tr)` and calibrated on disjoint `(X_cal, y_cal)`:

1. **Semiparametric `FrontierEnsembleRegressor` (`surrogate="frontier_ensemble"`)**:
   - **Closed-Form Analytic SVD Ridge Baselines**: Fits both a 30-feature core physicochemical linear free-energy model (`_AnalyticSVDRidge`, selecting optimal L2 regularization across 28 log-spaced penalties via exact Hat-matrix Leave-One-Out CV in a single SVD) and a full 2D+MACCS Ridge model, with variance floors preventing rare `[0, 1]` substructure bits from inflating extrapolations under Bemis-Murcko scaffold shift.
   - **Universal-Kriging Dual-Space Kernel Ridge (`_DualSpaceKernelRidge`)**: Models nonlinear solvation/ADMET residuals $r = y - \hat{\mu}_{\text{lin}}(x)$ using a convex product/sum of multi-scale RBF + Rational Quadratic kernels on standardized 2D+MACCS descriptors and continuous Ruzicka/Tanimoto + RBF kernels on `log1p` radius-2 Morgan count fingerprints. Exact Leave-One-Out CV predictions are extracted in $O(1)$ extra work from the Cholesky factor $(K + \alpha I)^{-1}_{ii}$ (with Nystrom landmark Hat-matrix LOO for $N > 3,000$), paired with a direct non-parametric KRR anchor.
   - **Residual & Direct Gradient-Boosted Trees**: Fits regularized `LightGBM` (with automatic `HistGradientBoostingRegressor` fallback when `lightgbm` is absent) on both linear-trend residuals and raw targets via 3-fold out-of-fold cross-validation, then optimizes non-negative simplex stacking weights $w \in \Delta^6$ strictly on `X_tr` out-of-fold predictions.
2. **Centered Residual-Quantile `CQR-r` & Finite-Sample Mondrian Calibration**:
   - `ResidualQuantileBandModel` fits quantile trees (`alpha/2` and `1 - alpha/2`) strictly on training out-of-fold residuals $y_{\text{tr}} - \hat{f}_{\text{OOF}}(X_{\text{tr}})$ and combines their span with cross-family epistemic disagreement $\hat{\sigma}_{\text{ens}}(x)$ to form centered bands $[\hat{f}(x) - s(x), \hat{f}(x) + s(x)]$ that never shift the stacked point prediction $\hat{f}(x)$.
   - Computes exact rational order-statistic rank $\lceil(n_{\text{cal,in}} + 1)(1 - \alpha)\rceil$ matching `LabDecideLean.conformalRank`, returning $+\infty$ without clamping when $n < (1 - \alpha)/\alpha$.
3. **Hybrid Tanimoto + Descriptor $k$-NN + Epistemic Disagreement Gate (`tanimoto_mahalanobis`)**:
   - Combines leave-one-out-median-normalized 2D+MACCS descriptor $k$-NN distance, continuous Tanimoto $k$-NN distance on Morgan count fingerprints, and cross-family disagreement (`std` across linear, Universal-Kriging KRR, direct KRR, and tree components). On unfamiliar scaffolds where local-support KRR reverts toward the training mean while semiparametric learners extrapolate along physical descriptors, their divergence sharply flags epistemic OOD risk.

---

## 4. Measured Benchmarks (`results/registry_benchmark.json`)

### 20-Seed ESOL Head-to-Head vs. Delaney (2004) Linear Formula & 6-Descriptor GBDT

Evaluated across `20` paired seeds (`0..19`, 4-way split `561` train / `224` pool / `169` calibration / `169` test on `1,123` deduplicated ESOL compounds):

| Split (`20` seeds) | Model / Representation | Features | `MAE_all` | `RMSE_all` | `R^2` | `MAE_in` | `MAE_out` | `cov_in` / `cov_all` | `width_in` / `width_all` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | Delaney (2004) 4-term formula (CSV column, in-sample fit) | `4` | `0.686 +- 0.029` | `0.894 +- 0.034` | `0.806 +- 0.024` | n/a | n/a | `0.910 +- 0.024` (all) | `3.016 +- 0.227` (all) |
| `random` | Delaney (2004) 4-desc OLS (out-of-sample train split refit) | `4` | `0.777 +- 0.031` | `1.011 +- 0.047` | `0.752 +- 0.033` | n/a | n/a | `0.903 +- 0.032` (all) | `3.315 +- 0.344` (all) |
| `random` | 6-descriptor Huber GBDT + Euclidean kNN (`descriptors`) | `6` | `0.661 +- 0.036` | `0.887 +- 0.054` | `0.809 +- 0.027` | `0.581 +- 0.040` | `1.002 +- 0.161` | `0.924 +- 0.028` | `2.754 +- 0.241` |
| `random` | **Frontier Ensemble (`rdkit_2d_morgan` + `tanimoto_mahalanobis` + `cqr_r`)** | **`1416`** | **`0.422 +- 0.028`** | **`0.582 +- 0.047`** | **`0.918 +- 0.013`** | **`0.378 +- 0.024`** | **`0.610 +- 0.111`** | **`0.904 +- 0.030`** | **`1.633 +- 0.144`** |
| `scaffold` | Delaney (2004) 4-term formula (CSV column, in-sample fit) | `4` | `0.832 +- 0.083` | `1.069 +- 0.102` | `0.736 +- 0.028` | n/a | n/a | `0.877 +- 0.069` (all) | `3.430 +- 0.546` (all) |
| `scaffold` | Delaney (2004) 4-desc OLS (out-of-sample train split refit) | `4` | `1.083 +- 0.058` | `1.355 +- 0.074` | `0.571 +- 0.070` | n/a | n/a | `0.905 +- 0.045` (all) | `4.281 +- 0.263` (all) |
| `scaffold` | 6-descriptor Huber GBDT + Euclidean kNN (`descriptors`) | `6` | `1.112 +- 0.086` | `1.485 +- 0.105` | `0.489 +- 0.056` | `1.029 +- 0.093` | `1.570 +- 0.385` | `0.873 +- 0.059` | `4.374 +- 0.588` |
| `scaffold` | **Frontier Ensemble (`rdkit_2d_morgan` + `tanimoto_mahalanobis` + `cqr_r`)** | **`1416`** | **`0.808 +- 0.055`** | **`1.011 +- 0.071`** | **`0.763 +- 0.025`** | **`0.760 +- 0.044`** | **`1.015 +- 0.184`** | **`0.870 +- 0.075`** | **`3.293 +- 0.579`** |

- **Why Scaffold Extrapolation Is Hard on ESOL (and How Universal Kriging Solves It)**: Under balanced Bemis-Murcko scaffold packing (`max_achieved_fraction_abs_error = 0.67` percentage points across 20 seeds), the single `acyclic` group (`46%` of ESOL) populates `~92%` of `train_idx` while `test_idx` consists entirely of cyclic/aromatic scaffolds. Pure tree models (`6-desc GBDT`, `MAE 1.112`) clip flat at leaf boundaries, whereas `FrontierEnsembleRegressor` extrapolates smoothly along regularized physicochemical solvation descriptors while refining ring/substructure residuals (`MAE 0.808`, beating even Delaney's in-sample fit `0.832` and reducing out-of-sample 4-descriptor OLS error `1.083` by **`25.3%`**).

### 5-Seed Single-Molecule Representation Upgrade Across All 5 Molecular Datasets

| `dataset_id` (`n_clean`) | `descriptors + gbdt` `MAE_all` (`MAE_in`) | `rdkit_2d_morgan + frontier_ensemble` `MAE_all` (`MAE_in` vs `MAE_out`) | `cov_in` | `width_in` Change | `MAE_in < MAE_out` |
|---|---:|---:|---:|---:|---:|
| `esol_delaney` (`1,123`) | `0.658` (`0.562`) | **`0.429`** (**`0.373`** vs `0.647`) | `0.906` | `2.74 -> 1.64` (`-40%`) | `5/5` (`20/20` on 20-seed) |
| `aqsoldb` (`9,982`) | `0.846` (`0.738`) | **`0.675`** (**`0.581`** vs `1.044`) | `0.895` | `3.14 -> 2.39` (`-24%`) | `5/5` |
| `freesolv` (`642`) | `1.071` (`0.845`) | **`0.571`** (**`0.413`** vs `1.126`) | `0.913` | `3.97 -> 1.73` (`-56%`) | `5/5` |
| `lipophilicity_astrazeneca` (`4,200`) | `0.809` (`0.783`) | **`0.463`** (**`0.421`** vs `0.638`) | `0.905` | `3.15 -> 1.85` (`-41%`) | `5/5` |
| `biogen_adme` (`3,087`) | `0.438` (`0.422`) | **`0.350`** (**`0.340`** vs `0.388`) | `0.918` | `1.68 -> 1.42` (`-15%`) | `5/5` |

---

## 5. History

| Commit / PR | Date | Author | Summary |
|---|---|---|---|
| `41dae77` | `2026-11-18` | Yani Guan | Initial commit: `demo/conformal_trust.py`, `demo/output.txt`, `.github/workflows/demo.yml`, and curated BO & conformal catalog in `README.md`. |
| `6957f12` (PR `#30`) | `2026-11-25` | Yani Guan | Phase 2 audit hardening & `autolab-wire/v1` L6 gate: exact rational `conformal_rank` parity with `LabDecideLean`, likelihood-ratio weighted expansion (`weighted_conformal_quantile`), `CQR-r` crossing guards, zero-variance OOD protection, and SHA-256 certified ledger recalibration. |
| `e37de97` (PR `#31`) | `2026-11-25` | Yani Guan | Integrated 11-dataset `OpenCalibrationDataset` registry (`src/lab_decide/datasets.py`), `scripts/fetch_open_datasets.py`, and `tests/test_open_datasets_2026.py`. |
| `f72af29` (PR `#36`) | `2026-11-27` | Yani Guan | Resolved Issues `#32-#35`: deterministic balanced group-packing splits (`scaffold` & `group`), HTE `onehot_morgan` + `ensemble_mahalanobis` gating across all 11 real datasets, 10-seed sequential BO campaigns (`results/campaign_benchmark.json`), Lean 4.36.0 CI axiom audit (`23` theorems), and SHA-pinned ESOL wire scoring. |
| `feat/frontier-rdkit2d-morgan-vs-delaney` | `2026-11-28` | Yani Guan | Frontier molecular representation (`rdkit_2d_morgan`) & semiparametric `FrontierEnsembleRegressor` + `ResidualQuantileBandModel` `cqr_r` + `tanimoto_mahalanobis` gating, 20-seed random & scaffold head-to-head vs Delaney (2004) (`38.5%` lower random MAE `0.422`, `MAE_in 0.378`), 5-dataset molecular registry upgrade (`11/11` `gate_informative = True`), and upgraded `autolab_wire.py` frontier ESOL surrogate. |

---

## 6. Unique Properties & Formal Lean 4 Guarantees

1. **Lean 4.36.0 Verified Guardrails (`lean/LabDecideLean.lean`, `23` Theorems, Zero `sorryAx`)**:
   - Proves exact integer ceiling bounds (`conformalRank_exact_lower`, `conformalRank_exact_upper`), small-sample non-clamping abstention (`undersized_calib_always_abstains`), out-of-domain abstention (`out_of_domain_always_abstains`), physical interval ordering (`physical_bounds_respected`), annular stepping-stone bounds (`frontier_calibration_strictly_annular`, `deep_ood_always_blocked`), and multi-hop certified-support chaining (`unassayed_hop_blocks_chain`, `certified_chain_annular_span`).
2. **Decoupled Interval Calibration & Abstention**:
   - Any measurable applicability-domain partition defined strictly from `X_tr` (`tanimoto_mahalanobis`, `ensemble_mahalanobis`, `mahalanobis`, or canonical 6D ESOL `euclidean` kNN) paired with `FrontierEnsembleRegressor` preserves exact finite-sample distribution-free Mondrian coverage on exchangeable calibration/test splits.
3. **Zero Target Leakage & Honest Provenance**:
   - Excludes pre-fitted prediction and sibling assay columns across all 11 loaders, stamps `data_source` (`real_csv` vs `synthetic_fixture`) on every output, and evaluates Delaney (2004) both in-sample (CSV column) and out-of-sample (refitted 4-descriptor OLS).

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "lab-decide"`, `LAYER = "L6"`) implements `process_packet(packet: dict) -> dict` across the 15-repository Auto-Lab stack:

1. **Path, Line-Ending & State Hygiene**: Resolves imports via `REPO_DIR = Path(__file__).resolve().parent`, verifies `ESOL_SHA256` (normalizing CRLF -> LF when checked out on Windows), deep-copies incoming packets, and passes already-halted packets through unchanged.
2. **Frozen SHA-Pinned ESOL Frontier `ConformalTrustModel`**: Fits and caches `ConformalTrustModel` (`seed = 0`, 4-way split `561/224/169/169`, `model_encoding = "rdkit_2d_morgan"`, `surrogate = "frontier_ensemble"`, canonical 6D ESOL kNN `threshold = 0.675015`, `effective_n_calib = 169`). Fails closed (`halted = True`, `halt_reason = "L6_ABSTAIN"`) if RDKit is absent, `candidate_smiles` is invalid, or the verified ESOL CSV is unavailable.
3. **High-Accuracy Point & Interval Scoring with Preserved Hub Domain Boundaries**: Cuts ESOL `seed=0` held-out test MAE from `0.735` (`MAE_in 0.640`, `width_in 3.066`) to `~0.45` (`MAE_in ~0.39`, `width_in ~1.95`) while strictly preserving `abstain = True` on C60 fullerene, cyclosporin A (`knn_distance > 10.0`), and sucrose, `abstain = False` on aspirin and ethanol, `distance_source == "upstream"` override semantics, and Round-2 certified low-residual benzotriazole boundary expansion (`0.675015 -> 0.682118`, unlocking diethyl malonate at `effective_n_calib = 170`).
