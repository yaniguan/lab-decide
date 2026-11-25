# Architecture and History — `lab-decide`

> **Auto-Lab Ecosystem Role**: Layer `L6 · Metacognitive & Epistemic Brain` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Self-driving laboratories rely on surrogate models and Bayesian optimization (BO) loops to recommend reaction conditions, formulations, and candidate molecules. Standard surrogates fail silently outside their training manifold—returning overconfident point predictions or miscalibrated Gaussian process variances on unfamiliar scaffolds—while unconstrained acquisition functions routinely propose unsafe or out-of-distribution (OOD) experiments.

`lab-decide` builds an **epistemic experiment advisor that knows where it can be trusted**:
1. Inside its empirical domain of applicability, it returns **distribution-free calibrated prediction intervals** at a user-specified target coverage ($1 - \alpha$).
2. Outside its trust region, it emits an explicit **"I don't know" abstention** rather than a poorly supported guess.
3. As new wet-lab measurements arrive at the boundary of the known manifold, it **recalibrates and expands its trusted region**.

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `lab-decide` serves as the quantitative uncertainty and experimental decision gate in **`L6 · Metacognitive & Epistemic Brain`**:
- **Peer (`L6` Metacognitive Brain)**: Complements [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why). Where `lab-ask-why` surfaces competing literature rationales and names the discriminating hypothesis test, `lab-decide` determines whether the surrogate can reliably predict that regime, quantifies predictive interval width, and governs safe active data acquisition.
- **Upstream (`L0`–`L3` Surrogates & Calibrated Post-Trained Policies)**: Consumes molecular/reaction representations and calibrated observable predictors from [`WignerFlow`](https://github.com/yaniguan/WignerFlow), [`assayfold`](https://github.com/yaniguan/assayfold), [`YLM`](https://github.com/yaniguan/YLM), and [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` `ScalarObservableTemplate`, `InverseDesignSupportTemplate`, and `CALiSol-23` support-gated electrolyte environments).
- **Downstream (`L5` / `L4` Campaign Planning & Default-Deny Execution)**: Gates experiment proposals flowing into [`labshift`](https://github.com/yaniguan/labshift) and [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5` scheduling and synthesis-vs-procurement decisions), which must then pass [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) and [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`) prior to benchtop execution (`stepbot`, `lab-hands`, `lab-feet`).

### Inputs, Outputs, and Scope Boundaries
- **Inputs**:
  - Tabular molecular/experimental descriptor matrix $X \in \mathbb{R}^{N \times 6}$ and measured continuous scalar observable $y \in \mathbb{R}^N$ (`measured log solubility in mols per litre` from ESOL `delaney-processed.csv`).
- **Outputs**:
  - Split conformal half-width $\hat{q}$ (`1.522` $\log_{10}\text{mol/L}$ at `90%` target coverage), empirical test coverage (`0.912`), interval width (`3.044`), and MAE (`0.704`).
  - Standardized 5-NN distance trust threshold $\tau$ (`0.758`, 80th percentile of calibration distances), binary answer/abstain mask (`inside`), and stratified metrics inside (`MAE 0.656`, coverage `0.922`) vs. outside (`MAE 0.892`, coverage `0.870`) the trust region.
  - Multi-round active boundary-expansion table tracking training size, answered fraction, stratified coverage, MAE, and interval width across rounds `0`–`3`.
- **Scope Boundaries**:
  - `demo/conformal_trust.py` implements **homoscedastic split conformal prediction + $k$-NN distance abstention + boundary-point acquisition** on CPU (`numpy`, `pandas`, `scikit-learn`).
  - Locally adaptive conformal intervals (Conformalized Quantile Regression / studentized residuals), strict separation between an unlabeled acquisition pool and a frozen held-out test set, and closed-loop `BoTorch` / `TuRBO` / `SafeOpt` Bayesian optimization inside the trust region are explicitly documented as **roadmapped next steps, not yet implemented**.

---

## 2. Data

### Active Benchmark Dataset: ESOL Aqueous Solubility (Delaney 2004)
`demo/conformal_trust.py` downloads `delaney-processed.csv` (`96,699` bytes) on demand from DeepChem S3 (`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv`) into gitignored `data/delaney-processed.csv`:

| Attribute | Value / Specification |
|---|---|
| **Total Records** | `1,128` small organic compounds (`10` CSV columns) |
| **Input Features (`FEATURES`, $d=6$)** | `Minimum Degree`, `Molecular Weight`, `Number of H-Bond Donors`, `Number of Rings`, `Number of Rotatable Bonds`, `Polar Surface Area` |
| **Target (`TARGET`)** | `measured log solubility in mols per litre` ($\log_{10} S \in [-11.60, 1.58]$, mean $-3.05$, std $2.10$) |
| **Leakage Prevention** | Explicitly excludes the CSV's pre-fitted `ESOL predicted log solubility in mols per litre` linear-model column |
| **Split Protocol (`SEED = 0`)** | Two-stage random `train_test_split`: **`676` train (`60%`) / `226` calibration (`20%`) / `226` test (`20%`)** |
| **Reference Capture** | `demo/output.txt` (`19` lines, Linux CPU `2026-11-18`) |
| **License & Provenance** | Public research benchmark; cite Delaney, *J. Chem. Inf. Comput. Sci.* 2004, 44(3), 1000–1005 (`10.1021/ci034243x`) |

- **Empirical Dataset Characteristics**:
  - `Minimum Degree` is highly skewed (`1` on `1,060 / 1,128 = 94.0%` of compounds, `2` on `67` unsubstituted ring systems, `0` on `1` single-heavy-atom compound).
  - Stripping trailing whitespace on the `smiles` column reveals `5` duplicate SMILES entries in `delaney-processed.csv`; uniform index splitting does not enforce Bemis-Murcko scaffold disjointness.

### Open Calibration Dataset Registry (`src/lab_decide/datasets.py` & `scripts/fetch_open_datasets.py`)
In addition to the default ESOL benchmark, `lab-decide` provides a typed `OpenCalibrationDataset` registry (`OPEN_CALIBRATION_DATASETS`) covering **`11` live-verified open-source datasets** across molecular solubility, electrolyte solvation, ADMET, and high-throughput reaction optimization (Bayesian optimization). Every entry carries verified SPDX open licensing (`commercial_use_ok = True`), optional SHA-256 pinning, on-demand atomic caching under gitignored `data/`, and deterministic leakage-free CSV/TSV/JSON/JSONL schema loaders feeding `fit_and_calibrate`, `ConformalTrustModel`, and `TrustRegionAdvisor`:

| `dataset_id` | Domain | Target (`target_unit`) | `n_records` | SPDX License | Leakage Exclusions | Primary Citation / Provenance |
|---|---|---|---:|---|---|---|
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

### Curated Optimization & Conformal Ecosystem Catalog
`README.md` curates:
- **11 Integrated Open Calibration Datasets + 4 Optimization Suites**: Spanning `ESOL`, `AqSolDB`, `BigSolDB 2.2`, `CALiSol-23`, `FreeSolv`, `Lipophilicity_AstraZeneca`, `Biogen Computational-ADME`, `Doyle Buchwald-Hartwig HTE`, `Perera Suzuki-Miyaura HTE`, `Shields Direct Arylation`, and `BH-HTE-OOD 2026`, alongside `EDBO`, `EDBO+`, `rxn_yields`, and `Olympus`.
- **12 Open-Source BO & Conformal Libraries**: `botorch`, `Ax`, `MAPIE`, `awesome-conformal-prediction`, `aangelopoulos/conformal-prediction`, `baybe`, `TuRBO`, `edbo`, `summit`, `SafeOpt`, `olympus`, and `gryffin`.
- **12 Foundational & 2024–2026 Papers**: Spanning Snoek et al. (2012), `BoTorch` (`arXiv:1910.06403`), `TuRBO` (`arXiv:1910.01739`), `SafeOpt` (`arXiv:1509.01066`), Angelopoulos & Bates (`arXiv:2107.07511`), Conformal BO (`arXiv:2210.12496`, `arXiv:2306.17815`, `arXiv:2411.17387`), Safe Counterfactual BO (`arXiv:2607.05620`), and Physical SDL Testbeds (`arXiv:2611.11690`).

---

## 3. Model

`demo/conformal_trust.py` couples three classical statistical components in `fit_and_calibrate`:

1. **Tabular Surrogate Regressor**:
   - `GradientBoostingRegressor(random_state=0)` (`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, squared-error loss) fit strictly on `(X_tr, y_tr)`.
2. **Distribution-Free Split Conformal Calibrator**:
   - Computes nonconformity scores $s_i = |y_{\text{cal},i} - \hat{f}(x_{\text{cal},i})|$ for $i = 1, \dots, n_{\text{cal}}$ ($n_{\text{cal}} = 226$).
   - Applies the finite-sample conformal quantile level:
     $$\text{level} = \min\!\left(1.0,\; \frac{\lceil(n_{\text{cal}} + 1)(1 - \alpha)\rceil}{n_{\text{cal}}}\right), \qquad \alpha = 0.10$$
     via `np.quantile(scores, level, method="higher")`, yielding prediction intervals $\hat{C}(x) = [\hat{f}(x) - \hat{q},\; \hat{f}(x) + \hat{q}]$ of width $2\hat{q}$.
3. **Standardized $k$-NN Applicability-Domain Gate ("Answer or Abstain")**:
   - Fits `StandardScaler` on `X_tr` and `NearestNeighbors(n_neighbors=5)` in standardized 6D descriptor space.
   - Defines point-to-manifold distance $d(x) = \frac{1}{K}\sum_{j=1}^K \| \tilde{x} - \tilde{x}_{\text{tr},(j)} \|_2$ ($K = 5$).
   - Sets the trust boundary $\tau$ to the `TRUST_PCT = 80`th percentile of $\{d(x) : x \in X_{\text{cal}}\}$. The advisor **answers** when $d(x) \le \tau$ and **abstains** when $d(x) > \tau$.

---

## 4. Training

### Fitting, Calibration & Active Boundary Expansion Loop
1. **Round 0 (Initial Fit & Split Conformal Audit)**:
   - Fits GBDT, `StandardScaler`, and 5-NN on `676` training compounds; calibrates $\hat{q} = 1.522$ and $\tau = 0.758$ on `226` calibration compounds; evaluates on `226` test compounds (`< 1s` CPU runtime).
   - **Inside vs. Outside Stratification**:
     - `inside` ($n = 180$, `79.6%` answered): empirical coverage **`0.922`**, MAE **`0.656`**, width `3.044`.
     - `outside` ($n = 46$, `20.4%` abstained): empirical coverage **`0.870`**, MAE **`0.892`**, width `3.044`.
2. **Rounds 1–3 (Boundary Acquisition Simulation)**:
   - At each round $r \in \{1, 2, 3\}$, selects up to `PER_ROUND = 20` outside test points (`~inside`) with smallest $d(x)$, moves them into `train_idx` (`20`, `20`, then the remaining `10`), removes them from `test_idx`, and refits `(model, q, dist, thr)` against `X_cal`:

| Round | `train` | `test` | `answered` | `cov_in` | `cov_out` | `cov_all` | `MAE_in` | `MAE_out` | `width` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `676` | `226` | `0.796` | `0.922` | `0.870` | `0.912` | `0.656` | `0.892` | `3.044` |
| `1` | `696` | `206` | `0.854` | `0.932` | `0.767` | `0.908` | `0.656` | `0.942` | `3.134` |
| `2` | `716` | `186` | `0.946` | `0.938` | `0.800` | `0.930` | `0.640` | `0.847` | `3.102` |
| `3` | `726` | `176` | `1.000` | `0.949` | `nan` | `0.949` | `0.642`* | `nan` | `3.210` |

*\*Note: Round 3 `MAE_in` prints `0.642` on Linux (`demo/output.txt`) and `0.641` on Windows/conda due to minor tree-split floating-point tie differences at the fourth decimal place.*

### Honest Methodological Caveat (Documented in `README.md`)
As `README.md` explicitly cautions, the rise in `answered` fraction from `0.796` (`180 / 226`) to `1.000` (`176 / 176`) occurs because the `50` outside points were moved out of `test_idx` into `train_idx`, while the absolute count of answered test points stayed nearly constant (`180` $\rightarrow$ `176`). Establishing true trust-region expansion onto unseen chemistry requires separating the **unlabeled acquisition pool** from an untouched **held-out evaluation test set**.

---

## 5. History

| Commit | Date | Author | Branch / Ref | Summary |
|---|---|---|---|---|
| `41dae77` (`41dae77ebcbabf62103cbb65fefe75eb129558bc`) | `2026-11-18` | Yani Guan | `main` | Initial commit (`Add curated resources and first demo`): added `demo/conformal_trust.py` (ESOL GBDT surrogate + 90% split conformal prediction + 5-NN 80th-percentile trust-region abstention + 3-round boundary acquisition demo), `demo/output.txt`, `.github/workflows/demo.yml` (Python 3.12 CI), and curated BO / conformal datasets, libraries, and papers (`2012`–`2026`) in `README.md`. |
| `6957f12` | `2026-11-25` | Yani Guan | `main` (PR `#30`) | Phase 2 audit hardening & `autolab-wire/v1` L6 gate: exact rational `conformal_rank` parity with `LabDecideLean`, likelihood-ratio weighted expansion (`weighted_conformal_quantile`), locally scaled `CQR-r` crossing guards, zero-variance OOD protection, and SHA-256 certified ledger recalibration. |
| `feat/open-datasets-2026` | `2026-11-25` | Yani Guan | `feat/open-datasets-2026` | Integrated `OpenCalibrationDataset` registry (`src/lab_decide/datasets.py`), `scripts/fetch_open_datasets.py`, and `tests/test_open_datasets_2026.py` covering `10` new open-source molecular solubility, electrolyte solvation, ADMET, and HTE reaction optimization benchmarks (`AqSolDB`, `BigSolDB 2.2`, `CALiSol-23`, `FreeSolv`, `Lipophilicity_AstraZeneca`, `Biogen ADME`, `Doyle Buchwald-Hartwig`, `Perera Suzuki-Miyaura`, `Shields Direct Arylation`, and `BH-HTE-OOD 2026`). |

---

## 6. Unique Properties

### Statistical & Architectural Invariants
1. **Distribution-Free Finite-Sample Marginal Coverage**: Uses the exact order-statistic correction $\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil / n_{\text{cal}}$ with `method="higher"` on held-out calibration residuals rather than Gaussian homoscedastic or uncalibrated ensemble-variance heuristics.
2. **Explicit Epistemic Abstention Gate**: Separates *interval calibration* ($\hat{q}$) from *applicability-domain membership* ($d_{\text{5-NN}}(x) \le \tau$), empirically isolating lower-error predictions inside the trust region (`MAE 0.656`, `92.2%` coverage) from higher-error OOD points (`MAE 0.892`, `87.0%` coverage).
3. **Zero Target Leakage from Pre-Fitted Descriptors**: Drops Delaney's fitted `ESOL predicted` column so surrogate performance reflects pure physical/topological descriptor learning.
4. **Scientific Transparency on Evaluation Artifacts**: Explicitly warns readers in `README.md` not to misinterpret pool-shrinking `answered = 1.000` as genuine generalization to novel test compounds.

### Hardened Statistical & Architectural Invariants (Audits `#1..#22` & `#24..#29` Resolved)
- **Mondrian Selective Coverage, Exact Rational Rank & Hierarchical Fallback**: Calibrates $\hat{q}_{\text{in}}$ on $\{i \in \text{cal} : d(x_i) \le \hat{\tau}\}$ using exact rational ceiling $\lceil(n+1)(1-\alpha)\rceil$ aligned 1-to-1 with `LabDecideLean.conformalRank`, freezing `ScaffoldMondrianACI` $\alpha_t$ across abstained cold-start steps and providing hierarchical global-pool fallback when low-$n$ `trust_pct` splits starve `in_mask`.
- **Disjoint 4-Way Split (`train / pool / cal / test`) & Likelihood-Ratio Reweighted Expansion**: Acquires boundary candidates exclusively from `pool_idx` via sequential maximin $k$-NN diversity and reweights outer-shell calibration additions via Tibshirani-Foygel-Barber Radon-Nikodym likelihood ratios (`weighted_conformal_quantile` with exact rational tied-score aggregation), holding `test_idx` strictly frozen (`n_test = 169`, `answered` rising `141 -> 144`, `cov_in = 0.931`, `cov_all = 0.917`).
- **Locally Scaled `CQR-r`, Zero-Variance OOD Invariant Guards & Physical Bounds**: Supports locally scaled Conformalized Quantile Regression (`cqr_r`) with monotonic midpoint crossing guards (`lo <= hi`), asymmetric directional `TrustRegionAdvisor` acquisition (`conformal_ucb` / dynamic-$\alpha$ `conformal_ei`), zero-variance training invariant protection + `CALiSol-23` `salt_other` / `solv_other_mass` OOV gating, and certified-support `Chem-RRT` chaining in `LabDecideLean`.
- **Picklable `ConformalTrustModel`, `DecisionEnvelope` Schema & `CALiSol-23` Adapter**: Exposes serializable 1D/2D `ConformalTrustModel`, `DecisionEnvelope` JSONL contract with `lab-ask-why`, and publication-DOI `CALiSol-23` benchmarking with Winkler interval-efficiency scoring.

### Sibling-Repository Integration Contracts
- **[`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (`L6`)**: Supplies competing hypothesis rationales and discriminating test candidates that `lab-decide` filters for applicability-domain support and expected information gain.
- **[`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4`)**: Performs post-execution causal blame attribution, NNLS precursor ghost-peak deconvolution, and SHA-256 `CertifiedLedger` signing so only verified wet-lab outcomes recalibrate `lab-decide`'s conformal support.
- **[`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3`)**: Aligns directly with `CALiSolDesign` / `InverseDesignSupportTemplate` (kernel support gating), `CALiSolPredict` / `ScalarObservableTemplate` (calibrated $\sigma$ rewards), and calibrated `L6_ABSTAIN` anti-hack RLVR scoring.
- **[`buy-or-make`](https://github.com/yaniguan/buy-or-make) & [`labshift`](https://github.com/yaniguan/labshift) (`L5`)**: Receive trusted/abstained experiment packets and acquisition priorities for reagent procurement triage and multi-instrument scheduling.
- **[`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) & [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`)**: Enforce deterministic physical, volumetric, and formal safety gates on every experiment promoted by `lab-decide` before hardware execution (`stepbot`, `lab-hands`, `lab-feet`).

---

## 7. 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "lab-decide"`, `LAYER = "L6"`) implements the standardized `autolab-wire/v1` packet node `process_packet(packet: dict) -> dict` across the 15-repository Auto-Lab stack:

1. **Path & State Hygiene**: Resolves repository imports strictly via `REPO_DIR = Path(__file__).resolve().parent` without relying on working-directory state, deep-copies incoming packets, and passes already-halted packets (`halted == True`) through unchanged.
2. **Upstream Handoff (`lab-ask-why` -> `lab-decide`)**: Consumes `candidate_smiles`, `knn_distance`, `knn_threshold`, `n_calib`, and `alpha` following `L6` Socratic claim probing in [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why).
3. **Exact Finite-Sample Conformal & Domain Guardrails (`LabDecideLean`)**: Evaluates `lab_decide.conformal.conformal_quantile` and `conformal_order_rank` (`ceil((n + 1) * (1 - alpha)) <= effective_n_calib`) alongside kNN applicability-domain membership (`knn_distance <= effective_threshold`). Any out-of-domain candidate or undersized calibration bucket sets `halted = True`, `halt_reason = "L6_ABSTAIN"`, `trace[-1]["status"] = "ABSTAIN"`, and logs `invariant = "out_of_domain_always_abstains"` in `bottom_reasoning_traces`.
4. **Downstream Handoff (`lab-decide` -> `buy-or-make`)**: Emits `packet["decision"]` (`candidate_smiles`, `in_trust_region`, `abstain`, `knn_distance`, `effective_threshold`, `effective_n_calib`, and physical-bound-clipped `conformal_interval`) directly to [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5`).
5. **Closed-Loop Recalibration (`lab-attrib` & `labbench-posttrain` -> `lab-decide`)**: Validates canonical 64-hex SHA-256 `ledger_entry_sha256` digests (or explicit `admit_to_memory=True` entries with unique run IDs) and deduplicates replayed entries from [`lab-attrib`](https://github.com/yaniguan/lab-attrib), applying likelihood-ratio weighted `CQR-r` calibration, incrementing `effective_n_calib = base_n_calib + len(valid_certified)`, and expanding `effective_threshold` up to a bounded cap while [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) rewards calibrated `L6_ABSTAIN` rollouts.
