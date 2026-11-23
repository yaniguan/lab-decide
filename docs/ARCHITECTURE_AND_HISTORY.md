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

### Curated Optimization & Conformal Ecosystem Catalog
`README.md` curates:
- **5 Benchmark Datasets**: `ESOL` (Delaney 2004), `EDBO` reaction optimization (`MIT`), `EDBO+` multi-objective reaction optimization, `rxn_yields` (Buchwald-Hartwig / Suzuki HTE), and `Olympus` (`MIT`).
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

---

## 6. Unique Properties

### Statistical & Architectural Invariants
1. **Distribution-Free Finite-Sample Marginal Coverage**: Uses the exact order-statistic correction $\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil / n_{\text{cal}}$ with `method="higher"` on held-out calibration residuals rather than Gaussian homoscedastic or uncalibrated ensemble-variance heuristics.
2. **Explicit Epistemic Abstention Gate**: Separates *interval calibration* ($\hat{q}$) from *applicability-domain membership* ($d_{\text{5-NN}}(x) \le \tau$), empirically isolating lower-error predictions inside the trust region (`MAE 0.656`, `92.2%` coverage) from higher-error OOD points (`MAE 0.892`, `87.0%` coverage).
3. **Zero Target Leakage from Pre-Fitted Descriptors**: Drops Delaney's fitted `ESOL predicted` column so surrogate performance reflects pure physical/topological descriptor learning.
4. **Scientific Transparency on Evaluation Artifacts**: Explicitly warns readers in `README.md` not to misinterpret pool-shrinking `answered = 1.000` as genuine generalization to novel test compounds.

### Known Failure Modes & Audit Caveats
- **Marginal vs. Conditional (Mondrian) Coverage Under Selection**: Calibrating $\hat{q}$ once over all `X_cal` guarantees unconditional marginal coverage $P(Y \in \hat{C}(X)) \ge 1 - \alpha$, not selective coverage conditioned on $X \in \text{inside}$ or $X \in \text{outside}$ (`cov_out` drops to `0.767` in Round 1). Mondrian conformal calibration partitioned by trust-region membership resolves this gap.
- **Broken Exchangeability & Pool-Test Conflation Across Active Rounds**: Acquiring non-i.i.d. boundary points from `test_idx` while holding `X_cal` fixed violates exchangeability between calibration and surviving test points.
- **Homoscedastic Interval Width (`2 * q`)**: Absolute residuals $|y - \hat{y}|$ yield constant interval width (`3.044` log units) everywhere; Conformalized Quantile Regression (CQR) or $k$-NN-scaled residuals are needed for locally adaptive widths.
- **Random vs. Bemis-Murcko Scaffold Splitting & Discrete Descriptor Skew**: Random index splitting permits structural analogues (and `5` whitespace-masked duplicate SMILES) to span splits, while unweighted z-scored Euclidean distance gives disproportionate leverage to `Minimum Degree` (`94%` equal to `1`).

### Sibling-Repository Integration Contracts
- **[`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (`L6`)**: Supplies competing hypothesis rationales and discriminating test candidates that `lab-decide` filters for applicability-domain support and expected information gain.
- **[`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3`)**: Aligns directly with `CALiSolDesign` / `InverseDesignSupportTemplate` (kernel support gating) and `CALiSolPredict` / `ScalarObservableTemplate` (calibrated $\sigma$ rewards), enabling conformal post-calibration of RLVR policies.
- **[`labshift`](https://github.com/yaniguan/labshift) & [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5`)**: Receive trusted/abstained experiment batches and acquisition priorities for autonomous campaign scheduling and reagent sourcing.
- **[`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) & [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`)**: Enforce deterministic physical, volumetric, and formal safety gates on every experiment promoted by `lab-decide` before hardware execution (`stepbot`, `lab-hands`, `lab-feet`).
