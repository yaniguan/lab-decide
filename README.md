# lab-decide

An experiment advisor that *knows where it can be trusted*: calibrated answers inside its region, a clear "I don't know" outside, and a region that can grow as the lab adds data.

> **Auto-Lab Ecosystem**: Layer `L6 · Metacognitive & Epistemic Brain` in [auto-lab](https://github.com/yaniguan/auto-lab) · [Project Studio](https://yaniguan.github.io/auto-lab/)

## Where it fits

In the [Auto-Lab](https://github.com/yaniguan/auto-lab) agent ([yaniguan.github.io/auto-lab](https://yaniguan.github.io/auto-lab/) · [yaniguan.github.io](https://yaniguan.github.io/)), the brain proposes the next experiment. This repo is the decision layer under it: Bayesian optimization (incl. trust-region and safe variants) to choose, and conformal prediction to say how sure it is. The first attempt is a conformal "answer or abstain" demo on public solubility data.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full architectural documentation, conformal calibration math, dataset diagnostics, commit lineage, and sibling-repository contracts are maintained in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md).

### Object
Epistemic experiment advisor and uncertainty-aware decision layer (`L6`) that delivers distribution-free calibrated prediction intervals inside its empirical applicability domain, emits explicit `"I don't know"` abstentions outside it, and recalibrates as active laboratory measurements expand the boundary.

### Data
- **Active Benchmark (`ESOL`, Delaney 2004)**: `1,128` compounds fetched on demand into gitignored `data/delaney-processed.csv` (`676` train / `226` calibration / `226` test at `SEED = 0`), predicting `measured log solubility in mols per litre` ($\log_{10} S \in [-11.60, 1.58]$) from `6` physical/topological descriptors while excluding the CSV's pre-fitted `ESOL predicted` column.
- **Open Calibration Dataset Registry (`src/lab_decide/datasets.py`)**: Typed `OpenCalibrationDataset` registry and deterministic CSV/TSV/JSON/JSONL schema loaders across **`11` live-verified open-source molecular solubility, electrolyte solvation, ADMET, and high-throughput reaction optimization (BO) benchmarks** (`esol_delaney`, `aqsoldb`, `bigsoldb_v2_2`, `calisol23`, `freesolv`, `lipophilicity_astrazeneca`, `biogen_adme`, `buchwald_hartwig_doyle`, `suzuki_miyaura_perera`, `direct_arylation_shields`, and `bh_hte_ood_2026`), fetched on demand to gitignored `data/` via `scripts/fetch_open_datasets.py`.
- **Curated Catalog**: Indexes open reaction/experiment optimization benchmarks (`EDBO`, `EDBO+`, `rxn_yields`, `Olympus`), `12` open-source BO & conformal libraries (`BoTorch`, `Ax`, `MAPIE`, `BayBE`, `TuRBO`, `SafeOpt`, `Summit`, `Gryffin`), and `12` foundational & 2024–2026 papers.

### Model
Huber `GradientBoostingRegressor(loss="huber", random_state=0)` paired with finite-sample Mondrian / selective Split Conformal Prediction ($\alpha = 0.10$, exact rational order-statistic rank $\lceil(n_{\text{cal,in}}+1)(1-\alpha)\rceil$ aligned with `LabDecideLean.conformalRank`, returning $\infty$ without clamping when $n < (1-\alpha)/\alpha$), Tibshirani-Foygel-Barber likelihood-ratio weighted calibration under active boundary covariate shift, locally scaled Conformalized Quantile Regression (`cqr_r`) with monotonic `lo <= hi` crossing guards, domain-bound clipping, zero-variance OOD invariant guarding on standardized `NearestNeighbors` applicability-domain gating, RDKit SMILES-to-descriptor bridging, and `TrustRegionAdvisor` constrained Bayesian optimization (`conformal_ucb`, `conformal_ei`, `safe_expand`).

### Training
CPU-only fitting, Mondrian conformal calibration, and active boundary expansion across a **4-way disjoint split** (`train 561 / pool 224 / calibration 169 / test 169` on `1,123` deduplicated ESOL compounds). On the frozen test set (`n = 169`), selective Mondrian conformal achieves **`0.936` inside coverage** (`MAE 0.640`, adaptive width `3.066`, `n = 141` answered) vs **`1.217` MAE outside** (`width 3.836`, `n = 28` abstained), and likelihood-ratio reweighted diverse boundary acquisition from `pool` tightens `width_in` (`3.066 -> 2.880`) while preserving **`0.931` inside / `0.917` marginal coverage** on the untouched test set.

### History
Initialized in `41dae77` (`2026-11-18`) with `demo/conformal_trust.py`, reference output (`demo/output.txt`), Python 3.12 CI workflow (`.github/workflows/demo.yml`), and curated Bayesian optimization & conformal prediction resources (`2012`–`2026`).

### Unique Properties
- **Decoupled Interval Calibration & Abstention**: Separates finite-sample conformal interval calibration ($\hat{q}$) from $k$-NN applicability-domain gating ($d_{\text{5-NN}}(x) \le \tau$) without relying on Gaussian homoscedastic assumptions.
- **Methodological Transparency**: Explicitly highlights pool-shrinking artifacts (`180` $\rightarrow$ `176` answered test points across active acquisition rounds) rather than overclaiming OOD generalization.
- **Ecosystem Interfaces**: Pairs at `L6` with [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (Socratic claim/rationale probing), aligns upstream with [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` support-gated `CALiSol-23` & `ScalarObservableTemplate`), [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` post-execution causal blame & SHA-256 certified ledger), [`WignerFlow`](https://github.com/yaniguan/WignerFlow), and [`assayfold`](https://github.com/yaniguan/assayfold), and gates downstream campaigns in [`buy-or-make`](https://github.com/yaniguan/buy-or-make), [`labshift`](https://github.com/yaniguan/labshift) (`L5`), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), and [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`).

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` exposes `process_packet(packet: dict) -> dict` (`REPO_ID = "lab-decide"`, `LAYER = "L6"`) for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) execution and learning loop:

- **Upstream (`L6` Epistemic Brain)**: Consumes candidate chemistry (`candidate_smiles`) and falsifiable protocol probes emitted by [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why).
- **Lean-Verified Conformal Decision Gate (`LabDecideLean`)**: Checks both exact finite-sample split-conformal calibration rank (`ceil((n + 1) * (1 - alpha)) <= effective_n_calib`, never clamping undersized buckets) and applicability-domain distance (`knn_distance <= effective_threshold`). Violation of either invariant sets `halted = True` and `halt_reason = "L6_ABSTAIN"` (`invariant = "out_of_domain_always_abstains"`).
- **Downstream (`L5` Make-vs-Buy Triage)**: Populates `packet["decision"]` (`in_trust_region`, `abstain`, `knn_distance`, `effective_threshold`, `effective_n_calib`, `conformal_interval`) for [`buy-or-make`](https://github.com/yaniguan/buy-or-make) before multi-instrument scheduling (`labshift`), 96-well rehearsal (`lab-rehearsal`), and physical verification (`lab-verify`).
- **Closed-Loop Boundary Recalibration (`L4` & `L3`)**: Ingests deduplicated 64-hex SHA-256 `certified_ledger_entries` from [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (verified free of robotic dispense errors and unreacted precursor ghost peaks) with likelihood-ratio covariate weighting and descriptor-conditioned `CQR-r` intervals alongside calibrated `L6_ABSTAIN` / RLVR updates from [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), growing `effective_n_calib` and expanding `effective_threshold` (`+0.02` per unique certified entry up to a bounded cap) without admitting quarantined or replayed runs.

## Repository Layout

```text
lab-decide/
├── .github/workflows/
│   └── demo.yml                      # CI: cache ESOL CSV, run offline pytest, verify output within +-0.005
├── community_extensions/             # Mondrian-ACI drift calibrator & stepped annular Chem-RRT explorer
├── demo/
│   ├── calisol_conformal.py          # CALiSol-23 publication-DOI conformal & kernel-support benchmark
│   ├── conformal_trust.py            # Mondrian conformal + 4-way split expansion + constrained BO demo
│   └── output.txt                    # Deterministic reference stdout capture on ESOL (CPU)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── lean/
│   └── LabDecideLean.lean            # Lean 4 formalization of finite-sample conformal & frontier guards
├── scripts/
│   └── fetch_open_datasets.py        # On-demand atomic downloader, live URL checker & calibrator
├── src/lab_decide/                   # Installable package (ConformalTrustModel, OpenCalibrationDataset, TrustRegionAdvisor)
├── tests/
│   ├── test_autolab_wire.py          # autolab-wire/v1 contract & closed-loop recalibration tests
│   ├── test_community_extensions.py  # Mondrian-ACI & Chem-RRT unit and benchmark tests
│   ├── test_conformal_trust.py       # Offline pytest regression suite (#1..#29)
│   └── test_open_datasets_2026.py    # Registry, SPDX license, anti-leakage & multi-dataset conformal tests
├── autolab_wire.py                   # L6 Auto-Lab 15-repository wire node (process_packet)
├── .gitignore                        # Excludes downloaded ESOL data/ & caches
├── LICENSE                           # MIT License (code)
├── pyproject.toml                    # Package metadata & `lab-decide` CLI console script
├── README.md                         # Overview, demo results, and curated BO & conformal catalog
└── requirements.txt                  # Bounded dependencies (numpy, pandas, scikit-learn)
```

## Quickstart & CLI

```bash
pip install -r requirements.txt -e .
python demo/conformal_trust.py
lab-decide --nonconformity cqr --split scaffold
lab-decide --dataset calisol23 --offline
lab-decide --dataset aqsoldb --offline --json
python scripts/fetch_open_datasets.py --list
python scripts/fetch_open_datasets.py --dataset freesolv,direct_arylation_shields --calibrate
lab-decide --export-schema decision_envelope.schema.json
```

Output (`demo/output.txt`, CPU only):

```
ESOL clean rows: 1123 (deduped 5 duplicate SMILES) | train 561 / pool 224 / calibration 169 / test 169
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

== Trust-region constrained BO (conformal UCB top-10 on pool) ==
Constrained top logS: 1.580 (pred MAE 0.514) | Unconstrained greedy pred MAE: 0.651 (OOD frac 0.200)
```

Coverage stays near the 90% target, and error is lower inside the region (MAE 0.66) than outside (0.89). The answered fraction rises from 0.80 to 1.00, but read this with care: the number of answered test points stays about the same (180, then 176). The rise comes from moving outside points into training, not from the region reaching new points. Next: locally adaptive interval widths, a fresh held-out test set, and a BoTorch loop that only proposes inside the region.

## Datasets

### Integrated Open Calibration Registry (`OpenCalibrationDataset`)

All `11` datasets in `lab_decide.datasets.OPEN_CALIBRATION_DATASETS` carry live-verified HTTPS download URLs, SPDX open-license metadata (`commercial_use_ok = True`), leakage-free CSV/TSV/JSON/JSONL schema loaders, and on-demand atomic caching under gitignored `data/`:

| `dataset_id` | Domain | Target Unit | Records | SPDX License | Source & Reference |
|---|---|---|---:|---|---|
| [`esol_delaney`](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | `molecular_solubility` | `log10(mol/L)` | `1,128` | `MIT` | Delaney, *JCICS* 2004 ([`10.1021/ci034243x`](https://doi.org/10.1021/ci034243x)) / MoleculeNet |
| [`aqsoldb`](https://doi.org/10.7910/DVN/OVHAW8) | `molecular_solubility` | `log10(mol/L)` | `9,982` | `CC0-1.0` | Sorkun et al., *Sci. Data* 2019 ([`10.1038/s41597-019-0151-1`](https://doi.org/10.1038/s41597-019-0151-1)) / Harvard Dataverse |
| [`bigsoldb_v2_2`](https://doi.org/10.5281/zenodo.22648301) | `molecular_solubility` | `log10(mol/L)` | `125,752` | `CC-BY-4.0` | Krasnov et al., *Sci. Data* 2025 / Zenodo 2026 ([`10.1038/s41597-025-05559-8`](https://doi.org/10.1038/s41597-025-05559-8)) |
| [`calisol23`](https://doi.org/10.11583/DTU.24559960) | `electrolyte_solvation` | `log1p(mS/cm)` | `13,825` | `CC-BY-4.0` | de Blasio et al., *Sci. Data* 2024 ([`10.1038/s41597-024-03575-8`](https://doi.org/10.1038/s41597-024-03575-8)) |
| [`freesolv`](https://github.com/MobleyLab/FreeSolv) | `admet` | `kcal/mol` | `642` | `CC-BY-4.0` | Mobley & Guthrie, *JCAMD* 2014 ([`10.1007/s10822-014-9747-x`](https://doi.org/10.1007/s10822-014-9747-x)) |
| [`lipophilicity_astrazeneca`](https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/) | `admet` | `logD7.4` | `4,200` | `CC-BY-SA-3.0` | AstraZeneca ChEMBL3301361 ([`10.6019/CHEMBL3301361`](https://doi.org/10.6019/CHEMBL3301361)) / TDC & ADMET-AI |
| [`biogen_adme`](https://github.com/molecularinformatics/Computational-ADME) | `admet` | `log10(mL/min/kg)` | `3,521` | `MIT` | Fang et al., *JCIM* 2023 ([`10.1021/acs.jcim.3c00160`](https://doi.org/10.1021/acs.jcim.3c00160)) / Biogen Research |
| [`buchwald_hartwig_doyle`](https://github.com/emdgroup/baybe) | `reaction_optimization` | `yield_pct` | `4,599` | `Apache-2.0` | Ahneman et al., *Science* 2018 ([`10.1126/science.aar5169`](https://doi.org/10.1126/science.aar5169)) / BayBE HTE |
| [`suzuki_miyaura_perera`](https://github.com/rxn4chemistry/rxn_yields) | `reaction_optimization` | `yield_pct` | `5,760` | `MIT` | Perera et al., *Science* 2018 ([`10.1126/science.aap9112`](https://doi.org/10.1126/science.aap9112)) / `rxn_yields` |
| [`direct_arylation_shields`](https://github.com/b-shields/edbo) | `reaction_optimization` | `yield_pct` | `1,728` | `Apache-2.0` | Shields et al., *Nature* 2021 ([`10.1038/s41586-021-03213-y`](https://doi.org/10.1038/s41586-021-03213-y)) / EDBO & BayBE |
| [`bh_hte_ood_2026`](https://doi.org/10.5281/zenodo.19636649) | `reaction_optimization` | `yield_pct` | `27,497` | `CC-BY-4.0` | Neves et al., *Nat. Comput. Sci.* 2026 ([`10.1038/s43588-026-01017-6`](https://doi.org/10.1038/s43588-026-01017-6)) / Schwaller & J&J |

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
