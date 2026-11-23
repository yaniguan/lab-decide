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
- **Curated Catalog**: Indexes `5` reaction/experiment optimization benchmarks (`ESOL`, `EDBO`, `EDBO+`, `rxn_yields`, `Olympus`), `12` open-source BO & conformal libraries (`BoTorch`, `Ax`, `MAPIE`, `BayBE`, `TuRBO`, `SafeOpt`, `Summit`, `Gryffin`), and `12` foundational & 2024–2026 papers.

### Model
`GradientBoostingRegressor(random_state=0)` paired with finite-sample Split Conformal Prediction ($\alpha = 0.10$, quantile level $\min(1, \lceil(n_{\text{cal}}+1)(1-\alpha)\rceil / n_{\text{cal}})$ with `method="higher"`) and a standardized 6D Euclidean `NearestNeighbors(n_neighbors=5)` applicability-domain gate (`StandardScaler` + 80th-percentile calibration distance threshold $\tau = 0.758$).

### Training
CPU-only fitting, split conformal calibration, and 3-round simulated boundary acquisition (`< 1s` on CPU; `numpy`, `pandas`, `scikit-learn`). On ESOL (`n_test = 226`), split conformal achieves **`0.912` marginal test coverage** (`q = 1.522`, width `3.044`, MAE `0.704`), stratifying cleanly into **`0.656` MAE / `0.922` coverage inside** (`n = 180`, `79.6%` answered) vs. **`0.892` MAE / `0.870` coverage outside** (`n = 46`, `20.4%` abstained). Locally adaptive intervals (CQR), separate pool/test splits, and `BoTorch` acquisition loops remain roadmapped next steps.

### History
Initialized in `41dae77` (`2026-11-18`) with `demo/conformal_trust.py`, reference output (`demo/output.txt`), Python 3.12 CI workflow (`.github/workflows/demo.yml`), and curated Bayesian optimization & conformal prediction resources (`2012`–`2026`).

### Unique Properties
- **Decoupled Interval Calibration & Abstention**: Separates finite-sample conformal interval calibration ($\hat{q}$) from $k$-NN applicability-domain gating ($d_{\text{5-NN}}(x) \le \tau$) without relying on Gaussian homoscedastic assumptions.
- **Methodological Transparency**: Explicitly highlights pool-shrinking artifacts (`180` $\rightarrow$ `176` answered test points across active acquisition rounds) rather than overclaiming OOD generalization.
- **Ecosystem Interfaces**: Pairs at `L6` with [`lab-ask-why`](https://github.com/yaniguan/lab-ask-why) (Socratic claim/rationale probing), aligns upstream with [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` support-gated `CALiSol-23` & `ScalarObservableTemplate`), [`WignerFlow`](https://github.com/yaniguan/WignerFlow), and [`assayfold`](https://github.com/yaniguan/assayfold), and gates downstream campaigns in [`labshift`](https://github.com/yaniguan/labshift), [`buy-or-make`](https://github.com/yaniguan/buy-or-make) (`L5`), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), and [`lab-verify`](https://github.com/yaniguan/lab-verify) (`L4`).

## Repository Layout

```text
lab-decide/
├── .github/workflows/
│   └── demo.yml                      # CI workflow running demo/conformal_trust.py on Python 3.12
├── demo/
│   ├── conformal_trust.py            # Split conformal + 5-NN trust-region abstention & active expansion
│   └── output.txt                    # Reference stdout capture on ESOL (CPU)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── .gitignore                        # Excludes downloaded ESOL data/
├── LICENSE                           # MIT License (code)
├── README.md                         # Overview, demo results, and curated BO & conformal catalog
└── requirements.txt                  # Dependencies (numpy, pandas, scikit-learn)
```

## First attempt

Data: ESOL aqueous solubility (1128 compounds, Delaney 2004). Inputs: six simple descriptors from the file. The file's own 'ESOL predicted' column is not used. Gradient boosting + split conformal prediction at 90%. A point is in the trusted region if its kNN distance to training data is below the 80th percentile of calibration distances. Then three rounds of "new lab data" move the nearest outside test points into training (20, 20, then the last 10).

```bash
pip install numpy pandas scikit-learn
python demo/conformal_trust.py
```

Output (Linux, CPU only, 2026-11-18):

```
ESOL rows: 1128 | train 676 / calibration 226 / test 226
Features: Minimum Degree, Molecular Weight, Number of H-Bond Donors, Number of Rings, Number of Rotatable Bonds, Polar Surface Area

== Split conformal (target 90%) ==
Conformal half-width q: 1.522 log units
Test coverage: 0.912 | mean interval width: 3.044 | MAE: 0.704

== Trust region (kNN k=5, threshold = 80th pct of calibration distances = 0.758) ==
region      n  coverage   width     MAE
inside    180     0.922   3.044   0.656
outside    46     0.870   3.044   0.892
API answers 0.796 of test points, abstains on 0.204

== Expansion: 3 rounds, 20 nearest outside points per round moved to training ==
round  train  test  answered  cov_in  cov_out  cov_all  MAE_in  MAE_out  width
0        676   226     0.796   0.922    0.870    0.912   0.656    0.892  3.044
1        696   206     0.854   0.932    0.767    0.908   0.656    0.942  3.134
2        716   186     0.946   0.938    0.800    0.930   0.640    0.847  3.102
3        726   176     1.000   0.949      nan    0.949   0.642      nan  3.210
```

Coverage stays near the 90% target, and error is lower inside the region (MAE 0.66) than outside (0.89). The answered fraction rises from 0.80 to 1.00, but read this with care: the number of answered test points stays about the same (180, then 176). The rise comes from moving outside points into training, not from the region reaching new points. Next: locally adaptive interval widths, a fresh held-out test set, and a BoTorch loop that only proposes inside the region.

## Datasets

| Dataset | What | License |
|---|---|---|
| [ESOL (Delaney 2004)](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | 1128 compounds, measured solubility | Public research data; cite [Delaney 2004](https://doi.org/10.1021/ci034243x) |
| [EDBO reaction data](https://github.com/b-shields/edbo) | Reaction optimization data used to benchmark BO | MIT |
| [EDBO+](https://github.com/doyle-lab-ucla/edboplus) | Multi-objective reaction optimization | See repo |
| [Reaction yields (Buchwald-Hartwig etc.)](https://github.com/rxn4chemistry/rxn_yields) | High-throughput yield datasets | See repo |
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
