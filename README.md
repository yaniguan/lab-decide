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
Huber `GradientBoostingRegressor(loss="huber", random_state=0)` paired with finite-sample Mondrian / selective Split Conformal Prediction ($\alpha = 0.10$, exact order-statistic level $\lceil(n_{\text{cal,in}}+1)(1-\alpha)\rceil / n_{\text{cal,in}}$ returning $\infty$ without clamping when $n < (1-\alpha)/\alpha$), locally adaptive distance-normalized & CQR intervals, domain-bound clipping, importance-weighted standardized `NearestNeighbors` applicability-domain gating (finite-sample conformal outlier threshold $\hat{\tau}$), RDKit SMILES-to-descriptor bridging, and `TrustRegionAdvisor` constrained Bayesian optimization (`conformal_ucb`, `conformal_ei`, `safe_expand`).

### Training
CPU-only fitting, Mondrian conformal calibration, and 3-round active expansion across a **4-way disjoint split** (`train 561 / pool 224 / calibration 169 / test 169` on `1,123` deduplicated ESOL compounds). On the frozen test set (`n = 169`), selective Mondrian conformal achieves **`0.936` inside coverage** (`MAE 0.640`, adaptive width `3.066`, `n = 141` answered) vs **`1.217` MAE outside** (`width 3.836`, `n = 28` abstained), and diverse boundary acquisition from `pool` tightens `width_in` (`3.066 -> 2.721`) while maintaining `>= 0.910` coverage on the untouched test set.

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
│   └── demo.yml                      # CI: cache ESOL CSV, run offline pytest, verify output within +-0.005
├── demo/
│   ├── calisol_conformal.py          # CALiSol-23 publication-DOI conformal & kernel-support benchmark
│   ├── conformal_trust.py            # Mondrian conformal + 4-way split expansion + constrained BO demo
│   └── output.txt                    # Deterministic reference stdout capture on ESOL (CPU)
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md   # Deep technical profile (Object, Data, Model, Training, History)
├── src/lab_decide/                   # Installable package (ConformalTrustModel, TrustRegionAdvisor, RDKit, DecisionEnvelope)
├── tests/
│   └── test_conformal_trust.py       # Offline pytest regression suite (#1..#22)
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
2        587   191   169   144    25     0.852   0.917    0.840    0.905   0.645    1.242    2.708
3        589   189   169   144    25     0.852   0.910    0.840    0.899   0.641    1.223    2.721

== Trust-region constrained BO (conformal UCB top-10 on pool) ==
Constrained top logS: 1.580 (pred MAE 0.514) | Unconstrained greedy pred MAE: 0.651 (OOD frac 0.200)
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
