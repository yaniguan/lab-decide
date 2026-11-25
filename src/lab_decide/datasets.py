"""Typed OpenCalibrationDataset registry and deterministic CSV/TSV/JSON/JSONL schema loaders.

Covers 11 live-verified open-source molecular solubility, electrolyte solvation, ADMET, and
high-throughput reaction optimization (Bayesian optimization) benchmark datasets feeding
`fit_and_calibrate`, `ConformalTrustModel`, and `TrustRegionAdvisor`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lab_decide.acquisition import benchmark_trust_region_bo
from lab_decide.calisol import (
    COMMON_SALTS,
    COMMON_SOLVENTS,
    assign_calisol_doi_split,
    featurize_calisol_records,
    load_calisol_csv_records,
)
from lab_decide.chemistry import ESOL_FEATURES, ESOL_TARGET
from lab_decide.conformal import (
    ConformalTrustModel,
    fit_and_calibrate,
    summarize,
    winkler_interval_score,
)
from lab_decide.data import (
    ESOL_SHA256,
    ESOL_URL,
    DatasetBundle,
    SplitIndices,
    compute_file_sha256,
    partition_splits,
    validate_and_clean_dataframe,
)

DEFAULT_OPEN_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
)

VALID_DOMAINS: tuple[str, ...] = (
    "molecular_solubility",
    "electrolyte_solvation",
    "admet",
    "reaction_optimization",
)

APPROVED_COMMERCIAL_SPDX_LICENSES: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-3.0",
        "MIT",
        "Apache-2.0",
        "BSD-3-Clause",
    }
)

_DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_BRACKET_ATOM_RE = re.compile(r"\[([^\]]+)\]")
_ORGANIC_ATOM_RE = re.compile(r"Br|Cl|[BCNOPSFI]|[bcnops]")
_RING_DIGIT_RE = re.compile(r"(?:%\d\d|\d)")
_POLAR_UNSAT_RE = re.compile(r"(?:=O|=N|=S|#N)")

_ATOMIC_MASS: dict[str, float] = {
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "Pd": 106.42,
    "I": 126.904,
    "Cs": 132.905,
    "Li": 6.94,
}

AQSOLDB_FEATURES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "RingCount",
    "TPSA",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
)

MOL_PHYSICOCHEM_FEATURES: tuple[str, ...] = (
    "mw_approx",
    "heavy_atoms",
    "aromatic_atoms",
    "aliphatic_carbons",
    "ring_closures",
    "n_nitrogen",
    "n_oxygen",
    "n_sulfur_phosphorus",
    "n_fluorine",
    "n_chlorine",
    "n_bromine_iodine",
    "polar_unsaturations",
    "hbd_motifs",
    "branch_and_charge",
)

REAGENT_SLOT_FEATURES: tuple[str, ...] = (
    "present",
    "mw_approx",
    "aromatic_atoms",
    "ring_closures",
    "n_no_hetero",
    "n_ps_ligand",
    "n_fluorine",
    "n_chlorine",
    "n_bromine",
    "n_iodine",
)

BIOGEN_ALL_ASSAY_COLUMNS: tuple[str, ...] = (
    "LOG HLM_CLint (mL/min/kg)",
    "LOG MDR1-MDCK ER (B-A/A-B)",
    "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "LOG RLM_CLint (mL/min/kg)",
)


@dataclass(frozen=True)
class OpenCalibrationDataset:
    """Metadata and schema contract for an open calibration/BO benchmark dataset."""

    dataset_id: str
    domain: str
    target_unit: str
    n_records: int
    url: str
    spdx_license: str
    commercial_use_ok: bool
    title: str = ""
    citation_doi: str = ""
    landing_url: str = ""
    local_filename: str = ""
    file_format: str = "csv"
    target_column: str = ""
    domain_bounds: tuple[float, float] | None = None
    expected_sha256: str | None = None
    leakage_excluded_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _DATASET_ID_RE.fullmatch(self.dataset_id):
            raise ValueError(f"Invalid dataset_id {self.dataset_id!r}")
        if self.domain not in VALID_DOMAINS:
            raise ValueError(
                f"Invalid domain {self.domain!r} for {self.dataset_id}; expected one of {VALID_DOMAINS}"
            )
        if not self.target_unit or not self.target_unit.strip():
            raise ValueError(f"target_unit must be non-empty for {self.dataset_id}")
        if int(self.n_records) <= 0:
            raise ValueError(f"n_records must be > 0 for {self.dataset_id}, got {self.n_records}")
        if not (isinstance(self.url, str) and self.url.startswith("https://")):
            raise ValueError(f"url must be an HTTPS URL for {self.dataset_id}, got {self.url!r}")
        if not self.spdx_license or not self.spdx_license.strip():
            raise ValueError(f"spdx_license must be non-empty for {self.dataset_id}")
        if not isinstance(self.commercial_use_ok, bool):
            raise ValueError(f"commercial_use_ok must be bool for {self.dataset_id}")
        if self.commercial_use_ok and "-NC" in self.spdx_license.upper():
            raise ValueError(
                f"Non-commercial SPDX license {self.spdx_license!r} cannot set commercial_use_ok=True"
            )
        if self.domain_bounds is not None:
            lo, hi = float(self.domain_bounds[0]), float(self.domain_bounds[1])
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"Invalid domain_bounds {self.domain_bounds!r} for {self.dataset_id}")
        if self.expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256):
            raise ValueError(f"Invalid expected_sha256 for {self.dataset_id}: {self.expected_sha256!r}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.domain_bounds is not None:
            payload["domain_bounds"] = [float(self.domain_bounds[0]), float(self.domain_bounds[1])]
        payload["leakage_excluded_columns"] = list(self.leakage_excluded_columns)
        return payload


OPEN_CALIBRATION_DATASETS: dict[str, OpenCalibrationDataset] = {
    "esol_delaney": OpenCalibrationDataset(
        dataset_id="esol_delaney",
        domain="molecular_solubility",
        target_unit="log10(mol/L)",
        n_records=1128,
        url=ESOL_URL,
        spdx_license="MIT",
        commercial_use_ok=True,
        title="ESOL Aqueous Solubility Reference Benchmark (Delaney 2004 / MoleculeNet)",
        citation_doi="10.1021/ci034243x",
        landing_url="https://github.com/deepchem/deepchem",
        local_filename="delaney-processed.csv",
        file_format="csv",
        target_column=ESOL_TARGET,
        domain_bounds=(-15.0, 1.75),
        expected_sha256=ESOL_SHA256,
        leakage_excluded_columns=("ESOL predicted log solubility in mols per litre",),
    ),
    "aqsoldb": OpenCalibrationDataset(
        dataset_id="aqsoldb",
        domain="molecular_solubility",
        target_unit="log10(mol/L)",
        n_records=9982,
        url="https://dataverse.harvard.edu/api/access/datafile/3407241?format=original",
        spdx_license="CC0-1.0",
        commercial_use_ok=True,
        title="AqSolDB Curated Aqueous Solubility Dataset (Sorkun et al., Sci. Data 2019)",
        citation_doi="10.1038/s41597-019-0151-1",
        landing_url="https://doi.org/10.7910/DVN/OVHAW8",
        local_filename="aqsoldb_curated_solubility.csv",
        file_format="csv",
        target_column="Solubility",
        domain_bounds=(-15.0, 2.0),
        expected_sha256="7c47e021d03a5720f53cfddb7aef90c8911ea276a36306c839798733d8630332",
        leakage_excluded_columns=("SD", "Ocurrences", "Group"),
    ),
    "bigsoldb_v2_2": OpenCalibrationDataset(
        dataset_id="bigsoldb_v2_2",
        domain="molecular_solubility",
        target_unit="log10(mol/L)",
        n_records=125752,
        url="https://zenodo.org/records/22648301/files/BigSolDBv2.2.csv?download=1",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="BigSolDB 2.2 Multi-Solvent Temperature-Dependent Solubility (Krasnov et al., 2025/2026)",
        citation_doi="10.1038/s41597-025-05559-8",
        landing_url="https://doi.org/10.5281/zenodo.22648301",
        local_filename="BigSolDBv2.2.csv",
        file_format="csv",
        target_column="LogS(mol/L)",
        domain_bounds=(-15.0, 2.5),
        expected_sha256=None,
        leakage_excluded_columns=(
            "Solubility(mole_fraction)",
            "Solubility(mol/L)",
            "LogS(mole_fraction)",
        ),
    ),
    "calisol23": OpenCalibrationDataset(
        dataset_id="calisol23",
        domain="electrolyte_solvation",
        target_unit="log1p(mS/cm)",
        n_records=13825,
        url="https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/CALiSol-23%20Dataset.csv",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="CALiSol-23 Lithium Electrolyte Ionic Conductivity Atlas (de Blasio et al., Sci. Data 2024)",
        citation_doi="10.1038/s41597-024-03575-8",
        landing_url="https://doi.org/10.11583/DTU.24559960",
        local_filename="calisol23.csv",
        file_format="csv",
        target_column="k",
        domain_bounds=(0.0, 6.0),
        expected_sha256="af560531c6b0d7ea0848581120f5f08a725c510bba8cbc3ae84c6b89ae32993f",
        leakage_excluded_columns=(),
    ),
    "freesolv": OpenCalibrationDataset(
        dataset_id="freesolv",
        domain="admet",
        target_unit="kcal/mol",
        n_records=642,
        url="https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="FreeSolv Small-Molecule Hydration Free Energy Database (Mobley & Guthrie, 2014)",
        citation_doi="10.1007/s10822-014-9747-x",
        landing_url="https://github.com/MobleyLab/FreeSolv",
        local_filename="freesolv_database.json",
        file_format="json",
        target_column="expt",
        domain_bounds=(-30.0, 5.0),
        expected_sha256="9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a",
        leakage_excluded_columns=(
            "calc",
            "d_calc",
            "calc_h",
            "calc_charging",
            "calc_vdw",
            "h_solv",
            "d_h_solv",
            "calc_s (cal/mol.K)",
        ),
    ),
    "lipophilicity_astrazeneca": OpenCalibrationDataset(
        dataset_id="lipophilicity_astrazeneca",
        domain="admet",
        target_unit="logD7.4",
        n_records=4200,
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        spdx_license="CC-BY-SA-3.0",
        commercial_use_ok=True,
        title="AstraZeneca ChEMBL3301361 Experimental Lipophilicity Benchmark (TDC / ADMET-AI)",
        citation_doi="10.6019/CHEMBL3301361",
        landing_url="https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/",
        local_filename="lipophilicity_astrazeneca.csv",
        file_format="csv",
        target_column="exp",
        domain_bounds=(-3.0, 6.0),
        expected_sha256="aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
        leakage_excluded_columns=(),
    ),
    "biogen_adme": OpenCalibrationDataset(
        dataset_id="biogen_adme",
        domain="admet",
        target_unit="log10(mL/min/kg)",
        n_records=3521,
        url="https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv",
        spdx_license="MIT",
        commercial_use_ok=True,
        title="Biogen Multi-Endpoint Computational ADME Public Set (Fang et al., JCIM 2023)",
        citation_doi="10.1021/acs.jcim.3c00160",
        landing_url="https://github.com/molecularinformatics/Computational-ADME",
        local_filename="biogen_adme_3521.csv",
        file_format="csv",
        target_column="LOG HLM_CLint (mL/min/kg)",
        domain_bounds=(0.0, 4.5),
        expected_sha256="2cfabc2667740c224487876c33b23124159ef43294e0f9e4d926cb6276c95a3b",
        leakage_excluded_columns=tuple(
            c for c in BIOGEN_ALL_ASSAY_COLUMNS if c != "LOG HLM_CLint (mL/min/kg)"
        ),
    ),
    "buchwald_hartwig_doyle": OpenCalibrationDataset(
        dataset_id="buchwald_hartwig_doyle",
        domain="reaction_optimization",
        target_unit="yield_pct",
        n_records=4599,
        url="https://raw.githubusercontent.com/emdgroup/baybe/main/benchmarks/data/aryl_halide/data.csv",
        spdx_license="Apache-2.0",
        commercial_use_ok=True,
        title="Doyle Buchwald-Hartwig Aryl Halide C-N Cross-Coupling HTE (Ahneman et al., Science 2018)",
        citation_doi="10.1126/science.aar5169",
        landing_url="https://github.com/emdgroup/baybe",
        local_filename="buchwald_hartwig_aryl_halide.csv",
        file_format="csv",
        target_column="yield",
        domain_bounds=(0.0, 100.0),
        expected_sha256="cce1da58d63c0c8f4245e8490b5ec58f0321c6556932842adb5f3ef11a185e12",
        leakage_excluded_columns=("plate", "row", "col", "aryl_halide_number", "additive_number"),
    ),
    "suzuki_miyaura_perera": OpenCalibrationDataset(
        dataset_id="suzuki_miyaura_perera",
        domain="reaction_optimization",
        target_unit="yield_pct",
        n_records=5760,
        url="https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Suzuki-Miyaura/random_splits/random_split_0.tsv",
        spdx_license="MIT",
        commercial_use_ok=True,
        title="Perera Automated Flow Suzuki-Miyaura Cross-Coupling HTE (Perera et al., Science 2018)",
        citation_doi="10.1126/science.aap9112",
        landing_url="https://github.com/rxn4chemistry/rxn_yields",
        local_filename="suzuki_miyaura_perera.tsv",
        file_format="tsv",
        target_column="y",
        domain_bounds=(0.0, 100.0),
        expected_sha256="d2acd54d3958a5369346177834e6c2b16388813bb28b105181a252a15c4a79a8",
        leakage_excluded_columns=("Unnamed: 0",),
    ),
    "direct_arylation_shields": OpenCalibrationDataset(
        dataset_id="direct_arylation_shields",
        domain="reaction_optimization",
        target_unit="yield_pct",
        n_records=1728,
        url="https://raw.githubusercontent.com/emdgroup/baybe/main/benchmarks/data/direct_arylation/data.csv",
        spdx_license="Apache-2.0",
        commercial_use_ok=True,
        title="Shields Pd-Catalyzed Direct C-H Arylation BO Benchmark (Shields et al., Nature 2021)",
        citation_doi="10.1038/s41586-021-03213-y",
        landing_url="https://github.com/b-shields/edbo",
        local_filename="direct_arylation_shields.csv",
        file_format="csv",
        target_column="yield",
        domain_bounds=(0.0, 100.0),
        expected_sha256="0aac33253d799c2e5f978b2962c34f9b57ba4863dae408721cabe8fa815d74ea",
        leakage_excluded_columns=("entry",),
    ),
    "bh_hte_ood_2026": OpenCalibrationDataset(
        dataset_id="bh_hte_ood_2026",
        domain="reaction_optimization",
        target_unit="yield_pct",
        n_records=27497,
        url="https://raw.githubusercontent.com/schwallergroup/bh-hte-ood/main/data/BH_HTE_Curated_Dataset_v20260219.csv",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        title="Unified Curated Buchwald-Hartwig HTE OOD Benchmark (Neves et al., Nat. Comput. Sci. 2026)",
        citation_doi="10.1038/s43588-026-01017-6",
        landing_url="https://doi.org/10.5281/zenodo.19636649",
        local_filename="bh_hte_curated_2026.csv",
        file_format="csv",
        target_column="Yield",
        domain_bounds=(0.0, 115.0),
        expected_sha256="247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333",
        leakage_excluded_columns=("labels", "DRFP", "DRFP_subst_prod", "QM"),
    ),
}

NEW_OPEN_DATASET_IDS_2026: tuple[str, ...] = tuple(
    ds_id for ds_id in OPEN_CALIBRATION_DATASETS if ds_id != "esol_delaney"
)


def list_open_datasets(
    *,
    domain: str | None = None,
    commercial_only: bool = False,
) -> list[OpenCalibrationDataset]:
    """List registered `OpenCalibrationDataset` entries filtered by domain or commercial license."""
    if domain is not None and domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {VALID_DOMAINS}")
    items = list(OPEN_CALIBRATION_DATASETS.values())
    if domain is not None:
        items = [d for d in items if d.domain == domain]
    if commercial_only:
        items = [d for d in items if d.commercial_use_ok]
    return items


def get_open_dataset(dataset_id: str) -> OpenCalibrationDataset:
    """Look up a registered `OpenCalibrationDataset` by canonical `dataset_id`."""
    key = (dataset_id or "").strip().lower()
    if key not in OPEN_CALIBRATION_DATASETS:
        valid = ", ".join(sorted(OPEN_CALIBRATION_DATASETS))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}; registered datasets: {valid}")
    return OPEN_CALIBRATION_DATASETS[key]


def default_dataset_cache_path(
    dataset_id: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Return local gitignored cache path under `data/` for `dataset_id`."""
    spec = get_open_dataset(dataset_id)
    base_dir = os.fspath(data_dir) if data_dir is not None else DEFAULT_OPEN_DATA_DIR
    fname = spec.local_filename or f"{spec.dataset_id}.{spec.file_format}"
    return os.path.normpath(os.path.join(base_dir, fname))


def fetch_open_dataset(
    dataset_id: str,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    force: bool = False,
    timeout: float = 60.0,
    verify_sha256: bool = True,
) -> str:
    """Atomically download `dataset_id` into gitignored `data/` and verify optional SHA-256."""
    spec = get_open_dataset(dataset_id)
    dest_path = default_dataset_cache_path(spec.dataset_id, data_dir=data_dir)
    expected = spec.expected_sha256 if verify_sha256 else None

    if os.path.exists(dest_path) and not force:
        if expected is not None:
            actual = compute_file_sha256(dest_path)
            if actual.lower() == expected.lower():
                return dest_path
            try:
                os.unlink(dest_path)
            except OSError:
                pass
        elif os.path.getsize(dest_path) > 0:
            return dest_path

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    part_path = f"{dest_path}.part.{os.getpid()}"
    hasher = hashlib.sha256()
    req = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "lab-decide-open-datasets/0.2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        digest = hasher.hexdigest()
        if expected is not None and digest.lower() != expected.lower():
            raise ValueError(
                f"SHA-256 mismatch for {spec.dataset_id} ({spec.url}): "
                f"expected {expected}, got {digest}"
            )
        os.replace(part_path, dest_path)
        return dest_path
    except Exception:
        if os.path.exists(part_path):
            try:
                os.unlink(part_path)
            except OSError:
                pass
        raise


def _parse_smiles_counts(smiles: str | None) -> dict[str, float]:
    """Extract deterministic, platform-invariant elemental & topological features from SMILES.

    Works identically across Python/OS environments with or without RDKit, and safely handles
    organometallic dative bonds (`<-Pd`), multi-fragment salts (`.`/`~`), and empty optional
    reagent slots without raising or producing NaNs.
    """
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        raw = ""
    else:
        raw = str(smiles).strip()
    if not raw or raw.lower() == "nan":
        return {
            "present": 0.0,
            "mw_approx": 0.0,
            "heavy_atoms": 0.0,
            "aromatic_atoms": 0.0,
            "aliphatic_carbons": 0.0,
            "ring_closures": 0.0,
            "n_nitrogen": 0.0,
            "n_oxygen": 0.0,
            "n_sulfur_phosphorus": 0.0,
            "n_ps_ligand": 0.0,
            "n_fluorine": 0.0,
            "n_chlorine": 0.0,
            "n_bromine": 0.0,
            "n_iodine": 0.0,
            "polar_unsaturations": 0.0,
            "hbd_motifs": 0.0,
            "branch_and_charge": 0.0,
        }

    elem_counts: dict[str, int] = {}
    aromatic_atoms = 0
    bracket_hbd = 0
    formal_charges = 0

    for match in _BRACKET_ATOM_RE.finditer(raw):
        body = match.group(1)
        m_el = re.search(r"([A-Z][a-z]?|[bcnops])", body)
        if not m_el:
            continue
        tok = m_el.group(1)
        if tok.islower():
            aromatic_atoms += 1
            sym = tok.upper()
        else:
            sym = tok
        if sym != "H":
            elem_counts[sym] = elem_counts.get(sym, 0) + 1
        if sym in ("N", "O") and "H" in body[m_el.end() :]:
            bracket_hbd += 1
        if "+" in body or "-" in body:
            formal_charges += 1

    outside = _BRACKET_ATOM_RE.sub("", raw)
    for m_org in _ORGANIC_ATOM_RE.finditer(outside):
        tok = m_org.group(0)
        if tok.islower():
            aromatic_atoms += 1
            sym = tok.upper()
        else:
            sym = tok
        elem_counts[sym] = elem_counts.get(sym, 0) + 1

    ring_digits = len(_RING_DIGIT_RE.findall(outside))
    ring_closures = 0.5 * float(ring_digits)

    heavy_atoms = sum(elem_counts.values())
    mw = 0.0
    for sym, cnt in elem_counts.items():
        mw += float(cnt) * _ATOMIC_MASS.get(sym, 28.0)
    # Approximate implicit hydrogens on carbon/heteroatom backbone for realistic mass scaling
    n_c = elem_counts.get("C", 0)
    n_n = elem_counts.get("N", 0)
    n_o = elem_counts.get("O", 0)
    implicit_h = max(0.0, 1.8 * float(n_c) - 0.8 * float(aromatic_atoms) + 0.5 * float(n_n + n_o))
    mw += 1.008 * implicit_h

    arom_c_est = min(n_c, aromatic_atoms)
    aliph_c = max(0, n_c - arom_c_est)

    polar_unsat = len(_POLAR_UNSAT_RE.findall(outside))
    # Heuristic hydroxyl/amine donor motifs outside brackets (e.g. terminal/branched O or N not in =O/#N)
    terminal_oh_nh = len(re.findall(r"(?:^O(?!=)|\(O\)|O$|B\(O\)O|(?<![=#])N(?![=#+]))", outside))
    hbd_total = float(bracket_hbd + terminal_oh_nh)
    branches = outside.count("(")

    n_s = elem_counts.get("S", 0)
    n_p = elem_counts.get("P", 0)
    n_b = elem_counts.get("B", 0)
    n_f = elem_counts.get("F", 0)
    n_cl = elem_counts.get("Cl", 0)
    n_br = elem_counts.get("Br", 0)
    n_i = elem_counts.get("I", 0)

    return {
        "present": 1.0,
        "mw_approx": round(mw, 4),
        "heavy_atoms": float(heavy_atoms),
        "aromatic_atoms": float(aromatic_atoms),
        "aliphatic_carbons": float(aliph_c),
        "ring_closures": float(ring_closures),
        "n_nitrogen": float(n_n),
        "n_oxygen": float(n_o),
        "n_sulfur_phosphorus": float(n_s + n_p),
        "n_ps_ligand": float(n_p + n_b + n_s),
        "n_fluorine": float(n_f),
        "n_chlorine": float(n_cl),
        "n_bromine": float(n_br),
        "n_iodine": float(n_i),
        "polar_unsaturations": float(polar_unsat),
        "hbd_motifs": hbd_total,
        "branch_and_charge": float(branches + 2 * formal_charges),
    }


def smiles_physicochem_vector(smiles: str | None) -> list[float]:
    """Return 14D deterministic molecular descriptor vector matching `MOL_PHYSICOCHEM_FEATURES`."""
    c = _parse_smiles_counts(smiles)
    return [
        c["mw_approx"],
        c["heavy_atoms"],
        c["aromatic_atoms"],
        c["aliphatic_carbons"],
        c["ring_closures"],
        c["n_nitrogen"],
        c["n_oxygen"],
        c["n_sulfur_phosphorus"],
        c["n_fluorine"],
        c["n_chlorine"],
        c["n_bromine"] + c["n_iodine"],
        c["polar_unsaturations"],
        c["hbd_motifs"],
        c["branch_and_charge"],
    ]


def reagent_slot_vector(smiles: str | None) -> list[float]:
    """Return 10D deterministic reagent-slot vector matching `REAGENT_SLOT_FEATURES`."""
    c = _parse_smiles_counts(smiles)
    return [
        c["present"],
        c["mw_approx"],
        c["aromatic_atoms"],
        c["ring_closures"],
        c["n_nitrogen"] + c["n_oxygen"],
        c["n_ps_ligand"],
        c["n_fluorine"],
        c["n_chlorine"],
        c["n_bromine"],
        c["n_iodine"],
    ]


def _detect_format(path: str, explicit_fmt: str | None = None, default_fmt: str = "csv") -> str:
    if explicit_fmt:
        return explicit_fmt.strip().lower()
    lower = path.lower()
    if lower.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".tsv"):
        return "tsv"
    if lower.endswith((".csv", ".tab")):
        return "csv"
    return default_fmt


def _read_tabular_or_jsonl(
    path: str,
    *,
    fmt: str,
    csv_usecols: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, mode="r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object per line in {path}")
                rows.append(obj)
                if max_rows is not None and len(rows) >= max_rows:
                    break
        return pd.DataFrame(rows)

    if fmt == "json":
        with open(path, mode="r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            rows = []
            for key in sorted(payload.keys()):
                rec = payload[key]
                if isinstance(rec, Mapping):
                    merged = dict(rec)
                    merged.setdefault("compound_id", str(key))
                    rows.append(merged)
                if max_rows is not None and len(rows) >= max_rows:
                    break
            return pd.DataFrame(rows)
        if isinstance(payload, list):
            sliced = payload[:max_rows] if max_rows is not None else payload
            return pd.DataFrame(sliced)
        raise ValueError(f"Unsupported JSON root type {type(payload).__name__} in {path}")

    sep = "\t" if fmt == "tsv" else ","
    # Check available columns first when csv_usecols is supplied so missing required columns
    # surface a clean schema ValueError rather than low-level pandas usecols errors.
    if csv_usecols is not None:
        header_df = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
        wanted = [c for c in csv_usecols if c in header_df.columns]
        return pd.read_csv(
            path,
            sep=sep,
            usecols=wanted if wanted else None,
            nrows=max_rows,
            encoding="utf-8-sig",
        )
    return pd.read_csv(path, sep=sep, nrows=max_rows, encoding="utf-8-sig")


def _finalize_feature_bundle(
    X_rows: list[list[float]],
    y_vals: list[float],
    keys: list[str],
    comp_ids: list[str],
    *,
    feature_names: list[str],
    target_name: str,
    raw_row_count: int,
    dropped_nonfinite: int,
    deduplicate: bool = True,
    min_rows: int = 4,
) -> DatasetBundle:
    if not X_rows:
        raise ValueError(
            f"Dataset produced 0 valid finite rows after schema cleaning (raw rows: {raw_row_count}); "
            f"at least {min_rows} rows are required."
        )

    X_arr = np.asarray(X_rows, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    row_finite = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
    extra_nonfinite = int(np.sum(~row_finite))
    if extra_nonfinite > 0:
        X_arr = X_arr[row_finite]
        y_arr = y_arr[row_finite]
        keys = [k for k, ok in zip(keys, row_finite.tolist()) if ok]
        comp_ids = [cid for cid, ok in zip(comp_ids, row_finite.tolist()) if ok]
    total_dropped = dropped_nonfinite + extra_nonfinite

    dedup_count = 0
    if deduplicate and len(keys) > 0:
        groups: dict[str, list[int]] = {}
        order_keys: list[str] = []
        for idx, k_str in enumerate(keys):
            if k_str not in groups:
                groups[k_str] = [idx]
                order_keys.append(k_str)
            else:
                groups[k_str].append(idx)
        if len(order_keys) < len(keys):
            dedup_count = len(keys) - len(order_keys)
            X_dedup = np.empty((len(order_keys), X_arr.shape[1]), dtype=np.float64)
            y_dedup = np.empty(len(order_keys), dtype=np.float64)
            id_dedup: list[str] = []
            for out_i, k_str in enumerate(order_keys):
                member_idx = groups[k_str]
                X_dedup[out_i] = np.mean(X_arr[member_idx], axis=0)
                y_dedup[out_i] = float(np.mean(y_arr[member_idx]))
                id_dedup.append(comp_ids[member_idx[0]])
            X_arr = X_dedup
            y_arr = y_dedup
            keys = order_keys
            comp_ids = id_dedup

    if len(y_arr) < min_rows:
        raise ValueError(
            f"Dataset contains only {len(y_arr)} valid finite rows after cleaning; "
            f"at least {min_rows} rows are required."
        )

    return DatasetBundle(
        X=X_arr,
        y=y_arr,
        feature_names=list(feature_names),
        target_name=target_name,
        smiles=list(keys),
        compound_ids=list(comp_ids),
        raw_row_count=raw_row_count,
        dropped_nonfinite_count=total_dropped,
        deduplicated_count=dedup_count,
    )


def _require_columns(df: pd.DataFrame, dataset_id: str, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset {dataset_id!r} is missing required schema columns {missing}; "
            f"available columns: {list(df.columns)}"
        )


def _parse_suzuki_reaction_features(rxn_smiles: str) -> list[float]:
    """Deterministically featurize a Perera Suzuki-Miyaura reaction SMILES `reactants>>product`."""
    parts = str(rxn_smiles).strip().split(">>")
    react_part = parts[0] if parts else ""
    prod_part = parts[1] if len(parts) > 1 else ""
    fragments = [f.strip() for f in react_part.split(".") if f.strip()]

    halide_frag = ""
    boron_frag = ""
    ligand_frag = ""
    solvent_base_frags: list[str] = []

    for frag in fragments:
        if "Pd" in frag or "[Pd]" in frag:
            continue
        c = _parse_smiles_counts(frag)
        if not boron_frag and re.search(r"B(?![r])", _BRACKET_ATOM_RE.sub("", frag)):
            boron_frag = frag
        elif not ligand_frag and c["n_ps_ligand"] > 0 and c["heavy_atoms"] >= 10:
            ligand_frag = frag
        elif not halide_frag and (c["n_chlorine"] + c["n_bromine"] + c["n_iodine"]) > 0:
            halide_frag = frag
        else:
            solvent_base_frags.append(frag)

    env_smiles = ".".join(solvent_base_frags)
    return (
        reagent_slot_vector(halide_frag)
        + reagent_slot_vector(boron_frag)
        + reagent_slot_vector(ligand_frag)
        + reagent_slot_vector(env_smiles)
        + reagent_slot_vector(prod_part)
    )


def _load_calisol_jsonl_or_csv(path: str, fmt: str, max_rows: int | None) -> tuple[list[dict[str, Any]], int, int]:
    """Load CALiSol-23 records from either upstream CSV or JSON/JSONL record format."""
    if fmt == "csv":
        if max_rows is None:
            with open(path, mode="r", encoding="utf-8", newline="") as fh:
                raw_total = max(0, sum(1 for _ in csv.reader(fh)) - 1)
            recs = load_calisol_csv_records(path)
            return recs, raw_total, max(0, raw_total - len(recs))

        df = pd.read_csv(path, nrows=max_rows, encoding="utf-8-sig")
        _require_columns(df, "calisol23", ["doi", "k", "T", "c", "salt"])
        solvent_cols = list(df.columns[7:])
        if not solvent_cols:
            raise ValueError(f"Malformed CALiSol-23 solvent columns in {path}")
        recs = []
        dropped = 0
        for _, row in df.iterrows():
            try:
                k_val = float(row["k"])
                t_k = float(row["T"])
                conc = float(row["c"])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (math.isfinite(k_val) and k_val >= 1e-4 and math.isfinite(conc) and conc >= 1e-3 and math.isfinite(t_k) and t_k > 0.0):
                dropped += 1
                continue
            raw_solv: dict[str, float] = {}
            bad = False
            for sc in solvent_cols:
                raw_cell = row[sc]
                v = 0.0 if pd.isna(raw_cell) else float(raw_cell)
                if not math.isfinite(v) or v < -1e-4:
                    bad = True
                    break
                if v > 1e-5:
                    raw_solv[str(sc).strip()] = v
            if bad or not raw_solv:
                dropped += 1
                continue
            tot = sum(raw_solv.values())
            if not (0.94 <= tot <= 1.06):
                dropped += 1
                continue
            norm_solv = {s: round(v / tot, 6) for s, v in sorted(raw_solv.items())}
            doi = str(row.get("doi") or "").strip()
            recs.append(
                {
                    "doi": doi,
                    "split": assign_calisol_doi_split(doi),
                    "salt": str(row.get("salt") or "").strip(),
                    "T_K": t_k,
                    "c": conc,
                    "k_mS_cm": k_val,
                    "solvents": norm_solv,
                }
            )
        return recs, len(df), dropped

    df_json = _read_tabular_or_jsonl(path, fmt=fmt, max_rows=max_rows)
    if df_json.empty:
        raise ValueError(f"No records found in {path}")
    # Support both structured JSONL (`solvents` dict + `k_mS_cm`/`T_K`) and flat CSV-mirror JSONL (`k`, `T`, `EC`, ...)
    recs = []
    dropped = 0
    for _, row in df_json.iterrows():
        row_d = row.to_dict()
        k_raw = row_d.get("k_mS_cm", row_d.get("k"))
        t_raw = row_d.get("T_K", row_d.get("T"))
        c_raw = row_d.get("c")
        salt_raw = row_d.get("salt")
        if k_raw is None or t_raw is None or c_raw is None or salt_raw is None:
            raise ValueError(
                f"Dataset 'calisol23' JSON record missing required keys ('k'/'k_mS_cm', 'T'/'T_K', 'c', 'salt'): {list(row_d)}"
            )
        try:
            k_val = float(k_raw)
            t_k = float(t_raw)
            conc = float(c_raw)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (math.isfinite(k_val) and k_val >= 1e-4 and math.isfinite(conc) and conc >= 1e-3 and math.isfinite(t_k) and t_k > 0.0):
            dropped += 1
            continue
        if isinstance(row_d.get("solvents"), Mapping):
            raw_solv = {str(s).strip(): float(v) for s, v in row_d["solvents"].items() if float(v) > 1e-5}
        else:
            skip_keys = {"doi", "k", "k_mS_cm", "T", "T_K", "c", "salt", "c units", "solvent ratio type", "split", "compound_id"}
            raw_solv = {}
            for k_col, v_col in row_d.items():
                if k_col in skip_keys or v_col is None or (isinstance(v_col, float) and math.isnan(v_col)):
                    continue
                fv = float(v_col)
                if fv > 1e-5:
                    raw_solv[str(k_col).strip()] = fv
        tot = sum(raw_solv.values())
        if not raw_solv or not (0.94 <= tot <= 1.06):
            dropped += 1
            continue
        norm_solv = {s: round(v / tot, 6) for s, v in sorted(raw_solv.items())}
        doi = str(row_d.get("doi") or "").strip()
        recs.append(
            {
                "doi": doi,
                "split": assign_calisol_doi_split(doi),
                "salt": str(salt_raw).strip(),
                "T_K": t_k,
                "c": conc,
                "k_mS_cm": k_val,
                "solvents": norm_solv,
            }
        )
    return recs, len(df_json), dropped


def load_open_dataset(
    dataset_id: str,
    path: str | os.PathLike[str] | None = None,
    *,
    fmt: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
    deduplicate: bool = True,
    offline: bool = False,
    min_rows: int = 4,
) -> DatasetBundle:
    """Load any registered `OpenCalibrationDataset` from CSV/TSV/JSON/JSONL into a `DatasetBundle`.

    Strictly excludes pre-fitted predictions, sibling assay outcomes, and plate-index columns
    listed in `spec.leakage_excluded_columns` so downstream conformal calibration is leakage-free.
    """
    spec = get_open_dataset(dataset_id)
    if path is None:
        resolved = default_dataset_cache_path(spec.dataset_id)
        if not os.path.exists(resolved):
            if offline:
                raise FileNotFoundError(
                    f"Dataset {spec.dataset_id!r} is not cached at {resolved} and offline=True."
                )
            resolved = fetch_open_dataset(spec.dataset_id)
    else:
        resolved = os.fspath(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Dataset file not found: {resolved}")

    eff_fmt = _detect_format(resolved, explicit_fmt=fmt, default_fmt=spec.file_format)
    tgt_col = target_column or spec.target_column

    if spec.dataset_id == "esol_delaney":
        df = _read_tabular_or_jsonl(resolved, fmt=eff_fmt, max_rows=max_rows)
        return validate_and_clean_dataframe(
            df,
            features=ESOL_FEATURES,
            target=tgt_col,
            deduplicate_smiles=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "aqsoldb":
        want_cols = ["ID", "SMILES", tgt_col, *AQSOLDB_FEATURES]
        df = _read_tabular_or_jsonl(resolved, fmt=eff_fmt, csv_usecols=want_cols, max_rows=max_rows)
        _require_columns(df, "aqsoldb", ["SMILES", tgt_col, *AQSOLDB_FEATURES])
        if "ID" in df.columns and "Compound ID" not in df.columns:
            df = df.rename(columns={"ID": "Compound ID"})
        df = df.rename(columns={"SMILES": "smiles"})
        return validate_and_clean_dataframe(
            df,
            features=AQSOLDB_FEATURES,
            target=tgt_col,
            deduplicate_smiles=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "calisol23":
        recs, raw_count, dropped = _load_calisol_jsonl_or_csv(resolved, eff_fmt, max_rows)
        if not recs:
            raise ValueError(f"Dataset 'calisol23' contained 0 valid records in {resolved}")
        X_cal, y_cal, feat_names = featurize_calisol_records(recs)
        keys = [
            f"{r['salt']}|T={r['T_K']:.2f}|c={r['c']:.4f}|"
            + ",".join(f"{s}:{v:.4f}" for s, v in sorted(r["solvents"].items()))
            for r in recs
        ]
        cids = [f"{r['doi'] or 'calisol'}#{i:05d}" for i, r in enumerate(recs)]
        return _finalize_feature_bundle(
            X_cal.tolist(),
            y_cal.tolist(),
            keys,
            cids,
            feature_names=feat_names,
            target_name="log1p(k_mS_cm)",
            raw_row_count=raw_count,
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "bigsoldb_v2_2":
        req_cols = ["SMILES_Solute", "Temperature_K", "SMILES_Solvent", tgt_col]
        opt_cols = ["PubChem_CID", "Compound_Name"]
        df = _read_tabular_or_jsonl(
            resolved, fmt=eff_fmt, csv_usecols=req_cols + opt_cols, max_rows=max_rows
        )
        _require_columns(df, "bigsoldb_v2_2", req_cols)
        feat_names = (
            ["Temperature_K", "inv_T_1000"]
            + [f"solute_{f}" for f in MOL_PHYSICOCHEM_FEATURES]
            + [f"solvent_{f}" for f in REAGENT_SLOT_FEATURES]
        )
        X_rows: list[list[float]] = []
        y_vals: list[float] = []
        keys: list[str] = []
        cids: list[str] = []
        dropped = 0
        for idx, row in df.iterrows():
            solute_smi = str(row["SMILES_Solute"]).strip() if pd.notna(row["SMILES_Solute"]) else ""
            solvent_smi = str(row["SMILES_Solvent"]).strip() if pd.notna(row["SMILES_Solvent"]) else ""
            try:
                t_k = float(row["Temperature_K"])
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (solute_smi and solvent_smi and math.isfinite(t_k) and t_k > 50.0 and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(
                [t_k, 1000.0 / t_k]
                + smiles_physicochem_vector(solute_smi)
                + reagent_slot_vector(solvent_smi)
            )
            y_vals.append(y_val)
            keys.append(f"{solute_smi}|{solvent_smi}|T={t_k:.2f}")
            cid_val = row.get("Compound_Name") if "Compound_Name" in df.columns else None
            cids.append(str(cid_val).strip() if pd.notna(cid_val) and str(cid_val).strip() else f"bigsol-{idx}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=feat_names,
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "freesolv":
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=["compound_id", "iupac", "smiles", tgt_col],
            max_rows=max_rows,
        )
        _require_columns(df, "freesolv", ["smiles", tgt_col])
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            smi = str(row["smiles"]).strip() if pd.notna(row["smiles"]) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (smi and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(smiles_physicochem_vector(smi))
            y_vals.append(y_val)
            keys.append(smi)
            raw_id = row.get("compound_id") or row.get("iupac")
            cids.append(str(raw_id).strip() if pd.notna(raw_id) and str(raw_id).strip() else f"freesolv-{idx}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=list(MOL_PHYSICOCHEM_FEATURES),
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "lipophilicity_astrazeneca":
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=["CMPD_CHEMBLID", "smiles", tgt_col],
            max_rows=max_rows,
        )
        _require_columns(df, "lipophilicity_astrazeneca", ["smiles", tgt_col])
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            smi = str(row["smiles"]).strip() if pd.notna(row["smiles"]) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (smi and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(smiles_physicochem_vector(smi))
            y_vals.append(y_val)
            keys.append(smi)
            raw_id = row.get("CMPD_CHEMBLID")
            cids.append(str(raw_id).strip() if pd.notna(raw_id) and str(raw_id).strip() else f"chembl-{idx}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=list(MOL_PHYSICOCHEM_FEATURES),
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "biogen_adme":
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=["Internal ID", "SMILES", tgt_col],
            max_rows=max_rows,
        )
        _require_columns(df, "biogen_adme", ["SMILES", tgt_col])
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            smi = str(row["SMILES"]).strip() if pd.notna(row["SMILES"]) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (smi and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(smiles_physicochem_vector(smi))
            y_vals.append(y_val)
            keys.append(smi)
            raw_id = row.get("Internal ID")
            cids.append(str(raw_id).strip() if pd.notna(raw_id) and str(raw_id).strip() else f"biogen-{idx}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=list(MOL_PHYSICOCHEM_FEATURES),
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "buchwald_hartwig_doyle":
        req_cols = ["base_smiles", "ligand_smiles", "aryl_halide_smiles", tgt_col]
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=[*req_cols, "additive_smiles"],
            max_rows=max_rows,
        )
        _require_columns(df, "buchwald_hartwig_doyle", req_cols)
        slot_prefixes = ("aryl_halide", "additive", "base", "ligand")
        feat_names = [f"{p}_{f}" for p in slot_prefixes for f in REAGENT_SLOT_FEATURES]
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            ah_s = str(row["aryl_halide_smiles"]).strip() if pd.notna(row["aryl_halide_smiles"]) else ""
            bs_s = str(row["base_smiles"]).strip() if pd.notna(row["base_smiles"]) else ""
            lg_s = str(row["ligand_smiles"]).strip() if pd.notna(row["ligand_smiles"]) else ""
            ad_raw = row.get("additive_smiles")
            ad_s = str(ad_raw).strip() if pd.notna(ad_raw) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (ah_s and bs_s and lg_s and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(
                reagent_slot_vector(ah_s)
                + reagent_slot_vector(ad_s)
                + reagent_slot_vector(bs_s)
                + reagent_slot_vector(lg_s)
            )
            y_vals.append(y_val)
            keys.append(f"{ah_s}.{ad_s}.{bs_s}.{lg_s}")
            cids.append(f"bh-doyle-{idx:04d}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=feat_names,
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "suzuki_miyaura_perera":
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=["rxn", tgt_col],
            max_rows=max_rows,
        )
        _require_columns(df, "suzuki_miyaura_perera", ["rxn", tgt_col])
        slot_prefixes = ("electrophile", "nucleophile", "ligand", "medium", "product")
        feat_names = [f"{p}_{f}" for p in slot_prefixes for f in REAGENT_SLOT_FEATURES]
        X_rows = []
        raw_y: list[float] = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            rxn_s = str(row["rxn"]).strip() if pd.notna(row["rxn"]) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (rxn_s and ">>" in rxn_s and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(_parse_suzuki_reaction_features(rxn_s))
            raw_y.append(y_val)
            keys.append(rxn_s)
            cids.append(f"suzuki-{idx:04d}")
        # Upstream rxn_yields TSV stores fractional yield in [0, 1]; convert to percentage [0, 100]
        if raw_y and max(raw_y) <= 1.05 and min(raw_y) >= -0.05:
            y_vals = [100.0 * v for v in raw_y]
        else:
            y_vals = raw_y
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=feat_names,
            target_name="yield_pct",
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "direct_arylation_shields":
        req_cols = ["Base_SMILES", "Ligand_SMILES", "Solvent_SMILES", "Concentration", "Temp_C", tgt_col]
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=req_cols,
            max_rows=max_rows,
        )
        _require_columns(df, "direct_arylation_shields", req_cols)
        slot_prefixes = ("base", "ligand", "solvent")
        feat_names = (
            ["Concentration", "Temp_C", "inv_T_1000"]
            + [f"{p}_{f}" for p in slot_prefixes for f in REAGENT_SLOT_FEATURES]
        )
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            bs_s = str(row["Base_SMILES"]).strip() if pd.notna(row["Base_SMILES"]) else ""
            lg_s = str(row["Ligand_SMILES"]).strip() if pd.notna(row["Ligand_SMILES"]) else ""
            sv_s = str(row["Solvent_SMILES"]).strip() if pd.notna(row["Solvent_SMILES"]) else ""
            try:
                conc = float(row["Concentration"])
                temp_c = float(row["Temp_C"])
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            temp_k = temp_c + 273.15
            if not (
                bs_s
                and lg_s
                and sv_s
                and math.isfinite(conc)
                and conc > 0.0
                and math.isfinite(temp_c)
                and temp_k > 50.0
                and math.isfinite(y_val)
            ):
                dropped += 1
                continue
            X_rows.append(
                [conc, temp_c, 1000.0 / temp_k]
                + reagent_slot_vector(bs_s)
                + reagent_slot_vector(lg_s)
                + reagent_slot_vector(sv_s)
            )
            y_vals.append(y_val)
            keys.append(f"{bs_s}|{lg_s}|{sv_s}|c={conc:.4f}|T={temp_c:.2f}")
            cids.append(f"shields-da-{idx:04d}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=feat_names,
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    if spec.dataset_id == "bh_hte_ood_2026":
        req_cols = ["Aryl SMILES", "Amine SMILES", "Catalyst SMILES", "Base SMILES", "Solvent SMILES", tgt_col]
        df = _read_tabular_or_jsonl(
            resolved,
            fmt=eff_fmt,
            csv_usecols=[*req_cols, "Source"],
            max_rows=max_rows,
        )
        _require_columns(df, "bh_hte_ood_2026", req_cols)
        slot_prefixes = ("aryl", "amine", "catalyst", "base", "solvent")
        feat_names = [f"{p}_{f}" for p in slot_prefixes for f in REAGENT_SLOT_FEATURES]
        X_rows = []
        y_vals = []
        keys = []
        cids = []
        dropped = 0
        for idx, row in df.iterrows():
            ar_s = str(row["Aryl SMILES"]).strip() if pd.notna(row["Aryl SMILES"]) else ""
            am_s = str(row["Amine SMILES"]).strip() if pd.notna(row["Amine SMILES"]) else ""
            cat_s = str(row["Catalyst SMILES"]).strip() if pd.notna(row["Catalyst SMILES"]) else ""
            bs_s = str(row["Base SMILES"]).strip() if pd.notna(row["Base SMILES"]) else ""
            sv_s = str(row["Solvent SMILES"]).strip() if pd.notna(row["Solvent SMILES"]) else ""
            try:
                y_val = float(row[tgt_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (ar_s and am_s and cat_s and bs_s and sv_s and math.isfinite(y_val)):
                dropped += 1
                continue
            X_rows.append(
                reagent_slot_vector(ar_s)
                + reagent_slot_vector(am_s)
                + reagent_slot_vector(cat_s)
                + reagent_slot_vector(bs_s)
                + reagent_slot_vector(sv_s)
            )
            y_vals.append(y_val)
            keys.append(f"{ar_s}|{am_s}|{cat_s}|{bs_s}|{sv_s}")
            src = str(row.get("Source") or "bh2026").strip()
            cids.append(f"{src}-{idx:05d}")
        return _finalize_feature_bundle(
            X_rows,
            y_vals,
            keys,
            cids,
            feature_names=feat_names,
            target_name=tgt_col,
            raw_row_count=len(df),
            dropped_nonfinite=dropped,
            deduplicate=deduplicate,
            min_rows=min_rows,
        )

    raise RuntimeError(f"Unhandled schema loader for {spec.dataset_id!r}")


def calibrate_open_dataset(
    dataset_id: str,
    path: str | os.PathLike[str] | None = None,
    *,
    fmt: str | None = None,
    alpha: float = 0.10,
    k: int = 5,
    trust_pct: float = 80.0,
    nonconformity: str = "normalized",
    split_strategy: str = "random",
    seed: int = 0,
    fracs: tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15),
    max_rows: int | None = None,
    offline: bool = False,
) -> tuple[ConformalTrustModel, DatasetBundle, SplitIndices, dict[str, Any]]:
    """Load `dataset_id`, fit `ConformalTrustModel`, and evaluate trust-region calibration + BO."""
    spec = get_open_dataset(dataset_id)
    bundle = load_open_dataset(
        spec.dataset_id,
        path=path,
        fmt=fmt,
        max_rows=max_rows,
        offline=offline,
    )
    splits = partition_splits(
        len(bundle.y),
        smiles=bundle.smiles if split_strategy == "scaffold" else None,
        strategy=split_strategy,
        seed=seed,
        fracs=fracs,
    )
    advisor = fit_and_calibrate(
        bundle.X[splits.train_idx],
        bundle.y[splits.train_idx],
        bundle.X[splits.cal_idx],
        bundle.y[splits.cal_idx],
        alpha=alpha,
        k=k,
        trust_pct=trust_pct,
        seed=seed,
        nonconformity=nonconformity,
        domain_bounds=spec.domain_bounds,
        feature_names=bundle.feature_names,
    )

    X_te = bundle.X[splits.test_idx]
    y_te = bundle.y[splits.test_idx]
    res = advisor.predict(X_te, use_mondrian=True)
    inside = np.asarray(res["trusted"], dtype=bool)
    c_in, w_in, e_in = summarize(y_te, res["lo"], res["hi"], inside, pred=res["pred"])
    c_out, w_out, e_out = summarize(y_te, res["lo"], res["hi"], ~inside, pred=res["pred"])
    c_all, w_all, e_all = summarize(
        y_te, res["lo"], res["hi"], np.ones(len(y_te), dtype=bool), pred=res["pred"]
    )
    wink_in = winkler_interval_score(y_te, res["lo"], res["hi"], alpha=alpha, mask=inside)

    maximize = spec.domain != "admet" or spec.dataset_id != "biogen_adme"
    bo_metrics = benchmark_trust_region_bo(
        advisor,
        bundle.X[splits.pool_idx],
        bundle.y[splits.pool_idx],
        batch_size=min(8, len(splits.pool_idx)),
        maximize=maximize,
    )

    metrics: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "domain": spec.domain,
        "target_unit": spec.target_unit,
        "spdx_license": spec.spdx_license,
        "commercial_use_ok": spec.commercial_use_ok,
        "n_clean": len(bundle.y),
        "n_train": int(splits.train_idx.size),
        "n_pool": int(splits.pool_idx.size),
        "n_cal": int(splits.cal_idx.size),
        "n_test": int(splits.test_idx.size),
        "n_features": int(bundle.X.shape[1]),
        "alpha": float(alpha),
        "trust_pct": float(trust_pct),
        "threshold": float(advisor.threshold),
        "q_in": float(advisor.q_in),
        "answered_fraction": float(np.mean(inside)),
        "cov_in": c_in,
        "cov_out": c_out,
        "cov_all": c_all,
        "mae_in": e_in,
        "mae_out": e_out,
        "mae_all": e_all,
        "width_in": w_in,
        "width_out": w_out,
        "width_all": w_all,
        "winkler_in": wink_in,
        "constrained_bo": bo_metrics,
    }
    return advisor, bundle, splits, metrics


def make_schema_sample_records(
    dataset_id: str,
    *,
    n: int = 120,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate deterministic, schema-exact fixture records (with realistic OOD tail) for testing.

    Every record mirrors the exact upstream column names—including leakage columns that
    `load_open_dataset` must ignore—so offline unit tests validate schema parsing, anti-leakage
    invariants, finite-sample conformal coverage, and OOD abstention across all 11 datasets.
    """
    spec = get_open_dataset(dataset_id)
    if n < 16:
        raise ValueError(f"Sample fixture requires n >= 16 for 4-way splits, got {n}")
    rng = np.random.default_rng(seed)

    def _make_organic_smiles(idx: int, is_ood: bool) -> str:
        chain_len = (idx % 6) + (9 if is_ood else 1)
        chain = "C" * chain_len
        if is_ood:
            ood_motifs = (
                "c1cc(Cl)ccc1c2ccc(Br)cc2",
                "c1cc(F)c(F)cc1S(=O)(=O)C(F)(F)F",
                "c1ccc2c(c1)ccc3ccccc23",
                "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1",
            )
            rings = ood_motifs[idx % len(ood_motifs)]
        else:
            rings = "c1ccccc1" if idx % 2 == 0 else "C1CCCCC1"
        polar = "C(=O)O" if idx % 3 == 0 else ("O" * ((idx % 2) + 1) if not is_ood else "Cl")
        hal = "C(F)(F)F" if (is_ood and idx % 2 == 0) else ("F" if idx % 5 == 0 else "")
        tail = "N" if idx % 4 == 0 else ""
        # Bijection over (idx % 6, idx // 6) guarantees every idx yields a unique SMILES
        branch = "(C)" * (idx // 6)
        return f"{chain}{branch}{rings}{polar}{hal}{tail}"

    records: list[dict[str, Any]] = []

    if spec.dataset_id == "esol_delaney":
        for i in range(n):
            ood = bool(rng.random() < 0.20)
            rings = float(rng.integers(4, 7) if ood else rng.integers(0, 3))
            mw = float((390.0 if ood else 95.0) + 42.0 * rings + rng.uniform(10.0, 120.0))
            hbd = float(rng.integers(0, 5))
            rot = float(rng.integers(7, 13) if ood else rng.integers(0, 6))
            tpsa = float(18.5 * hbd + rng.uniform(5.0, 70.0))
            min_deg = 2.0 if i % 10 == 0 else 1.0
            noise = float(rng.normal(0.0, 1.15 if ood else 0.20))
            y_val = float(np.clip(0.8 - 0.012 * mw - 0.35 * rings + 0.014 * tpsa - 0.04 * rot + noise, -11.5, 1.55))
            records.append(
                {
                    "Compound ID": f"ESOL-FIX-{i:04d}",
                    "ESOL predicted log solubility in mols per litre": round(y_val + 0.1, 4),
                    "Minimum Degree": min_deg,
                    "Molecular Weight": round(mw, 3),
                    "Number of H-Bond Donors": hbd,
                    "Number of Rings": rings,
                    "Number of Rotatable Bonds": rot,
                    "Polar Surface Area": round(tpsa, 2),
                    ESOL_TARGET: round(y_val, 4),
                    "smiles": _make_organic_smiles(i, ood),
                }
            )
        return records

    if spec.dataset_id == "aqsoldb":
        for i in range(n):
            ood = bool(rng.random() < 0.20)
            smi = _make_organic_smiles(i, ood)
            rings = float(rng.integers(4, 7) if ood else rng.integers(0, 3))
            arom_rings = float(min(rings, rng.integers(2, 5) if ood else rng.integers(0, 2)))
            sat_rings = max(0.0, rings - arom_rings)
            mw = float((420.0 if ood else 110.0) + 38.0 * rings + rng.uniform(5.0, 110.0))
            logp = float((4.5 if ood else 1.2) + 0.55 * rings + rng.uniform(-0.8, 1.2))
            hba = float(rng.integers(0, 7))
            hbd = float(rng.integers(0, 4))
            tpsa = float(14.0 * hba + 20.0 * hbd + rng.uniform(0.0, 25.0))
            rot = float(rng.integers(8, 15) if ood else rng.integers(0, 6))
            heavy = round(mw / 13.5)
            noise = float(rng.normal(0.0, 1.25 if ood else 0.22))
            sol = float(np.clip(0.65 - 0.75 * logp - 0.006 * mw + 0.012 * tpsa + noise, -13.0, 1.6))
            records.append(
                {
                    "ID": f"A-{i + 1}",
                    "Name": f"Compound-{i + 1}",
                    "InChI": f"InChI=1S/C{int(heavy)}H{int(heavy * 2)}",
                    "InChIKey": f"KEY{i:011d}-UHFFFAOYSA-N",
                    "SMILES": smi,
                    "Solubility": round(sol, 5),
                    "SD": 0.05,
                    "Ocurrences": 1,
                    "Group": "G1",
                    "MolWt": round(mw, 3),
                    "MolLogP": round(logp, 4),
                    "MolMR": round(0.26 * mw, 3),
                    "HeavyAtomCount": float(heavy),
                    "NumHAcceptors": hba,
                    "NumHDonors": hbd,
                    "NumHeteroatoms": hba + hbd,
                    "NumRotatableBonds": rot,
                    "NumValenceElectrons": float(heavy * 4 + 4),
                    "NumAromaticRings": arom_rings,
                    "NumSaturatedRings": sat_rings,
                    "NumAliphaticRings": sat_rings,
                    "RingCount": rings,
                    "TPSA": round(tpsa, 2),
                    "LabuteASA": round(0.42 * mw, 3),
                    "BalabanJ": round(float(rng.uniform(1.4, 3.2)), 4),
                    "BertzCT": round(18.0 * heavy + 45.0 * rings, 3),
                }
            )
        return records

    if spec.dataset_id == "bigsoldb_v2_2":
        solvents = [("ethanol", "CCO"), ("water", "O"), ("acetonitrile", "CC#N"), ("ethyl acetate", "CCOC(C)=O")]
        for i in range(n):
            ood = bool(rng.random() < 0.20)
            solute = _make_organic_smiles(i, ood)
            solv_name, solv_smi = ("chloroform", "ClC(Cl)Cl") if ood else solvents[i % len(solvents)]
            temp_k = float(rng.uniform(355.0, 390.0) if ood else rng.uniform(278.15, 333.15))
            c = _parse_smiles_counts(solute)
            sc = _parse_smiles_counts(solv_smi)
            noise = float(rng.normal(0.0, 1.15 if ood else 0.18))
            logs = float(
                np.clip(
                    2.2
                    - 0.011 * c["mw_approx"]
                    + 0.22 * c["n_oxygen"]
                    - 420.0 / temp_k
                    + 0.15 * sc["n_oxygen"]
                    + noise,
                    -10.0,
                    1.8,
                )
            )
            mol_l = 10.0 ** logs
            records.append(
                {
                    "SMILES_Solute": solute,
                    "Temperature_K": round(temp_k, 2),
                    "Solvent": solv_name,
                    "SMILES_Solvent": solv_smi,
                    "Solubility(mole_fraction)": round(min(0.99, mol_l * 0.05), 6),
                    "Solubility(mol/L)": round(mol_l, 6),
                    "LogS(mol/L)": round(logs, 5),
                    "LogS(mole_fraction)": round(logs - 1.3, 5),
                    "Compound_Name": f"Solute-{i:04d}",
                    "CAS": f"100-{i % 90 + 10}-{i % 9}",
                    "PubChem_CID": 10000 + i,
                    "FDA_Approved": "No",
                    "Source": "10.1038/s41597-025-05559-8",
                    "solid_form_label": "",
                }
            )
        return records

    if spec.dataset_id == "calisol23":
        all_solv_cols = list(COMMON_SOLVENTS) + ["THF", "AN", "DMSO", "Sulfolane"]
        ood_salts = ("NaPF6", "LiClO4", "LiFSI", "KPF6")
        ood_solvs = ("THF", "AN", "DMSO", "Sulfolane")
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            salt = ood_salts[i % len(ood_salts)] if ood else str(COMMON_SALTS[i % 3])
            t_k = float(rng.uniform(335.0, 385.0) if ood else rng.uniform(275.0, 315.0))
            conc = float(rng.uniform(2.3, 4.2) if ood else rng.uniform(0.6, 1.5))
            ec_w = round(float(rng.uniform(0.25, 0.75)), 4)
            base_k = 11.2 * math.exp(-((conc - 1.05) ** 2) / 0.9) * math.exp(0.011 * (t_k - 298.15))
            ood_shift = (8.5 if (i % 2 == 0) else -3.5) if ood else 0.0
            k_val = max(0.05, base_k + ood_shift + float(rng.normal(0.0, 4.5 if ood else 0.18)))
            row_d: dict[str, Any] = {
                "doi": f"10.1149/calisol.fix.{i:04d}",
                "k": round(k_val, 5),
                "T": round(t_k, 2),
                "c": round(conc, 4),
                "salt": salt,
                "c units": "mol/kg",
                "solvent ratio type": "w",
            }
            for sc in all_solv_cols:
                row_d[sc] = 0.0
            if ood:
                row_d[ood_solvs[i % len(ood_solvs)]] = 1.0
            else:
                row_d["EC"] = ec_w
                row_d["DMC"] = round(1.0 - ec_w, 4)
            records.append(row_d)
        return records

    if spec.dataset_id == "freesolv":
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            smi = _make_organic_smiles(i, ood)
            c = _parse_smiles_counts(smi)
            ood_shift = (-5.5 * math.sin(float(i))) if ood else 0.0
            noise = float(rng.normal(0.0, 3.2 if ood else 0.25))
            expt = float(
                np.clip(
                    1.2
                    - 2.6 * c["n_oxygen"]
                    - 2.1 * c["n_nitrogen"]
                    - 1.4 * c["polar_unsaturations"]
                    + 0.015 * c["aliphatic_carbons"]
                    + ood_shift
                    + noise,
                    -25.0,
                    3.5,
                )
            )
            records.append(
                {
                    "compound_id": f"mobley_{100000 + i}",
                    "iupac": f"synthetic-solute-{i}",
                    "smiles": smi,
                    "expt": round(expt, 3),
                    "d_expt": 0.6,
                    "calc": round(expt + 0.4, 3),
                    "d_calc": 0.05,
                }
            )
        return records

    if spec.dataset_id == "lipophilicity_astrazeneca":
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            smi = _make_organic_smiles(i, ood)
            c = _parse_smiles_counts(smi)
            ood_shift = (2.4 * (1.0 if i % 2 == 0 else -1.0)) if ood else 0.0
            noise = float(rng.normal(0.0, 1.65 if ood else 0.15))
            logd = float(
                np.clip(
                    -0.4
                    + 0.011 * c["mw_approx"]
                    + 0.14 * c["aromatic_atoms"]
                    - 0.65 * c["n_oxygen"]
                    - 0.45 * c["n_nitrogen"]
                    + ood_shift
                    + noise,
                    -2.8,
                    5.8,
                )
            )
            records.append(
                {
                    "CMPD_CHEMBLID": f"CHEMBL{500000 + i}",
                    "exp": round(logd, 3),
                    "smiles": smi,
                }
            )
        return records

    if spec.dataset_id == "biogen_adme":
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            smi = _make_organic_smiles(i, ood)
            c = _parse_smiles_counts(smi)
            ood_shift = (1.35 * (1.0 if i % 2 == 0 else -1.0)) if ood else 0.0
            noise = float(rng.normal(0.0, 1.05 if ood else 0.10))
            hlm = float(
                np.clip(
                    1.25
                    + 0.0035 * c["mw_approx"]
                    + 0.06 * c["aliphatic_carbons"]
                    - 0.22 * c["n_oxygen"]
                    + ood_shift
                    + noise,
                    0.2,
                    4.2,
                )
            )
            records.append(
                {
                    "Internal ID": f"Mol{i + 1}",
                    "Vendor ID": f"{300000000 + i}",
                    "SMILES": smi,
                    "CollectionName": "emolecules",
                    "LOG HLM_CLint (mL/min/kg)": round(hlm, 4),
                    "LOG MDR1-MDCK ER (B-A/A-B)": round(1.1 + 0.2 * (i % 4), 4),
                    "LOG SOLUBILITY PH 6.8 (ug/mL)": round(1.5 - 0.3 * hlm, 4),
                    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)": 0.85,
                    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)": 0.75,
                    "LOG RLM_CLint (mL/min/kg)": round(hlm + 0.3, 4),
                }
            )
        return records

    if spec.dataset_id == "buchwald_hartwig_doyle":
        bases = [
            ("P2Et", "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC"),
            ("BTMG", "CN(C)C(=NC(C)(C)C)N(C)C"),
            ("MTBD", "CN1CCCN2CCCN=C12"),
        ]
        ligands = [
            ("XPhos", "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1"),
            ("tBuXPhos", "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C(C)(C)C)C(C)(C)C)c(C(C)C)c1"),
            ("AdBrettPhos", "COc1ccc(OC)c(-c2c(C(C)C)cc(C(C)C)cc2C(C)C)c1P(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2"),
        ]
        halides = [
            ("ArCl-CF3", "FC(F)(F)c1ccc(Cl)cc1", 32.0),
            ("ArBr-CF3", "FC(F)(F)c1ccc(Br)cc1", 68.0),
            ("ArI-CF3", "FC(F)(F)c1ccc(I)cc1", 84.0),
            ("ArBr-OMe", "COc1ccc(Br)cc1", 52.0),
            ("ArCl-CN", "N#Cc1ccc(Cl)cc1", 44.0),
        ]
        ood_additives = (
            "CCCCCCCCc1cc(on1)c2ccc(Br)cc2",
            "FC(F)(F)c1cc(Cl)cc(I)c1O",
            "CCCCCCCCCCCCc1ccc2ccccc2c1",
            "N#Cc1c(F)c(F)c(Br)c(F)c1F",
        )
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            b_name, b_smi = bases[i % len(bases)]
            l_name, l_smi = ligands[(i // 3) % len(ligands)]
            h_name, h_smi, h_base_y = halides[(i // 2) % len(halides)]
            extra = "C" * (i // 15)
            h_smi_u = f"{extra}{h_smi}"
            if ood:
                add_smi = f"{'C' * (i % 7 + 1)}{ood_additives[i % len(ood_additives)]}"
                h_smi_u = f"{'C' * (i % 5 + 4)}{h_smi}"
            else:
                add_smi = "" if i % 4 == 0 else f"Cc1cc({'C' * (i % 3 + 1)})on1"
            b_bonus = 9.0 if b_name == "BTMG" else (4.0 if b_name == "P2Et" else -5.0)
            ood_shift = (-32.0 if i % 2 == 0 else 24.0) if ood else (3.5 if add_smi else 0.0)
            noise = float(rng.normal(0.0, 22.0 if ood else 2.2))
            yld = float(np.clip(h_base_y + b_bonus + ood_shift + noise, 0.0, 99.9))
            records.append(
                {
                    "plate": (i // 48) + 1,
                    "row": (i % 8) + 1,
                    "col": (i % 12) + 1,
                    "base": b_name,
                    "base_cas_number": "00-00-0",
                    "base_smiles": b_smi,
                    "ligand": l_name,
                    "ligand_cas_number": "00-00-1",
                    "ligand_smiles": l_smi,
                    "aryl_halide_number": float((i % 5) + 1),
                    "aryl_halide": h_name,
                    "aryl_halide_smiles": h_smi_u,
                    "additive_number": float(i % 6),
                    "additive": f"Add-{i % 6}" if add_smi else "",
                    "additive_smiles": add_smi,
                    "product_smiles": "Cc1ccc(Nc2ccccc2)cc1",
                    "yield": round(yld, 3),
                }
            )
        return records

    if spec.dataset_id == "suzuki_miyaura_perera":
        halides_sz = [
            ("Clc1ccc2ncccc2c1", 0.32),
            ("Brc1ccc2ncccc2c1", 0.68),
            ("Ic1ccc2ncccc2c1", 0.85),
        ]
        ligands_sz = [
            ("CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1", 0.08),
            ("C1CCC(P(C2CCCCC2)C2CCCCC2)CC1", 0.03),
            ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", -0.09),
        ]
        bases_sz = [("CCN(CC)CC", 0.04), ("[Na+].[OH-]", 0.06), ("O=C([O-])[O-].[Cs+].[Cs+]", 0.09)]
        ood_borons = (
            "c1c(F)c(F)c(B(O)O)c(F)c1F",
            "CCCCCCCCc1ccc(S(=O)(=O)C)c(B(O)O)c1",
            "FC(F)(F)c1cc(N#C)ccc1B(O)O",
            "c1ccc2c(c1)ccc1ccccc21B(O)O",
        )
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            h_smi, h_y = halides_sz[i % len(halides_sz)]
            l_smi, l_y = ligands_sz[(i // 3) % len(ligands_sz)]
            b_smi, b_y = bases_sz[(i // 5) % len(bases_sz)]
            boron = (
                f"{'C' * (i % 7 + 2)}{ood_borons[i % len(ood_borons)]}"
                if ood
                else f"{'C' * (i % 4 + 1)}c1ccc2c(cnn2C2CCCCO2)c1B(O)O"
            )
            solv = ("ClC(Cl)Cl" if i % 2 == 0 else "CS(C)=O.FC(F)(F)C(=O)O") if ood else ("CC#N.O" if i % 2 == 0 else "C1CCOC1.O")
            prod = f"{'C' * (i // 12 + 1)}c1ccc2c(cnn2C2CCCCO2)c1-c1ccc2ncccc2c1"
            rxn = f"{h_smi}.CC(=O)O~[Pd].{l_smi}.{b_smi}.{solv}.{boron}>>{prod}"
            ood_shift = (-0.36 if i % 2 == 0 else 0.25) if ood else 0.0
            noise = float(rng.normal(0.0, 0.24 if ood else 0.02))
            y_frac = float(np.clip(h_y + l_y + b_y + ood_shift + noise, 0.0, 0.995))
            records.append(
                {
                    "Unnamed: 0": i,
                    "rxn": rxn,
                    "y": round(y_frac, 5),
                }
            )
        return records

    if spec.dataset_id == "direct_arylation_shields":
        bases_da = [("Potassium acetate", "O=C([O-])C.[K+]"), ("Cesium pivalate", "CC(C)(C)C(=O)[O-].[Cs+]")]
        ligands_da = [
            ("BrettPhos", "COc1ccc(OC)c(-c2c(C(C)C)cc(C(C)C)cc2C(C)C)c1P(C1CCCCC1)C1CCCCC1", 18.0),
            ("PCy3", "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1", 34.0),
            ("PPh3", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", 8.0),
        ]
        solvents_da = [("DMAc", "CC(N(C)C)=O", 14.0), ("BuCN", "CCCC#N", 6.0), ("p-Xylene", "Cc1ccc(C)cc1", -6.0)]
        ood_solv_da = (
            "FC(F)(F)c1cc(Cl)ccc1Br",
            "CCCCCCCCc1ccc(I)cc1",
            "ClC(Cl)(Cl)C(F)(F)F",
            "CS(=O)(=O)c1ccc(F)cc1",
        )
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            b_name, b_smi = bases_da[i % len(bases_da)]
            l_name, l_smi, l_y = ligands_da[(i // 2) % len(ligands_da)]
            s_name, s_smi, s_y = solvents_da[(i // 3) % len(solvents_da)]
            conc = float(rng.uniform(0.32, 0.85) if ood else rng.uniform(0.08, 0.18))
            temp_c = float(rng.uniform(145.0, 195.0) if ood else rng.uniform(90.0, 120.0))
            if ood:
                s_smi = f"{'C' * (i % 6)}{ood_solv_da[i % len(ood_solv_da)]}"
            ood_shift = (-34.0 if i % 2 == 0 else 28.0) if ood else 0.0
            noise = float(rng.normal(0.0, 22.0 if ood else 2.0))
            yld = float(np.clip(12.0 + l_y + s_y + 0.42 * (temp_c - 80.0) + 28.0 * conc + ood_shift + noise, 0.0, 99.8))
            records.append(
                {
                    "entry": i,
                    "Base_SMILES": b_smi,
                    "Ligand_SMILES": l_smi,
                    "Solvent_SMILES": s_smi,
                    "Concentration": round(conc, 5),
                    "Temp_C": round(temp_c, 2),
                    "yield": round(yld, 3),
                    "Base": b_name,
                    "Ligand": l_name,
                    "Solvent": s_name,
                }
            )
        return records

    if spec.dataset_id == "bh_hte_ood_2026":
        aryls = [("Clc1cccnc1", 34.0), ("Brc1cccnc1", 66.0), ("Ic1cccnc1", 82.0)]
        amines = [("CC(N)CCc1ccccc1", 10.0), ("Nc1ccccc1", 4.0), ("CNCc1ccccc1", 14.0)]
        cats = [
            ("binap-pd-g3", "CS(=O)(=O)O[Pd]c1ccccc1P(c2ccccc2)c3ccccc3", 8.0),
            ("xphos-pd-g3", "CS(=O)(=O)O[Pd]c1ccccc1P(C2CCCCC2)C3CCCCC3", 14.0),
        ]
        bases_bh = [("DBU", "C1CCC2=NCCCN2CC1"), ("BTMG", "CN(C)C(=NC(C)(C)C)N(C)C")]
        solvs_bh = [("DMSO", "CS(C)=O"), ("Dioxane", "C1COCCO1")]
        ood_amines = (
            "Nc1c(F)c(F)c(Br)c(F)c1F",
            "Nc1cc(Cl)cc(I)c1C(F)(F)F",
            "CCCCCCCCCCNc1ccc2ccccc2c1",
            "Nc1ccc(S(=O)(=O)F)cc1Br",
        )
        for i in range(n):
            ood = bool(rng.random() < 0.16)
            ar_s, ar_y = aryls[i % len(aryls)]
            am_s, am_y = amines[(i // 3) % len(amines)]
            cat_n, cat_s, cat_y = cats[(i // 2) % len(cats)]
            b_n, b_s = bases_bh[i % len(bases_bh)]
            s_n, s_s = solvs_bh[(i // 4) % len(solvs_bh)]
            ar_u = f"{'C' * (i // 12)}{ar_s}"
            if ood:
                am_s = f"{'C' * (i % 7 + 2)}{ood_amines[i % len(ood_amines)]}"
            ood_shift = (-36.0 if i % 2 == 0 else 24.0) if ood else 0.0
            noise = float(rng.normal(0.0, 22.0 if ood else 2.2))
            yld = float(np.clip(ar_y + am_y + cat_y + ood_shift + noise, 0.0, 104.0))
            records.append(
                {
                    "Source": "JnJ25" if ood else "Sci.18",
                    "Aryl SMILES": ar_u,
                    "Amine SMILES": am_s,
                    "Catalyst Name": cat_n,
                    "Catalyst SMILES": cat_s,
                    "Base SMILES": b_s,
                    "Base Name": b_n,
                    "Solvent Name": s_n,
                    "Solvent SMILES": s_s,
                    "Product SMILES": "CC(CCc1ccccc1)Nc1cccnc1",
                    "DRFP": "[0, 1, 0]",
                    "DRFP_subst_prod": "[1, 0, 1]",
                    "QM": "[1.84, -0.32]",
                    "Yield": round(yld, 3),
                    "labels": int(yld >= 50.0),
                    "iteration_0 cluster": int(ood),
                    "iteration_0 train_test": "test" if ood else "train",
                }
            )
        return records

    raise RuntimeError(f"Unhandled sample fixture generator for {spec.dataset_id!r}")


def write_schema_sample_fixture(
    dataset_id: str,
    dest_path: str | os.PathLike[str],
    *,
    fmt: str | None = None,
    n: int = 120,
    seed: int = 0,
) -> str:
    """Write a deterministic schema-exact sample fixture in CSV, TSV, JSON, or JSONL format."""
    spec = get_open_dataset(dataset_id)
    out = os.fspath(dest_path)
    eff_fmt = _detect_format(out, explicit_fmt=fmt, default_fmt=spec.file_format)
    records = make_schema_sample_records(spec.dataset_id, n=n, seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if eff_fmt == "jsonl":
        with open(out, mode="w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    if eff_fmt == "json":
        if spec.dataset_id == "freesolv":
            # Mirror upstream MobleyLab/FreeSolv dict-keyed database.json schema
            keyed = {str(r["compound_id"]): {k: v for k, v in r.items() if k != "compound_id"} for r in records}
            with open(out, mode="w", encoding="utf-8") as fh:
                json.dump(keyed, fh, indent=2)
        else:
            with open(out, mode="w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
        return out

    sep = "\t" if eff_fmt == "tsv" else ","
    pd.DataFrame(records).to_csv(out, sep=sep, index=False, encoding="utf-8")
    return out
