"""SMILES validation, RDKit ESOL 6D descriptor calculation, and Bemis-Murcko scaffolds."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

ESOL_FEATURES: list[str] = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
]
ESOL_TARGET: str = "measured log solubility in mols per litre"

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    HAS_RDKIT = True
except ImportError:  # pragma: no cover - optional dependency fallback
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
    HAS_RDKIT = False

_RING_DIGITS_RE = re.compile(r"(?:%\d\d|\d)")


def normalize_smiles_str(smiles: str, *, canonicalize: bool = False) -> str:
    """Strip whitespace from SMILES and optionally canonicalize via RDKit."""
    if not isinstance(smiles, str):
        raise ValueError(f"Expected string SMILES, got {type(smiles).__name__}")
    trimmed = smiles.strip()
    if not trimmed:
        raise ValueError("SMILES string must not be empty")
    if not canonicalize:
        return trimmed
    if not HAS_RDKIT:
        raise ImportError("RDKit is required when canonicalize=True; install `lab-decide[chem]`.")
    mol = Chem.MolFromSmiles(trimmed)
    if mol is None:
        raise ValueError(f"Invalid SMILES string could not be parsed by RDKit: {trimmed!r}")
    return str(Chem.MolToSmiles(mol, canonical=True))


def smiles_to_descriptors(smiles: str) -> dict[str, float]:
    """Compute exact ESOL 6D descriptors from a SMILES string via RDKit."""
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit is required for smiles_to_descriptors(); install `pip install rdkit`."
        )
    clean = normalize_smiles_str(smiles, canonicalize=False)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
    min_deg = float(min(degrees)) if degrees else 0.0
    return {
        "Minimum Degree": min_deg,
        "Molecular Weight": float(Descriptors.MolWt(mol)),
        "Number of H-Bond Donors": float(Lipinski.NumHDonors(mol)),
        "Number of Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "Number of Rotatable Bonds": float(Lipinski.NumRotatableBonds(mol)),
        "Polar Surface Area": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def featurize_smiles_batch(
    smiles_list: Sequence[str],
    *,
    feature_names: Sequence[str] = ESOL_FEATURES,
) -> np.ndarray:
    """Convert a batch of SMILES strings into a 2D descriptor matrix `(N, len(feature_names))`."""
    if not smiles_list:
        return np.empty((0, len(feature_names)), dtype=np.float64)
    rows: list[list[float]] = []
    for smi in smiles_list:
        desc = smiles_to_descriptors(smi)
        missing = [f for f in feature_names if f not in desc]
        if missing:
            raise ValueError(
                f"Requested features {missing} cannot be computed by ESOL SMILES bridge"
            )
        rows.append([desc[f] for f in feature_names])
    return np.asarray(rows, dtype=np.float64)


def murcko_scaffold_group(smiles: str) -> str:
    """Return Bemis-Murcko scaffold SMILES (or deterministic topology signature without RDKit)."""
    clean = smiles.strip()
    if not clean:
        return "empty"
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(clean)
        if mol is not None:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            return scaf if scaf else "acyclic"
    # Deterministic pure-Python fallback grouping cyclic ring-closure signatures vs acyclic chains
    rings = len(_RING_DIGITS_RE.findall(clean)) // 2
    arom = sum(1 for c in clean if c in "cnos")
    return f"fallback_r{rings}_a{arom > 0}"


def compute_scaffold_groups(smiles_list: Sequence[str]) -> np.ndarray:
    """Compute scaffold group labels for Bemis-Murcko scaffold splitting."""
    return np.asarray([murcko_scaffold_group(s) for s in smiles_list], dtype=object)
