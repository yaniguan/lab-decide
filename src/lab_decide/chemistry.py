"""SMILES validation, RDKit ESOL 6D descriptors, Morgan fingerprints, and Bemis-Murcko scaffolds."""
from __future__ import annotations

from functools import lru_cache
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

RDKIT_MOL_EXTRA_FEATURES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:  # pragma: no cover - exercised in no-RDKit CI job
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    rdFingerprintGenerator = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
    HAS_RDKIT = False


@lru_cache(maxsize=8)
def _get_morgan_generator(radius: int = 2, fp_size: int = 32):
    if not HAS_RDKIT or rdFingerprintGenerator is None:
        raise ImportError(
            "RDKit is required for Morgan fingerprint featurization; install `lab-decide[chem]`."
        )
    return rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))


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


def canonical_group_smiles(smiles: str) -> str:
    """Return canonical SMILES when RDKit is available and SMILES parses, else stripped string."""
    clean = str(smiles).strip()
    if not clean:
        raise ValueError("Group SMILES string must not be empty")
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(clean)
        if mol is not None:
            return str(Chem.MolToSmiles(mol, canonical=True))
    return clean


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


@lru_cache(maxsize=16384)
def smiles_to_morgan_physicochem(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 32,
) -> tuple[float, ...]:
    """Compute `(present, 6 RDKit physicochemical descriptors, n_bits Morgan FP)` via RDKit."""
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit is required for Morgan + physicochemical reagent encoding; install `lab-decide[chem]`."
        )
    clean = (smiles or "").strip()
    dim = 1 + len(RDKIT_MOL_EXTRA_FEATURES) + int(n_bits)
    if not clean or clean.upper() == "NONE" or clean.lower() == "nan":
        return tuple(0.0 for _ in range(dim))
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        # Handle organometallic complexes with dative/coordination bonds where full sanitization fails
        mol = Chem.MolFromSmiles(clean, sanitize=False)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
            except Exception:
                mol = None
    if mol is None:
        return (1.0,) + tuple(0.0 for _ in range(dim - 1))

    try:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd = float(Lipinski.NumHDonors(mol))
        hba = float(Lipinski.NumHAcceptors(mol))
        rings = float(rdMolDescriptors.CalcNumRings(mol))
    except Exception:
        mw = float(mol.GetNumHeavyAtoms() * 14.0)
        logp = 0.0
        tpsa = 0.0
        hbd = 0.0
        hba = 0.0
        rings = 0.0

    try:
        gen = _get_morgan_generator(radius=radius, fp_size=n_bits)
        fp_arr = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float64)
        fp_vals = tuple(float(x) for x in fp_arr.tolist())
    except Exception:
        fp_vals = tuple(0.0 for _ in range(int(n_bits)))

    return (1.0, mw, logp, tpsa, hbd, hba, rings, *fp_vals)


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
    """Return exact RDKit Bemis-Murcko scaffold SMILES (`'acyclic'` for valid acyclic molecules).

    Raises `ImportError` if RDKit is unavailable and `ValueError` if `smiles` cannot be parsed
    as a single molecule (never falls back to regex ring-digit heuristics).
    """
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit is required for Bemis-Murcko scaffold splitting (`strategy='scaffold'`); "
            "install `lab-decide[chem]` (`pip install rdkit`) or use `--split group` / `--split random`."
        )
    if not isinstance(smiles, str):
        raise ValueError(f"Expected string SMILES for scaffold computation, got {type(smiles).__name__}")
    clean = smiles.strip()
    if not clean:
        raise ValueError("SMILES string must not be empty for Bemis-Murcko scaffold computation")
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(
            f"Invalid SMILES string could not be parsed by RDKit for Bemis-Murcko scaffold: {clean!r}. "
            "For multi-component formulations or reaction records, use `--split group`."
        )
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return str(scaf) if scaf else "acyclic"


def compute_scaffold_groups(smiles_list: Sequence[str]) -> np.ndarray:
    """Compute exact RDKit Bemis-Murcko scaffold group labels (never uses regex fallback)."""
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit is required for compute_scaffold_groups(); install `lab-decide[chem]` (`pip install rdkit`)."
        )
    return np.asarray([murcko_scaffold_group(s) for s in smiles_list], dtype=object)
