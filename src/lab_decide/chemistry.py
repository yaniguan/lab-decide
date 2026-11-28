"""SMILES validation, RDKit ESOL 6D & 2D+Morgan descriptors, Delaney 2004 formula, and Bemis-Murcko scaffolds."""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Callable, Sequence

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
DELANEY_CSV_PRED_COLUMN: str = "ESOL predicted log solubility in mols per litre"

DELANEY_2004_FEATURES: tuple[str, ...] = (
    "cLogP",
    "MWT",
    "RB",
    "AP",
)

RDKIT_MOL_EXTRA_FEATURES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
)

CORE_RDKIT_2D_NAMES: tuple[str, ...] = (
    "MolLogP",
    "MolWt",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "HeavyAtomCount",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "MaxPartialCharge",
    "MinPartialCharge",
)

INTENSIVE_RATIO_FEATURES: tuple[str, ...] = (
    "arom_atom_prop",
    "hetero_atom_prop",
    "halogen_atom_prop",
    "log1p_heavy_atoms",
)

RDKIT_MACCS_BITS: int = 167
RDKIT_2D_MORGAN_FP_BITS: int = 1024
_DESCRIPTOR_ABS_CLIP: float = 1e6

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import (
        Crippen,
        Descriptors,
        Lipinski,
        MACCSkeys,
        rdFingerprintGenerator,
        rdMolDescriptors,
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:  # pragma: no cover - exercised in no-RDKit CI job
    Chem = None  # type: ignore[assignment]
    Crippen = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    MACCSkeys = None  # type: ignore[assignment]
    rdFingerprintGenerator = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
    HAS_RDKIT = False


def _require_rdkit(context: str) -> None:
    if not HAS_RDKIT or Chem is None:
        raise ImportError(
            f"RDKit is required for {context}; install `lab-decide[chem]` (`pip install rdkit`)."
        )


@lru_cache(maxsize=8)
def _get_morgan_generator(radius: int = 2, fp_size: int = 32) -> Any:
    _require_rdkit("Morgan fingerprint featurization")
    return rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))


@lru_cache(maxsize=1)
def _get_rdkit_2d_descriptor_specs() -> tuple[tuple[tuple[str, Callable[[Any], Any]], ...], int, int]:
    """Return ordered `(name, fn)` RDKit 2D descriptor functions plus `(n_core_phys, n_desc_maccs)`.

    Excludes raw `Ipc` (`AvgIpc` is retained; unnormalized `Ipc` overflows binary64 on macrocycles)
    and redundant slow `qed` alerts while placing `CORE_RDKIT_2D_NAMES` first for semiparametric
    linear-trend conditioning under scaffold shift.
    """
    _require_rdkit("encoding='rdkit_2d_morgan'")
    desc_map: dict[str, Callable[[Any], Any]] = dict(Descriptors.descList)
    core_specs: list[tuple[str, Callable[[Any], Any]]] = [
        (nm, desc_map[nm]) for nm in CORE_RDKIT_2D_NAMES if nm in desc_map
    ]
    core_set = {nm for nm, _ in core_specs}
    extra_specs: list[tuple[str, Callable[[Any], Any]]] = [
        (nm, fn)
        for nm, fn in sorted(Descriptors.descList, key=lambda kv: kv[0])
        if nm not in ("Ipc", "qed") and nm not in core_set
    ]
    all_specs = tuple(core_specs + extra_specs)
    n_core_phys = len(ESOL_FEATURES) + len(INTENSIVE_RATIO_FEATURES) + len(core_specs)
    n_desc_maccs = len(ESOL_FEATURES) + len(INTENSIVE_RATIO_FEATURES) + len(all_specs) + RDKIT_MACCS_BITS
    return all_specs, n_core_phys, n_desc_maccs


def get_rdkit_2d_morgan_layout() -> tuple[int, int, int]:
    """Return `(n_core_phys, n_desc_maccs, n_fp_bits)` column layout for `rdkit_2d_morgan`."""
    _, n_core_phys, n_desc_maccs = _get_rdkit_2d_descriptor_specs()
    return n_core_phys, n_desc_maccs, RDKIT_2D_MORGAN_FP_BITS


@lru_cache(maxsize=1)
def get_rdkit_2d_morgan_feature_names() -> tuple[str, ...]:
    """Return deterministic feature names for `encoding='rdkit_2d_morgan'`."""
    all_specs, _, _ = _get_rdkit_2d_descriptor_specs()
    return (
        *ESOL_FEATURES,
        *INTENSIVE_RATIO_FEATURES,
        *(f"rdkit2d_{nm}" for nm, _ in all_specs),
        *(f"maccs_{b}" for b in range(RDKIT_MACCS_BITS)),
        *(f"morgan_r2_cnt_{b}" for b in range(RDKIT_2D_MORGAN_FP_BITS)),
    )


def normalize_smiles_str(smiles: str, *, canonicalize: bool = False) -> str:
    """Strip whitespace from SMILES and optionally canonicalize via RDKit."""
    if not isinstance(smiles, str):
        raise ValueError(f"Expected string SMILES, got {type(smiles).__name__}")
    trimmed = smiles.strip()
    if not trimmed:
        raise ValueError("SMILES string must not be empty")
    if not canonicalize:
        return trimmed
    _require_rdkit("canonicalize=True")
    mol = Chem.MolFromSmiles(trimmed)
    if mol is None:
        raise ValueError(f"Invalid SMILES string could not be parsed by RDKit: {trimmed!r}")
    return str(Chem.MolToSmiles(mol, canonical=True))


def canonical_group_smiles(smiles: str) -> str:
    """Return canonical SMILES when RDKit is available and SMILES parses, else stripped string."""
    clean = str(smiles).strip()
    if not clean:
        raise ValueError("Group SMILES string must not be empty")
    if HAS_RDKIT and Chem is not None:
        mol = Chem.MolFromSmiles(clean)
        if mol is not None:
            return str(Chem.MolToSmiles(mol, canonical=True))
    return clean


def smiles_to_descriptors(smiles: str) -> dict[str, float]:
    """Compute exact ESOL 6D descriptors from a SMILES string via RDKit."""
    if not HAS_RDKIT or Chem is None:
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


def delaney_2004_descriptors(smiles: str) -> tuple[float, float, float, float]:
    """Compute Delaney's (2004) 4 descriptors `(cLogP, MWT, RB, AP)` via RDKit."""
    _require_rdkit("delaney_2004_descriptors()")
    clean = normalize_smiles_str(smiles, canonicalize=False)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    clogp = float(Crippen.MolLogP(mol))
    mwt = float(Descriptors.MolWt(mol))
    rb = float(Lipinski.NumRotatableBonds(mol))
    n_heavy = int(mol.GetNumHeavyAtoms())
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = float(n_arom) / float(n_heavy) if n_heavy > 0 else 0.0
    return (clogp, mwt, rb, ap)


def featurize_delaney_4d(smiles_list: Sequence[str]) -> np.ndarray:
    """Compute `(N, 4)` Delaney (2004) descriptor matrix `(cLogP, MWT, RB, AP)`."""
    _require_rdkit("featurize_delaney_4d()")
    if not smiles_list:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray([delaney_2004_descriptors(s) for s in smiles_list], dtype=np.float64)


def delaney_2004_formula_predict(smiles_list: Sequence[str]) -> np.ndarray:
    """Evaluate Delaney's published 2004 linear formula: `0.16 - 0.63 cLogP - 0.0062 MWT + 0.066 RB - 0.74 AP`."""
    X4 = featurize_delaney_4d(smiles_list)
    if X4.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return (
        0.16
        - 0.63 * X4[:, 0]
        - 0.0062 * X4[:, 1]
        + 0.066 * X4[:, 2]
        - 0.74 * X4[:, 3]
    )


@lru_cache(maxsize=16384)
def _cached_rdkit_2d_morgan(clean_smiles: str, strict: bool) -> bytes:
    """Compute and cache packed float64 `rdkit_2d_morgan` vector bytes for `clean_smiles`."""
    all_specs, _, n_desc_maccs = _get_rdkit_2d_descriptor_specs()
    total_dim = n_desc_maccs + RDKIT_2D_MORGAN_FP_BITS
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None and not strict:
        mol = Chem.MolFromSmiles(clean_smiles, sanitize=False)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
            except Exception:
                mol = None
    if mol is None:
        if strict:
            raise ValueError(f"Invalid SMILES string: {clean_smiles!r}")
        return np.zeros(total_dim, dtype=np.float64).tobytes()

    vec = np.zeros(total_dim, dtype=np.float64)
    try:
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        vec[0] = float(min(degrees)) if degrees else 0.0
        vec[1] = float(Descriptors.MolWt(mol))
        vec[2] = float(Lipinski.NumHDonors(mol))
        vec[3] = float(rdMolDescriptors.CalcNumRings(mol))
        vec[4] = float(Lipinski.NumRotatableBonds(mol))
        vec[5] = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        vec[1] = float(mol.GetNumHeavyAtoms() * 14.0)

    n_heavy = max(int(mol.GetNumHeavyAtoms()), 1)
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    n_hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    vec[6] = float(n_arom) / float(n_heavy)
    vec[7] = float(n_hetero) / float(n_heavy)
    vec[8] = float(n_hal) / float(n_heavy)
    vec[9] = math.log1p(float(n_heavy))

    offset = len(ESOL_FEATURES) + len(INTENSIVE_RATIO_FEATURES)
    for j, (_, fn) in enumerate(all_specs):
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        if not math.isfinite(v):
            v = 0.0
        elif v > _DESCRIPTOR_ABS_CLIP:
            v = _DESCRIPTOR_ABS_CLIP
        elif v < -_DESCRIPTOR_ABS_CLIP:
            v = -_DESCRIPTOR_ABS_CLIP
        vec[offset + j] = v

    maccs_start = offset + len(all_specs)
    try:
        maccs_bv = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.frombuffer(maccs_bv.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
        vec[maccs_start : maccs_start + RDKIT_MACCS_BITS] = maccs_arr.astype(np.float64)
    except Exception:
        pass

    try:
        gen = _get_morgan_generator(radius=2, fp_size=RDKIT_2D_MORGAN_FP_BITS)
        cnt_fp = gen.GetCountFingerprintAsNumPy(mol)
        vec[n_desc_maccs:] = np.log1p(cnt_fp.astype(np.float64))
    except Exception:
        pass

    # Final finite hygiene guarantee
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=_DESCRIPTOR_ABS_CLIP, neginf=-_DESCRIPTOR_ABS_CLIP)
    return vec.tobytes()


def smiles_to_rdkit_2d_morgan(smiles: str, *, strict: bool = True) -> np.ndarray:
    """Compute standardized 1D `rdkit_2d_morgan` descriptor + MACCS + 1024-bit count-Morgan vector."""
    _require_rdkit("smiles_to_rdkit_2d_morgan()")
    clean = normalize_smiles_str(smiles, canonicalize=False)
    raw_bytes = _cached_rdkit_2d_morgan(clean, bool(strict))
    return np.frombuffer(raw_bytes, dtype=np.float64).copy()


def featurize_rdkit_2d_morgan_batch(
    smiles_list: Sequence[str],
    *,
    strict: bool = False,
) -> np.ndarray:
    """Convert a batch of SMILES strings into a 2D `rdkit_2d_morgan` matrix `(N, D)`."""
    _require_rdkit("featurize_rdkit_2d_morgan_batch()")
    feat_names = get_rdkit_2d_morgan_feature_names()
    total_dim = len(feat_names)
    n = len(smiles_list)
    if n == 0:
        return np.empty((0, total_dim), dtype=np.float64)

    out = np.empty((n, total_dim), dtype=np.float64)
    local_cache: dict[str, np.ndarray] = {}
    for i, raw_smi in enumerate(smiles_list):
        if not isinstance(raw_smi, str) or not raw_smi.strip():
            if strict:
                raise ValueError("SMILES string must not be empty")
            out[i] = 0.0
            continue
        clean = raw_smi.strip()
        cached_arr = local_cache.get(clean)
        if cached_arr is None:
            raw_b = _cached_rdkit_2d_morgan(clean, bool(strict))
            cached_arr = np.frombuffer(raw_b, dtype=np.float64)
            local_cache[clean] = cached_arr
        out[i] = cached_arr
    return out


@lru_cache(maxsize=16384)
def smiles_to_morgan_physicochem(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 32,
) -> tuple[float, ...]:
    """Compute `(present, 6 RDKit physicochemical descriptors, n_bits Morgan FP)` via RDKit."""
    _require_rdkit("Morgan + physicochemical reagent encoding")
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
    """Convert a batch of SMILES strings into a 2D feature matrix matching `feature_names`."""
    if HAS_RDKIT and len(feature_names) > 1000 and tuple(feature_names) == get_rdkit_2d_morgan_feature_names():
        return featurize_rdkit_2d_morgan_batch(smiles_list, strict=True)
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
    if not HAS_RDKIT or Chem is None:
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
    if not HAS_RDKIT or Chem is None:
        raise ImportError(
            "RDKit is required for compute_scaffold_groups(); install `lab-decide[chem]` (`pip install rdkit`)."
        )
    return np.asarray([murcko_scaffold_group(s) for s in smiles_list], dtype=object)
