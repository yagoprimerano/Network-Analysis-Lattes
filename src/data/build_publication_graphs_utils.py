"""
Utility functions and constants for building coauthorship graphs.
Kept simple and stateless so they can be reused by other data pipelines.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import unidecode

SCOPES = ["abrangente", "restritivo", "aplicacoes"]


def standardize_string(x) -> str:
    """Lowercase + strip accents; safe on NaN."""
    if pd.isna(x):
        return ""
    return unidecode.unidecode(str(x).lower().strip())


def read_dataset(data_dir: Path, scope: str, dataset: str) -> pd.DataFrame:
    """Read a processed CSV like data/processed/{scope}/{dataset}.csv"""
    path = data_dir / "processed" / scope / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # low_memory=False avoids DtypeWarning on large heterogeneous CSVs
    return pd.read_csv(path, low_memory=False)


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def cluster_by_year_blocks(indexes: pd.Index, years: pd.Series, next_id: int,
                           mapping: Dict[int, int]) -> int:
    """
    Assign cluster ids by scanning sorted years and starting a new cluster when the gap > 1.
    Unassigned (NaN year) rows get their own cluster (one per row).
    """
    yrs = pd.to_numeric(years, errors="coerce")
    ordered = yrs.dropna().astype(int).sort_values()
    last = None
    for idx, y in zip(ordered.index, ordered.values):
        if (last is None) or (abs(y - last) > 1):
            mapping[idx] = next_id
            next_id += 1
        else:
            mapping[idx] = next_id - 1
        last = y

    # Remaining (missing year) -> unique cluster per row
    remaining = [i for i in indexes if i not in mapping]
    for idx in remaining:
        mapping[idx] = next_id
        next_id += 1

    return next_id
