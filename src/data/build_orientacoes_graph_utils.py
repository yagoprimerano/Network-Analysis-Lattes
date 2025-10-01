"""
Utility functions and constants for building the supervision (orientações) graph.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import unidecode

SCOPES = ["abrangente", "restritivo", "aplicacoes"]
NULL_SENTINEL = -100000


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
    return pd.read_csv(path)
