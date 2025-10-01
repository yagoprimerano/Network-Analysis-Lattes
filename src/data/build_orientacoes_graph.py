#!/usr/bin/env python3
"""
Build directed supervision graphs (advisor -> advisee) for all scopes by default.

- Filters STATUS == 'CONCLUIDA'
- Approach 1: direct ID links
- Approach 2: recover advisor IDs by name against the union of gerais of all scopes
- Deduplicate and remove self-loops
- Prefix 'LattesID_' and export graph

Outputs:
- data/processed/{scope}/orientacoes_edges.csv
- data/processed/graphs/{scope}/orientacoes.gexf
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import unidecode

SCOPES = ["abrangente", "restritivo", "aplicacoes"]
NULL_SENTINEL = -100000


def standardize_string(x) -> str:
    if pd.isna(x):
        return ""
    return unidecode.unidecode(str(x).lower().strip())


def read_dataset(data_dir: Path, scope: str, dataset: str) -> pd.DataFrame:
    path = data_dir / "processed" / scope / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def process_scope(scope: str, ROOT: Path) -> None:
    DATA_DIR = ROOT / "data"

    # Orientacoes
    df_orien = read_dataset(DATA_DIR, scope, "orientacoes")
    if "STATUS" in df_orien.columns:
        df_orien = df_orien[df_orien["STATUS"] == "CONCLUIDA"].copy()

    req_cols = ["LattesID", "NomeDoOrientador", "NumeroIdOrientado"]
    missing = [c for c in req_cols if c not in df_orien.columns]
    if missing:
        raise KeyError(f"Missing columns in orientacoes ({scope}): {missing}")

    df_orien = df_orien[req_cols].rename(columns={
        "LattesID": "LattesID_Orientando",
        "NumeroIdOrientado": "LattesID_Orientador"
    })
    df_orien["NomeDoOrientador"] = df_orien["NomeDoOrientador"].fillna("orientador com nome nao preenchido")
    df_orien["NomeDoOrientador"] = df_orien["NomeDoOrientador"].apply(standardize_string)
    df_orien["LattesID_Orientador"] = pd.to_numeric(df_orien["LattesID_Orientador"], errors="coerce").fillna(NULL_SENTINEL).astype(int)

    # Advisee names (for reference)
    df_geral = read_dataset(DATA_DIR, scope, "gerais")[["LattesID", "NOME-COMPLETO"]].copy()
    df_geral = df_geral.rename(columns={"NOME-COMPLETO": "NomeDoOrientando", "LattesID": "LattesID_Orientando"})
    df_geral["NomeDoOrientando"] = df_geral["NomeDoOrientando"].apply(standardize_string)

    df_merged = df_orien.merge(df_geral, on="LattesID_Orientando", how="left")
    df_merged = df_merged[["LattesID_Orientando", "NomeDoOrientando", "LattesID_Orientador", "NomeDoOrientador"]]

    # Final edges
    df_final = pd.DataFrame(columns=["LattesID_Orientando", "LattesID_Orientador"])

    # Approach 1: direct IDs
    first_mask = df_merged.drop(columns=["NomeDoOrientando", "NomeDoOrientador"]).copy()
    first_mask["LattesID_Orientador"] = first_mask["LattesID_Orientador"].replace(NULL_SENTINEL, np.nan)
    first_mask = first_mask.dropna(subset=["LattesID_Orientador"])
    first_mask["LattesID_Orientador"] = first_mask["LattesID_Orientador"].astype(int)
    df_final = pd.concat([df_final, first_mask], axis=0, ignore_index=True)

    # Approach 2: recover by name (union of all scopes)
    all_geral = []
    for sc in SCOPES:
        g = read_dataset(DATA_DIR, sc, "gerais")[["LattesID", "NOME-COMPLETO"]].copy()
        g["NOME-COMPLETO"] = g["NOME-COMPLETO"].apply(standardize_string)
        all_geral.append(g)
    df_all = pd.concat(all_geral, axis=0, ignore_index=True).drop_duplicates(subset=["LattesID", "NOME-COMPLETO"])
    df_all = df_all.rename(columns={"LattesID": "LattesID_Orientador", "NOME-COMPLETO": "NomeDoOrientador"})

    sec_mask = df_merged.drop(columns=["LattesID_Orientador"]).merge(df_all, on="NomeDoOrientador", how="inner")
    sec_mask = sec_mask[["LattesID_Orientando", "LattesID_Orientador"]]
    df_final = pd.concat([df_final, sec_mask], axis=0, ignore_index=True)

    # Cleanup
    df_final.drop_duplicates(inplace=True)
    df_final = df_final[df_final["LattesID_Orientando"] != df_final["LattesID_Orientador"]]
    df_final["Source"] = df_final["LattesID_Orientador"].apply(lambda x: f"LattesID_{x}")
    df_final["Target"] = df_final["LattesID_Orientando"].apply(lambda x: f"LattesID_{x}")
    df_final = df_final[["Source", "Target"]].copy()

    # Save
    out_edges_csv = DATA_DIR / "processed" / scope / "orientacoes_edges.csv"
    out_edges_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_edges_csv, index=False)

    graphs_dir = DATA_DIR / "processed" / "graphs" / scope
    graphs_dir.mkdir(parents=True, exist_ok=True)
    G = nx.DiGraph()
    for _, row in df_final.iterrows():
        G.add_edge(row["Source"], row["Target"])
    nx.write_gexf(G, graphs_dir / "orientacoes.gexf")

    print(f"[{scope}] Saved CSV  : {out_edges_csv}")
    print(f"[{scope}] Saved GEXF : {graphs_dir / 'orientacoes.gexf'}")


def main():
    parser = argparse.ArgumentParser(description="Build supervision graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope. If omitted, runs all scopes.")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    if args.scope:
        process_scope(args.scope, ROOT)
    else:
        for sc in SCOPES:
            try:
                process_scope(sc, ROOT)
            except Exception as e:
                print(f"[ERROR][{sc}] {e}")


if __name__ == "__main__":
    main()
