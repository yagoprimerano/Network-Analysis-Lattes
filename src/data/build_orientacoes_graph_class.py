"""
Class-based orchestrator for building the directed supervision (orientações) graph.

Behavior mirrors the original script:
- Filter STATUS == 'CONCLUIDA'
- Approach 1: direct ID links
- Approach 2: recover advisor IDs by name (matching union of 'gerais' across all scopes)
- Deduplicate and remove self-loops
- Prefix with 'LattesID_' and export CSV edges and GEXF
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

from .build_orientacoes_graph_utils import (
    SCOPES,
    NULL_SENTINEL,
    standardize_string,
    read_dataset,
)


class OrientacoesGraphBuilder:
    def __init__(self, root: Path, scope: str):
        self.root = root
        self.scope = scope
        self.data_dir = self.root / "data"
        self.graphs_dir = self.data_dir / "processed" / "graphs" / self.scope
        self.out_edges_csv = self.data_dir / "processed" / self.scope / "orientacoes_edges.csv"

    def _approach_union_gerais(self) -> pd.DataFrame:
        """Build union of 'gerais' across all scopes for (name -> id) matching."""
        all_geral = []
        for sc in SCOPES:
            g = read_dataset(self.data_dir, sc, "gerais")[["LattesID", "NOME-COMPLETO"]].copy()
            g["NOME-COMPLETO"] = g["NOME-COMPLETO"].apply(standardize_string)
            all_geral.append(g)
        df_all = pd.concat(all_geral, axis=0, ignore_index=True).drop_duplicates(subset=["LattesID", "NOME-COMPLETO"])
        df_all = df_all.rename(columns={"LattesID": "LattesID_Orientador", "NOME-COMPLETO": "NomeDoOrientador"})
        return df_all

    def run_full_pipeline(self) -> None:
        """Create edges CSV and directed GEXF for the given scope."""
        # Load and filter orientacoes
        df_orien = read_dataset(self.data_dir, self.scope, "orientacoes")
        if "STATUS" in df_orien.columns:
            df_orien = df_orien[df_orien["STATUS"] == "CONCLUIDA"].copy()

        req_cols = ["LattesID", "NomeDoOrientador", "NumeroIdOrientado"]
        missing = [c for c in req_cols if c not in df_orien.columns]
        if missing:
            raise KeyError(f"Missing columns in orientacoes ({self.scope}): {missing}")

        df_orien = df_orien[req_cols].rename(columns={
            "LattesID": "LattesID_Orientando",
            "NumeroIdOrientado": "LattesID_Orientador"
        })

        df_orien["NomeDoOrientador"] = df_orien["NomeDoOrientador"].fillna("orientador com nome nao preenchido")
        df_orien["NomeDoOrientador"] = df_orien["NomeDoOrientador"].apply(standardize_string)
        df_orien["LattesID_Orientador"] = (
            pd.to_numeric(df_orien["LattesID_Orientador"], errors="coerce")
            .fillna(NULL_SENTINEL)
            .astype(int)
        )

        # Advisee names (reference)
        df_geral = read_dataset(self.data_dir, self.scope, "gerais")[["LattesID", "NOME-COMPLETO"]].copy()
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
        df_all = self._approach_union_gerais()
        sec_mask = df_merged.drop(columns=["LattesID_Orientador"]).merge(df_all, on="NomeDoOrientador", how="inner")
        sec_mask = sec_mask[["LattesID_Orientando", "LattesID_Orientador"]]
        df_final = pd.concat([df_final, sec_mask], axis=0, ignore_index=True)

        # Cleanup
        df_final.drop_duplicates(inplace=True)
        df_final = df_final[df_final["LattesID_Orientando"] != df_final["LattesID_Orientador"]]
        df_final["Source"] = df_final["LattesID_Orientador"].apply(lambda x: f"LattesID_{x}")
        df_final["Target"] = df_final["LattesID_Orientando"].apply(lambda x: f"LattesID_{x}")
        df_final = df_final[["Source", "Target"]].copy()

        # Save CSV and GEXF
        self.out_edges_csv.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(self.out_edges_csv, index=False)

        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        G = nx.DiGraph()
        for _, row in df_final.iterrows():
            G.add_edge(row["Source"], row["Target"])
        nx.write_gexf(G, self.graphs_dir / "orientacoes.gexf")
