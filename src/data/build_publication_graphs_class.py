"""
Class-based orchestrator for building coauthorship graphs.

This class encapsulates the logic found in the original script:
- clustering books, journals, chapters, events
- creating author-work pairs
- building weighted coauthorship graphs per set and combined
- writing CSV weights and GEXF files to the project structure
"""

from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List

import pandas as pd
import networkx as nx
from tqdm import tqdm

from .build_publication_graphs_utils import (
    standardize_string,
    read_dataset,
    ensure_dir,
    cluster_by_year_blocks,
)


class PublicationGraphBuilder:
    def __init__(self, root: Path, scope: str, verbose: bool = False):
        self.root = root
        self.scope = scope
        self.verbose = verbose
        self.data_dir = self.root / "data"
        self.weights_dir = self.data_dir / "processed" / self.scope / "graphs_weights"
        self.graphs_dir = self.data_dir / "processed" / "graphs" / self.scope

    # --------------------------- Clustering methods --------------------------- #

    def cluster_books(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Books: ISBN → standardized title + year±1 -> returns df with 'book_id' and author-work pairs."""
        df = df.copy()
        if "STANDARDIZE-TITULO-DO-LIVRO" not in df.columns:
            df["STANDARDIZE-TITULO-DO-LIVRO"] = df["TITULO-DO-LIVRO"].astype(str).apply(standardize_string)
        df["std_isbn"] = df["ISBN"].fillna("").apply(standardize_string)
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")

        clusters, next_id = {}, 0
        for _, grp in df[df["std_isbn"] != ""].groupby("std_isbn", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_code = df[df["std_isbn"] == ""]
        for _, grp in no_code.groupby("STANDARDIZE-TITULO-DO-LIVRO", sort=False):
            next_id = cluster_by_year_blocks(grp.index, grp["ANO"], next_id, clusters)

        df["book_id"] = df.index.map(clusters)
        pairs = df[["LattesID", "book_id"]].copy()
        return df, pairs

    def cluster_periodicos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Journals: DOI → ISSN → (std title + std journal + year±1)."""
        df = df.copy()
        df["TITULO-DO-ARTIGO"] = df["TITULO-DO-ARTIGO"].astype(str).apply(standardize_string)
        df["TITULO-DO-PERIODICO-OU-REVISTA"] = df["TITULO-DO-PERIODICO-OU-REVISTA"].astype(str).apply(standardize_string)
        df["std_doi"] = df["DOI"].fillna("").apply(standardize_string)
        df["std_issn"] = df["ISSN"].fillna("").apply(standardize_string)
        df["ANO-DO-ARTIGO"] = pd.to_numeric(df["ANO-DO-ARTIGO"], errors="coerce")

        clusters, next_id = {}, 0
        for _, grp in df[df["std_doi"] != ""].groupby("std_doi", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_doi = df[df["std_doi"] == ""]
        for _, grp in no_doi[no_doi["std_issn"] != ""].groupby("std_issn", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_code = df[(df["std_doi"] == "") & (df["std_issn"] == "")]
        for _, grp in no_code.groupby(["TITULO-DO-ARTIGO", "TITULO-DO-PERIODICO-OU-REVISTA"], sort=False):
            next_id = cluster_by_year_blocks(grp.index, grp["ANO-DO-ARTIGO"], next_id, clusters)

        df["paper_id"] = df.index.map(clusters)
        pairs = df[["LattesID", "paper_id"]].copy()
        return df, pairs

    def cluster_capitulos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chapters: ISBN → (std chapter title + std book title + year±1)."""
        df = df.copy()
        if "LATTES_ID" in df.columns:
            df["LATTES_ID"] = df["LATTES_ID"].astype(str).str.replace("'", "", regex=False)
            df["LattesID"] = df.get("LattesID", df["LATTES_ID"])
        else:
            df["LattesID"] = df["LattesID"].astype(str)

        df["TITULO-DO-CAPITULO-DO-LIVRO"] = df["TITULO-DO-CAPITULO-DO-LIVRO"].astype(str).apply(standardize_string)
        df["TITULO-DO-LIVRO"] = df["TITULO-DO-LIVRO"].astype(str).apply(standardize_string)
        df["std_isbn"] = df["ISBN"].fillna("").apply(standardize_string)
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")

        clusters, next_id = {}, 0
        for _, grp in df[df["std_isbn"] != ""].groupby("std_isbn", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_code = df[df["std_isbn"] == ""]
        for _, grp in no_code.groupby(["TITULO-DO-CAPITULO-DO-LIVRO", "TITULO-DO-LIVRO"], sort=False):
            next_id = cluster_by_year_blocks(grp.index, grp["ANO"], next_id, clusters)

        df["chapter_id"] = df.index.map(clusters)
        pairs = df[["LattesID", "chapter_id"]].copy()
        return df, pairs

    def cluster_eventos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Events: choose best year column → DOI → ISBN → (std title + std event + year±1)."""
        df = df.copy()
        df["STANDARDIZE-TITULO-DO-TRABALHO"] = df["TITULO-DO-TRABALHO"].astype(str).apply(standardize_string)
        df["STANDARDIZE-NOME-DO-EVENTO"] = df["NOME-DO-EVENTO"].astype(str).apply(standardize_string)

        c1, c2 = "ANO-DO-TRABALHO", "ANO-DE-REALIZACAO"
        c1_nonnull = df[c1].notna().sum() if c1 in df.columns else -1
        c2_nonnull = df[c2].notna().sum() if c2 in df.columns else -1
        year_col = c1 if c1_nonnull >= c2_nonnull else c2
        if year_col not in df.columns:
            raise ValueError("No valid year column among 'ANO-DO-TRABALHO' or 'ANO-DE-REALIZACAO'.")

        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df["std_doi"] = df["DOI"].fillna("").apply(standardize_string)
        df["std_isbn"] = df["ISBN"].fillna("").apply(standardize_string)

        clusters, next_id = {}, 0
        for _, grp in df[df["std_doi"] != ""].groupby("std_doi", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_doi = df[df["std_doi"] == ""]
        for _, grp in no_doi[no_doi["std_isbn"] != ""].groupby("std_isbn", sort=False):
            for idx in grp.index:
                clusters[idx] = next_id
            next_id += 1

        no_code = df[(df["std_doi"] == "") & (df["std_isbn"] == "")]
        for _, grp in no_code.groupby(["STANDARDIZE-TITULO-DO-TRABALHO", "STANDARDIZE-NOME-DO-EVENTO"], sort=False):
            next_id = cluster_by_year_blocks(grp.index, grp[year_col], next_id, clusters)

        df["paper_id"] = df.index.map(clusters)
        pairs = df[["LattesID", "paper_id"]].copy()
        return df, pairs

    # --------------------------- Graph helpers -------------------------------- #

    def _create_and_save_graph(self, df_pairs: pd.DataFrame, set_type: str) -> nx.Graph:
        """Build an undirected weighted coauthorship graph from (LattesID, work_id) pairs."""
        # 1) gerar todas as combinações (pares) por trabalho
        pair_list: List[Tuple[str, str]] = []
        gp = df_pairs.groupby("work_id")
        for _, group in tqdm(gp, total=len(gp), desc=f"Pairs[{set_type}]"):
            users = list(group["LattesID"])
            if len(users) > 1:
                # usa combinations e ordena o par para evitar (A,B) e (B,A)
                pair_list.extend([tuple(sorted(p)) for p in combinations(users, 2)])

        # Se não houver pares, ainda assim escrevemos arquivos vazios
        if not pair_list:
            coauthor_weighted = pd.DataFrame(columns=["author1", "author2", "count"])
            ensure_dir(self.weights_dir)
            coauthor_weighted.to_csv(self.weights_dir / f"{set_type}_weighted.csv", index=False)
            G_empty = nx.Graph()
            ensure_dir(self.graphs_dir)
            nx.write_gexf(G_empty, self.graphs_dir / f"{set_type}_graph.gexf")
            return G_empty

        # 2) contar ocorrências (peso)
        pairs_df = pd.DataFrame(pair_list, columns=["author1", "author2"])
        coauthor_weighted = (
            pairs_df.assign(count=1)
                    .groupby(["author1", "author2"], as_index=False)["count"]
                    .sum()
        )

        # 3) salvar CSV de pesos
        ensure_dir(self.weights_dir)
        coauthor_weighted.to_csv(self.weights_dir / f"{set_type}_weighted.csv", index=False)

        # 4) construir grafo e salvar GEXF
        G = nx.Graph()
        for _, row in tqdm(coauthor_weighted.iterrows(), total=len(coauthor_weighted), desc=f"Graph[{set_type}]"):
            G.add_edge(row["author1"], row["author2"], weight=int(row["count"]))

        ensure_dir(self.graphs_dir)
        nx.write_gexf(G, self.graphs_dir / f"{set_type}_graph.gexf")
        return G

    @staticmethod
    def _prefix_work_ids(df_user_per, df_user_eventos, df_user_cap, df_user_book) -> Dict[str, pd.DataFrame]:
        df_user_per = df_user_per.rename(columns={"paper_id": "work_id"}).copy()
        df_user_per["work_id"] = df_user_per["work_id"].astype(str).apply(lambda x: f"periodicos_{x}")

        df_user_eventos = df_user_eventos.rename(columns={"paper_id": "work_id"}).copy()
        df_user_eventos["work_id"] = df_user_eventos["work_id"].astype(str).apply(lambda x: f"eventos_{x}")

        df_user_cap = df_user_cap.rename(columns={"chapter_id": "work_id"}).copy()
        df_user_cap["work_id"] = df_user_cap["work_id"].astype(str).apply(lambda x: f"capitulos_{x}")

        df_user_book = df_user_book.rename(columns={"book_id": "work_id"}).copy()
        df_user_book["work_id"] = df_user_book["work_id"].astype(str).apply(lambda x: f"livros_{x}")

        return {
            "periodicos": df_user_per,
            "eventos": df_user_eventos,
            "capitulos": df_user_cap,
            "livros": df_user_book,
        }

    # --------------------------- Orchestrator --------------------------------- #

    def run_full_pipeline(self) -> None:
        """Read datasets, cluster, assemble pairs, build and save graphs for all sets + coauthorship."""
        # Read base datasets
        df_user = read_dataset(self.data_dir, self.scope, "gerais")
        df_book_raw = read_dataset(self.data_dir, self.scope, "livros")
        df_per_raw = read_dataset(self.data_dir, self.scope, "periodicos")
        df_cap_raw = read_dataset(self.data_dir, self.scope, "capitulos")
        df_evt_raw = read_dataset(self.data_dir, self.scope, "eventos")

        # Cluster
        _, df_user_book = self.cluster_books(df_book_raw)
        _, df_user_per = self.cluster_periodicos(df_per_raw)
        _, df_user_cap = self.cluster_capitulos(df_cap_raw)
        _, df_user_evt = self.cluster_eventos(df_evt_raw)

        # Prefix work ids
        bundles = self._prefix_work_ids(df_user_per, df_user_evt, df_user_cap, df_user_book)

        # Prefix researcher ids and filter to those present in 'gerais'
        valid_ids = set(df_user["LattesID"].apply(lambda x: f"LattesID_{x}"))
        for key, dfp in bundles.items():
            dfp["LattesID"] = dfp["LattesID"].apply(lambda x: f"LattesID_{x}")
            dfp.drop_duplicates(inplace=True)
            bundles[key] = dfp[dfp["LattesID"].isin(valid_ids)]

        # Combined coauthorship
        df_co = pd.concat(list(bundles.values()), axis=0, ignore_index=True).drop_duplicates()

        # Build and save graphs
        self._create_and_save_graph(bundles["periodicos"], "periodicos")
        self._create_and_save_graph(bundles["eventos"], "eventos")
        self._create_and_save_graph(bundles["capitulos"], "capitulos")
        self._create_and_save_graph(bundles["livros"], "livros")
        self._create_and_save_graph(df_co, "coauthorship")
