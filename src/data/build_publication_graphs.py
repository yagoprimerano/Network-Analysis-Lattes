#!/usr/bin/env python3
"""
Build coauthorship graphs for the Lattes dataset across all scopes by default.

Default behavior:
- Runs for scopes: ['abrangente', 'restritivo', 'aplicacoes'].
- Saves weighted edges CSVs to data/processed/{scope}/graphs_weights/{type}_weighted.csv
- Saves graphs (GEXF) to data/processed/graphs/{scope}/{type}_graph.gexf

You can still run only one scope with:  python src/data/build_publication_graphs.py --scope aplicacoes
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import unidecode


SCOPES = ["abrangente", "restritivo", "aplicacoes"]


# --------------------------- Utils & IO -------------------------------------- #

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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cluster_by_year_blocks(indexes: pd.Index, years: pd.Series, next_id: int, mapping: Dict[int, int]) -> int:
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


# --------------------------- Per-set clustering ------------------------------ #

def cluster_books(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Books: ISBN → standardized title + year±1 -> returns df with 'book_id' and author-work pairs."""
    df = df.copy()
    if "STANDARDIZE-TITULO-DO-LIVRO" not in df.columns:
        df["STANDARDIZE-TITULO-DO-LIVRO"] = df["TITULO-DO-LIVRO"].astype(str).apply(standardize_string)
    df["std_isbn"] = df["ISBN"].fillna("").apply(standardize_string)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")

    clusters, next_id = {}, 0
    for isbn, grp in df[df["std_isbn"] != ""].groupby("std_isbn", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_code = df[df["std_isbn"] == ""]
    for title, grp in no_code.groupby("STANDARDIZE-TITULO-DO-LIVRO", sort=False):
        next_id = _cluster_by_year_blocks(grp.index, grp["ANO"], next_id, clusters)

    df["book_id"] = df.index.map(clusters)
    pairs = df[["LattesID", "book_id"]].copy()
    return df, pairs


def cluster_periodicos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Journals: DOI → ISSN → (std title + std journal + year±1)."""
    df = df.copy()
    df["TITULO-DO-ARTIGO"] = df["TITULO-DO-ARTIGO"].astype(str).apply(standardize_string)
    df["TITULO-DO-PERIODICO-OU-REVISTA"] = df["TITULO-DO-PERIODICO-OU-REVISTA"].astype(str).apply(standardize_string)
    df["std_doi"] = df["DOI"].fillna("").apply(standardize_string)
    df["std_issn"] = df["ISSN"].fillna("").apply(standardize_string)
    df["ANO-DO-ARTIGO"] = pd.to_numeric(df["ANO-DO-ARTIGO"], errors="coerce")

    clusters, next_id = {}, 0
    for doi, grp in df[df["std_doi"] != ""].groupby("std_doi", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_doi = df[df["std_doi"] == ""]
    for issn, grp in no_doi[no_doi["std_issn"] != ""].groupby("std_issn", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_code = df[(df["std_doi"] == "") & (df["std_issn"] == "")]
    for (art, journ), grp in no_code.groupby(["TITULO-DO-ARTIGO", "TITULO-DO-PERIODICO-OU-REVISTA"], sort=False):
        next_id = _cluster_by_year_blocks(grp.index, grp["ANO-DO-ARTIGO"], next_id, clusters)

    df["paper_id"] = df.index.map(clusters)
    pairs = df[["LattesID", "paper_id"]].copy()
    return df, pairs


def cluster_capitulos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    for isbn, grp in df[df["std_isbn"] != ""].groupby("std_isbn", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_code = df[df["std_isbn"] == ""]
    for (chap, book), grp in no_code.groupby(["TITULO-DO-CAPITULO-DO-LIVRO", "TITULO-DO-LIVRO"], sort=False):
        next_id = _cluster_by_year_blocks(grp.index, grp["ANO"], next_id, clusters)

    df["chapter_id"] = df.index.map(clusters)
    pairs = df[["LattesID", "chapter_id"]].copy()
    return df, pairs


def cluster_eventos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    for doi, grp in df[df["std_doi"] != ""].groupby("std_doi", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_doi = df[df["std_doi"] == ""]
    for isbn, grp in no_doi[no_doi["std_isbn"] != ""].groupby("std_isbn", sort=False):
        for idx in grp.index:
            clusters[idx] = next_id
        next_id += 1

    no_code = df[(df["std_doi"] == "") & (df["std_isbn"] == "")]
    for (title, ev), grp in no_code.groupby(["STANDARDIZE-TITULO-DO-TRABALHO", "STANDARDIZE-NOME-DO-EVENTO"], sort=False):
        next_id = _cluster_by_year_blocks(grp.index, grp[year_col], next_id, clusters)

    df["paper_id"] = df.index.map(clusters)
    pairs = df[["LattesID", "paper_id"]].copy()
    return df, pairs


# --------------------------- Graph building ---------------------------------- #

def create_and_save_graph(df_pairs: pd.DataFrame, set_type: str, weights_dir: Path, graphs_dir: Path) -> nx.Graph:
    """Build an undirected weighted coauthorship graph from (LattesID, work_id) pairs."""
    # Identify coauthor pairs per work
    pairs = []
    gp = df_pairs.groupby("work_id")
    for work_id, group in tqdm(gp, total=len(gp), desc=f"Pairs[{set_type}]"):
        users = list(group["LattesID"])
        if len(users) > 1:
            pairs.extend(combinations(users, 2))

    co_df = pd.DataFrame(pairs, columns=["author1", "author2"])
    if co_df.empty:
        G = nx.Graph()
        ensure_dir(weights_dir)
        co_df.assign(count=pd.Series(dtype=int)).to_csv(weights_dir / f"{set_type}_weighted.csv", index=False)
        ensure_dir(graphs_dir)
        nx.write_gexf(G, graphs_dir / f"{set_type}_graph.gexf")
        return G

    co_df["count"] = 1
    coauthor_weighted = co_df.groupby(["author1", "author2"], as_index=False)["count"].sum()

    ensure_dir(weights_dir)
    coauthor_weighted.to_csv(weights_dir / f"{set_type}_weighted.csv", index=False)

    G = nx.Graph()
    for _, row in tqdm(coauthor_weighted.iterrows(), total=len(coauthor_weighted), desc=f"Graph[{set_type}]"):
        G.add_edge(row["author1"], row["author2"], weight=int(row["count"]))

    ensure_dir(graphs_dir)
    nx.write_gexf(G, graphs_dir / f"{set_type}_graph.gexf")
    return G


def build_coauthorship_inputs(df_user_per: pd.DataFrame,
                              df_user_eventos: pd.DataFrame,
                              df_user_cap: pd.DataFrame,
                              df_user_book: pd.DataFrame,
                              df_user: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Normalize ids, prefix work types, filter to researchers present in 'gerais', and combine."""
    # Ensure strings
    if "paper_id" in df_user_per.columns:
        df_user_per["paper_id"] = df_user_per["paper_id"].astype(str)
    if "paper_id" in df_user_eventos.columns:
        df_user_eventos["paper_id"] = df_user_eventos["paper_id"].astype(str)
    if "chapter_id" in df_user_cap.columns:
        df_user_cap["chapter_id"] = df_user_cap["chapter_id"].astype(str)
    if "book_id" in df_user_book.columns:
        df_user_book["book_id"] = df_user_book["book_id"].astype(str)

    # Prefix work ids
    df_user_per = df_user_per.rename(columns={"paper_id": "work_id"}).copy()
    df_user_per["work_id"] = df_user_per["work_id"].apply(lambda x: f"periodicos_{x}")

    df_user_eventos = df_user_eventos.rename(columns={"paper_id": "work_id"}).copy()
    df_user_eventos["work_id"] = df_user_eventos["work_id"].apply(lambda x: f"eventos_{x}")

    df_user_cap = df_user_cap.rename(columns={"chapter_id": "work_id"}).copy()
    df_user_cap["work_id"] = df_user_cap["work_id"].apply(lambda x: f"capitulos_{x}")

    df_user_book = df_user_book.rename(columns={"book_id": "work_id"}).copy()
    df_user_book["work_id"] = df_user_book["work_id"].apply(lambda x: f"livros_{x}")

    # Prefix researchers and filter to those present in gerais
    valid_ids = set(df_user["LattesID"].apply(lambda x: f"LattesID_{x}"))
    out = []
    for df in (df_user_per, df_user_eventos, df_user_cap, df_user_book):
        df = df.copy()
        df["LattesID"] = df["LattesID"].apply(lambda x: f"LattesID_{x}")
        df = df[df["LattesID"].isin(valid_ids)]
        df.drop_duplicates(inplace=True)
        out.append(df)

    df_user_per, df_user_eventos, df_user_cap, df_user_book = out
    df_co = pd.concat(out, axis=0, ignore_index=True).drop_duplicates()

    return {
        "periodicos": df_user_per,
        "eventos": df_user_eventos,
        "capitulos": df_user_cap,
        "livros": df_user_book,
        "coauthorship": df_co,
    }


# --------------------------- Orchestrator ------------------------------------ #

def process_scope(scope: str, ROOT: Path) -> None:
    logging.info("=== Processing scope: %s ===", scope)
    DATA_DIR = ROOT / "data"
    weights_dir = DATA_DIR / "processed" / scope / "graphs_weights"
    graphs_dir = DATA_DIR / "processed" / "graphs" / scope

    # Read datasets
    df_user = read_dataset(DATA_DIR, scope, "gerais")
    df_book_raw = read_dataset(DATA_DIR, scope, "livros")
    df_per = read_dataset(DATA_DIR, scope, "periodicos")
    df_cap = read_dataset(DATA_DIR, scope, "capitulos")
    df_eventos = read_dataset(DATA_DIR, scope, "eventos")

    # Cluster & pairs
    df_book, df_user_book = cluster_books(df_book_raw)
    df_per, df_user_per = cluster_periodicos(df_per)
    df_cap, df_user_cap = cluster_capitulos(df_cap)
    df_eventos, df_user_eventos = cluster_eventos(df_eventos)

    bundles = build_coauthorship_inputs(df_user_per, df_user_eventos, df_user_cap, df_user_book, df_user)

    # Graphs
    create_and_save_graph(bundles["periodicos"], "periodicos", weights_dir, graphs_dir)
    create_and_save_graph(bundles["eventos"], "eventos", weights_dir, graphs_dir)
    create_and_save_graph(bundles["capitulos"], "capitulos", weights_dir, graphs_dir)
    create_and_save_graph(bundles["livros"], "livros", weights_dir, graphs_dir)
    create_and_save_graph(bundles["coauthorship"], "coauthorship", weights_dir, graphs_dir)

    logging.info("Saved graphs to: %s", graphs_dir)
    logging.info("Saved edge weights to: %s", weights_dir)


def main():
    parser = argparse.ArgumentParser(description="Build Lattes coauthorship graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope. If omitted, runs all scopes.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    ROOT = Path(__file__).resolve().parents[2]

    if args.scope:
        process_scope(args.scope, ROOT)
    else:
        for sc in SCOPES:
            try:
                process_scope(sc, ROOT)
            except Exception as e:
                logging.error("Failed scope %s: %s", sc, e)


if __name__ == "__main__":
    main()