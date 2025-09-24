# analysis_pipeline.py
"""
Exploratory analysis pipeline for coauthorship and Lattes applications datasets.

Structure:
    • DatasetAnalysisDatasetAnalysisConfig – global parameters
    • Utility functions (I/O, cleaning, wordcloud, plots, etc.)
    • Domain-specific analysis functions (general, areas, education, addresses, research lines, projects)
    • main() – entry point
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import logging
import math
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from wordcloud import WordCloud

# --------------------------------------------------------------------------- #
# Global configurations                                                       #
# --------------------------------------------------------------------------- #
class DatasetAnalysisConfig:
    DATA_DIR = Path("../data/processed/aplicacoes")
    GRAPH_PATH = Path("../graphs/coauthorship_graph.xml")
    OUTPUT_DIR = Path("../reports/analysis") 
    TOP_N = 5                  # top categories in plots
    WORDCLOUD_SIZE = (800, 400)
    WRAP_WIDTH = 12
    RANDOM_SEED = 42

def set_data_dir(path: str | Path):
    DatasetAnalysisConfig.DATA_DIR = Path(path).expanduser().resolve()

sns.set(style="whitegrid")       #  default aesthetics
tqdm.pandas()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def load_csv(name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Loads a CSV from the data folder and warns if the file does not exist."""
    path = DatasetAnalysisConfig.DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, **read_csv_kwargs)
    logging.info("✔️  %s loaded: %s rows × %s columns", name, *df.shape)
    return df


def fix_typo_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Renames columns according to `mapping`, fixing typos.
    Example: mapping = {"ANO-INICIO": "ANO-INÍCIO"}
    """
    return df.rename(columns=mapping)


def describe_dataframe(df: pd.DataFrame) -> None:
    """Prints shape, nulls and sample."""
    print(f"Shape: {df.shape}")
    display(df.isna().sum())
    display(df.head())


def value_counts_report(df: pd.DataFrame, columns: Sequence[str], top_n: int = 5):
    """Shows TOP N values per column."""
    print(f"\n=== DESCRIPTIVE ANALYSIS (TOP {top_n}) ===")
    for col in columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False).head(top_n))
        print("-" * 40)


def wordcloud_from_series(series: pd.Series, title: str | None = None, filename= "") -> None:
    freq = series.value_counts().to_dict()
    wc = WordCloud(width=DatasetAnalysisConfig.WORDCLOUD_SIZE[0],
                   height=DatasetAnalysisConfig.WORDCLOUD_SIZE[1],
                   background_color="white").generate_from_frequencies(freq)

    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    if filename:  # <<< ADDED
        save_path = DatasetAnalysisConfig.OUTPUT_DIR / f"{filename}.png"
        plt.savefig(save_path, bbox_inches="tight")
        logging.info("📁 WordCloud saved at: %s", save_path)
    plt.show()


# ------------------------------ plots -------------------------------------- #
def _wrap_labels(label: str, width: int = DatasetAnalysisConfig.WRAP_WIDTH) -> str:
    return "\n".join(textwrap.wrap(label, width=width))


from matplotlib.ticker import FixedLocator

def barplot_topn(
    df: pd.DataFrame,
    category_col: str | Sequence[str],
    total_rows: int,
    top_n: int = DatasetAnalysisConfig.TOP_N,
    hue: str | None = None,
    palette: str | list[str] | None = "Set2",
    title: str | None = None,
    filename: str = "" 
) -> None:
    """
    Displays a barplot of the TOP N categories in `category_col`.
    Accepts single columns or tuples (e.g.: ["CITY", "STATE"]).
    """
    if isinstance(category_col, str):
        group_cols = [category_col]
    else:
        group_cols = list(category_col)

    df_grouped = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    df_grouped["proportion"] = df_grouped["count"] / total_rows

    plt.figure()
    ax = sns.barplot(
        data=df_grouped,
        x=group_cols[-1],        # last field on the horizontal axis
        y="proportion",
        hue=hue or group_cols[0],
        palette=palette,
    )
    if ax.get_legend():
        ax.get_legend().remove()

    plt.title(title or f"TOP {top_n} – {', '.join(group_cols)}")
    plt.xlabel(group_cols[-1])
    plt.ylabel("Proportion")

    # Fixes the warning: explicitly set ticks before setting labels
    ticks = ax.get_xticks()
    labels = [_wrap_labels(lbl.get_text()) for lbl in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels, rotation=45)

    # count above each bar
    for patch, (_, row) in zip(ax.patches, df_grouped.iterrows()):
        ax.annotate(
            f'{int(row["count"])}',
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center", va="center",
            xytext=(0, 5), textcoords="offset points",
        )
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    if filename:  # <<< ADDED
        save_path = DatasetAnalysisConfig.OUTPUT_DIR / f"{filename}.png"
        plt.savefig(save_path, bbox_inches="tight")
        logging.info("📁 Plot saved at: %s", save_path)
    plt.show()


def lineplot_timeline(
    df: pd.DataFrame,
    year_col: str,
    count_col: str,
    color: str,
    title: str,
) -> None:
    plt.figure()
    ax = sns.lineplot(
        data=df,
        x=year_col,
        y=count_col,
        marker="o",
        sort=True,
        color=color,
    )
    plt.title(title)
    plt.xlabel(year_col.replace("-", " ").title())
    plt.ylabel("Quantity")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    for _, row in df.iterrows():
        plt.text(row[year_col], row[count_col] + 0.5, int(row[count_col]), ha="center")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Specific analyses                                                           #
# --------------------------------------------------------------------------- #
def analyze_gerais(df_pesq: pd.DataFrame):
    cols = df_pesq.columns.tolist()
    describe_dataframe(df_pesq)
    value_counts_report(df_pesq, cols, DatasetAnalysisConfig.TOP_N)

    barplot_topn(
        df_pesq,
        ["CIDADE-NASCIMENTO", "UF-NASCIMENTO"],
        len(df_pesq),
        title="(City, State) Most Frequent",
        palette="viridis",
        filename="cidade_uf_nascimento",
    )
    barplot_topn(
        df_pesq,
        "UF-NASCIMENTO",
        len(df_pesq),
        title="Most Frequent States",
        palette="coolwarm",
        filename="uf_nascimento",
    )
    barplot_topn(
        df_pesq,
        "PAIS-DE-NASCIMENTO",
        len(df_pesq),
        title="Most Frequent Countries",
        palette="Set2",
        filename="pais_nascimento",
    )

    wordcloud_from_series(df_pesq["PAIS-DE-NASCIMENTO"], "WordCloud – Country", filename="pais_nascimento_wc")


def analyze_areas(df_areas: pd.DataFrame):
    cols_to_analyze = [c for c in df_areas.columns if c != "LattesID"]
    describe_dataframe(df_areas)
    value_counts_report(df_areas, cols_to_analyze)

    wordcloud_from_series(
        df_areas["NOME-GRANDE-AREA-DO-CONHECIMENTO"],
        "WordCloud – Major Area",
        filename="grande_area_wc",
    )

    for col in cols_to_analyze:
        barplot_topn(
            df_areas,
            col,
            len(df_areas),
            title=f"Categories – {col}",
            palette="Set2",
        )


def analyze_formacoes(df_forms: pd.DataFrame):
    cols_to_analyze = [c for c in df_forms.columns if c != "LattesID"]
    describe_dataframe(df_forms)
    value_counts_report(df_forms, cols_to_analyze)
    wordcloud_from_series(df_forms["NOME-INSTITUICAO"], "WordCloud – Institutions", filename="instituicoes_wc")

    for col in cols_to_analyze:
        barplot_topn(
            df_forms,
            col,
            len(df_forms),
            title=f"Categories – {col}",
            palette="Set2",
        )


def analyze_enderecos(df_end: pd.DataFrame):
    describe_dataframe(df_end)
    columns = ["NOME-INSTITUICAO-EMPRESA", "NOME-ORGAO"]
    value_counts_report(df_end, columns)
    wordcloud_from_series(
        df_end["NOME-INSTITUICAO-EMPRESA"],
        "WordCloud – Institutions/Companies",
        filename="instituicoes_empresas_wc",
    )

    barplot_topn(
        df_end,
        "NOME-INSTITUICAO-EMPRESA",
        len(df_end),
        title="Most Frequent Institutions",
        palette="Set2",
    )
    barplot_topn(
        df_end,
        ["NOME-INSTITUICAO-EMPRESA", "NOME-ORGAO"],
        len(df_end),
        title="(Institution, Organization) Most Frequent",
        palette="viridis",
    )


def analyze_linhas(df_linhas: pd.DataFrame):
    describe_dataframe(df_linhas)
    col = "TITULO-DA-LINHA-DE-PESQUISA"
    value_counts_report(df_linhas, [col])
    wordcloud_from_series(df_linhas[col], "WordCloud – Research Lines")

    barplot_topn(
        df_linhas,
        col,
        len(df_linhas),
        title="Research Lines – Top",
        palette="Set2",
    )


def analyze_projetos(df_proj: pd.DataFrame):
    describe_dataframe(df_proj)
    value_counts_report(df_proj, ["NATUREZA", "SITUACAO"], top_n=3)
    wordcloud_from_series(df_proj["NOME-INSTITUICAO"], "WordCloud – Institution")

    # Nature / Status
    barplot_topn(df_proj, "NATUREZA", len(df_proj), top_n=3, palette="Set2")
    barplot_topn(df_proj, "SITUACAO", len(df_proj), top_n=3, palette="coolwarm")

    # Temporal analyses
    for col, color, label in [
        ("ANO-INICIO", "blue", "Projects Started"),
        ("ANO-FIM", "green", "Projects Finished"),
    ]:
        df_year = (
            df_proj.dropna(subset=[col])
            .groupby(col)
            .size()
            .reset_index(name="count")
            .sort_values(col)
        )
        lineplot_timeline(df_year, col, "count", color, label)


# --------------------------------------------------------------------------- #
# main – orchestrates the pipeline                                            #
# --------------------------------------------------------------------------- #
def main():
    """Runs all analyses end-to-end."""
    logging.info("📊 Starting analysis pipeline…")
    
    DatasetAnalysisConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

    # --- graphs ------------------------------------------------------------- #
    try:
        G = nx.read_graphml(DatasetAnalysisConfig.GRAPH_PATH)
        logging.info("✔️  Graph loaded (%s nodes)", G.number_of_nodes())
    except FileNotFoundError:
        logging.warning("⚠️  Graph not found at %s – skipping graph analysis",
                        DatasetAnalysisConfig.GRAPH_PATH)


    for dataset in ['aplicacoes', 'restritivo', 'abrangente']:
        # --- DataFrames ----------------------------------------------------- #
        df_pesq = load_csv("gerais")
        df_areas = load_csv("areas")
        df_forms = load_csv("formacoes")
        df_end   = load_csv("enderecos")
        df_linhas = load_csv("linhas")
        df_proj  = load_csv("projetos")

        # --- Run analyses --------------------------------------------------- #
        analyze_gerais(df_pesq)
        analyze_areas(df_areas)
        analyze_formacoes(df_forms)
        analyze_enderecos(df_end)
        analyze_linhas(df_linhas)
        analyze_projetos(df_proj)

        logging.info("✅ Pipeline finished!")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Runs only if the script is called directly
    main()
