# Network Analysis Lattes 

This project models the Brazilian academic network Lattes as graphs:
- Nodes: researchers (LattesID_*)
- Edges: co-authorship (journals, events, book chapters, books) and supervision (advisor → advisee)

The pipeline reads CSVs in data/processed/{scope}/ and produces:
- Weighted co-authorship tables (*_weighted.csv)
- Graphs in GEXF (Gephi-compatible) under data/processed/graphs/{scope}/
- Quick-look PNGs under reports/analysis/{scope}/

All scripts run automatically for the three scopes: abrangente, restritivo, and aplicacoes.

## Requirements

- Recommended: Graphviz + pygraphviz for faster layouts (optional)
  - Ubuntu/Debian: sudo apt-get install graphviz graphviz-dev

## Installation (with uv)

Install project dependencies:

```
uv sync
```

## Data Acquisition & Preparation 

1. Download the Dataset

Access the data via Google Drive:
https://drive.google.com/drive/folders/15l9MfqX2i0WNELkWVCTkmUDjSzfsSEt8?usp=sharing

_Note: Skip downloading files from the cabecalhos (headers) folder._

2. Extract Files

Unzip all downloaded directories into data/raw/.

Ensure the extracted folders are at the same level as the headers directory.

3. Prepare Data

Run the header-adding notebook to normalize the CSV schemas:

- Open and execute all cells in src/data/add_header.ipynb.

_This notebook may be converted to a Python script in future updates._

## Building Graphs (batch for the 3 scopes)

Run these from the repository root.

1. Publications / Co-authorship graphs

```
python3 src/build_publication_graphs.py
```

Outputs (per scope):

```
data/processed/{scope}/graphs_weights/
  - periodicos_weighted.csv
  - eventos_weighted.csv
  - capitulos_weighted.csv
  - livros_weighted.csv
  - coauthorship_weighted.csv

data/processed/graphs/{scope}/
  - periodicos_graph.gexf
  - eventos_graph.gexf
  - capitulos_graph.gexf
  - livros_graph.gexf
  - coauthorship_graph.gexf
```

2) Supervision (advisor → advisee) graphs

```
python3 src/build_orientacoes_graph.py
```

```
Outputs (per scope):

data/processed/{scope}/
  - orientacoes_edges.csv        # directed edges Source -> Target

data/processed/graphs/{scope}/
  - orientacoes.gexf
```

## Plotting Co-authorship Graphs

Produce a PNG for each scope from the combined co-authorship GEXF:

```
python3 src/plot_coauthorship.py
```

Outputs (per scope):

```
reports/analysis/{scope}/
  - coauthorship_graph.png
```


## Dataset Exploration & Charts (from the original README)

1. Open and run notebooks/generic_dataset_analysis.ipynb.

2. Charts are saved automatically in reports/analysis/.


## Additional Notes 

-  Ensure all dependencies are installed before running notebooks or scripts.
- For questions or issues, please open an issue in this repository.
