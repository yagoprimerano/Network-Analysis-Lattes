#!/usr/bin/env python3
"""
Main entry to build directed supervision graphs (advisor -> advisee) for all scopes by default.

- Uses helper class in src/data/build_orientacoes_graph_class.py
- Utilities live in src/data/build_orientacoes_graph_utils.py
- Preserves original behavior and outputs:
  * data/processed/{scope}/orientacoes_edges.csv
  * data/processed/graphs/{scope}/orientacoes.gexf
"""

from __future__ import annotations
import argparse
from pathlib import Path

from data.build_orientacoes_graph_class import OrientacoesGraphBuilder
from data.build_orientacoes_graph_utils import SCOPES


def process_scope(scope: str, root: Path) -> None:
    builder = OrientacoesGraphBuilder(root=root, scope=scope)
    builder.run_full_pipeline()  # builds CSV edges + directed GEXF


def main():
    parser = argparse.ArgumentParser(description="Build supervision graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope. If omitted, runs all scopes.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]  # project root
    if args.scope:
        process_scope(args.scope, root)
    else:
        for sc in SCOPES:
            try:
                process_scope(sc, root)
            except Exception as e:
                print(f"[ERROR][{sc}] {e}")


if __name__ == "__main__":
    main()
