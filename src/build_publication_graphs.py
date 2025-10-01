#!/usr/bin/env python3
"""
Main entry to build coauthorship graphs for Lattes across all scopes by default.

- Uses helper class in src/data/build_publication_graphs_class.py
- Utilities live in src/data/build_publication_graphs_utils.py
- Preserves original behavior and outputs:
  * data/processed/{scope}/graphs_weights/{type}_weighted.csv
  * data/processed/graphs/{scope}/{type}_graph.gexf
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

# Make 'src' a package root (requires __init__.py, see instructions below)
from data.build_publication_graphs_class import PublicationGraphBuilder
from data.build_publication_graphs_utils import SCOPES


def process_scope(scope: str, root: Path, verbose: bool = False) -> None:
    logging.info("=== Processing scope: %s ===", scope)
    builder = PublicationGraphBuilder(root=root, scope=scope, verbose=verbose)
    builder.run_full_pipeline()  # builds periodicos, eventos, capitulos, livros, and coauthorship


def main():
    parser = argparse.ArgumentParser(description="Build Lattes coauthorship graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope. If omitted, runs all scopes.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    root = Path(__file__).resolve().parents[1]  # project root
    if args.scope:
        process_scope(args.scope, root, args.verbose)
    else:
        for sc in SCOPES:
            try:
                process_scope(sc, root, args.verbose)
            except Exception as e:
                logging.error("Failed scope %s: %s", sc, e)


if __name__ == "__main__":
    main()
