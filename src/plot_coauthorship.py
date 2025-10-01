#!/usr/bin/env python3
"""
Main entry to plot coauthorship graph(s). By default, plots all scopes.

- Uses helper functions from src/viz/plot_coauthorship_utils.py
- Preserves original behavior:
  * Reads:  data/processed/graphs/{scope}/coauthorship_graph.gexf
  * Saves:  reports/analysis/{scope}/coauthorship_graph.png

Extras (new):
  * --min-weight, --min-degree, --largest-component, --limit-nodes, --layout, --iterations
  * If the graph is very large, auto-heuristics kick in to keep things fast.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from viz.plot_coauthorship_utils import (
    SCOPES,
    plot_one,
)


def main():
    parser = argparse.ArgumentParser(description="Plot coauthorship graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope to plot.")
    parser.add_argument("--gexf", default=None, help="Optional explicit path to a .gexf file.")
    parser.add_argument("--min-weight", type=int, default=None, help="Keep only edges with weight >= this.")
    parser.add_argument("--min-degree", type=int, default=None, help="Keep only nodes with degree >= this (after edge filter).")
    parser.add_argument("--largest-component", action="store_true", help="Keep only the largest connected component.")
    parser.add_argument("--limit-nodes", type=int, default=None, help="Keep only top-N nodes by degree (after filters).")
    parser.add_argument("--layout", choices=["auto", "sfdp", "spring", "kk", "random"], default="auto",
                        help="Layout algorithm. 'auto' tries sfdp (graphviz) and falls back to spring.")
    parser.add_argument("--iterations", type=int, default=None, help="Iterations for the spring layout.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]  # project root

    if args.gexf:
        gexf = Path(args.gexf)
        out = root / "reports" / "analysis" / "manual" / (gexf.stem + ".png")
        plot_one(
            gexf_path=gexf,
            out_png=out,
            min_weight=args.min_weight,
            min_degree=args.min_degree,
            largest_component=args.largest_component,
            limit_nodes=args.limit_nodes,
            layout=args.layout,
            iterations=args.iterations,
            dpi=args.dpi,
        )
        return

    scopes = [args.scope] if args.scope else SCOPES
    for sc in scopes:
        gexf_path = root / "data" / "processed" / "graphs" / sc / "coauthorship_graph.gexf"
        if not gexf_path.exists():
            print(f"[skip] Missing: {gexf_path} (did you run build_publication_graphs.py?)")
            continue
        out_png = root / "reports" / "analysis" / sc / "coauthorship_graph.png"
        plot_one(
            gexf_path=gexf_path,
            out_png=out_png,
            min_weight=args.min_weight,
            min_degree=args.min_degree,
            largest_component=args.largest_component,
            limit_nodes=args.limit_nodes,
            layout=args.layout,
            iterations=args.iterations,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
