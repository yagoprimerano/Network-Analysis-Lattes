#!/usr/bin/env python3
"""
Plot the coauthorship graph(s). By default, plots all scopes:
- Reads: data/processed/graphs/{scope}/coauthorship_graph.gexf
- Saves: reports/analysis/{scope}/coauthorship_graph.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx

SCOPES = ["abrangente", "restritivo", "aplicacoes"]


def plot_one(gexf_path: Path, out_png: Path) -> None:
    G = nx.read_gexf(gexf_path)
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1, seed=42)
    edges = list(G.edges(data=True))
    weights = [e[2].get("weight", 1) for e in edges]
    nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges],
                           width=[w * 0.2 for w in weights], alpha=0.5)
    plt.axis("off")
    plt.title("Coauthorship Graph")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot coauthorship graphs (all scopes by default).")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Optional single scope to plot.")
    parser.add_argument("--gexf", default=None, help="Optional explicit path to a .gexf file.")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    if args.gexf:
        gexf = Path(args.gexf)
        if not gexf.exists():
            sys.exit(f"GEXF not found: {gexf}")
        out = ROOT / "reports" / "analysis" / "manual" / (gexf.stem + ".png")
        plot_one(gexf, out)
        return

    scopes = [args.scope] if args.scope else SCOPES
    for sc in scopes:
        gexf_path = ROOT / "data" / "processed" / "graphs" / sc / "coauthorship_graph.gexf"
        if not gexf_path.exists():
            print(f"[skip] Missing: {gexf_path} (did you run build_publication_graphs.py?)")
            continue
        out_png = ROOT / "reports" / "analysis" / sc / "coauthorship_graph.png"
        plot_one(gexf_path, out_png)


if __name__ == "__main__":
    main()
