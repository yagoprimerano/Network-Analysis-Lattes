"""
Utilities for plotting coauthorship graphs.

Fast plotting strategies included:
- Edge weight threshold
- Node degree threshold
- Keep only largest connected component
- Keep only top-N nodes by degree
- Try Graphviz's sfdp layout if available (pygraphviz + graphviz), fallback to spring
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

SCOPES = ["abrangente", "restritivo", "aplicacoes"]


def _try_graphviz_layout(G: nx.Graph, prog: str = "sfdp"):
    """Return Graphviz layout if pygraphviz/graphviz are installed, else None."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        return None


def _auto_heuristics(G: nx.Graph) -> Tuple[int, int, bool, int, int]:
    """
    Decide sensible defaults for very large graphs.
    Returns: (min_weight, min_degree, largest_component, limit_nodes, iterations)
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    # Baseline
    min_weight = 1
    min_degree = 0
    largest_component = False
    limit_nodes = None
    iterations = 50

    # Heuristics for speed
    if n > 8000 or m > 80000:
        min_weight = 2
        largest_component = True
        iterations = 30
    if n > 20000 or m > 150000:
        min_weight = max(min_weight, 3)
        largest_component = True
        iterations = 25
        limit_nodes = 8000
    if n > 40000 or m > 300000:
        min_weight = max(min_weight, 4)
        largest_component = True
        iterations = 20
        limit_nodes = 5000
    return min_weight, min_degree, largest_component, limit_nodes, iterations


def _filter_graph(
    G: nx.Graph,
    min_weight: Optional[int],
    min_degree: Optional[int],
    largest_component: bool,
    limit_nodes: Optional[int],
) -> nx.Graph:
    """Apply edge and node filters and return a new subgraph (copy=False for speed)."""
    H = G

    # Edge filter by weight
    if min_weight and min_weight > 1:
        to_remove = [(u, v) for u, v, d in H.edges(data=True) if d.get("weight", 1) < min_weight]
        if to_remove:
            H = H.copy()
            H.remove_edges_from(to_remove)

    # Keep only largest connected component (after edge filtering)
    if largest_component and len(H) > 0:
        components = list(nx.connected_components(H))
        if len(components) > 1:
            biggest = max(components, key=len)
            H = H.subgraph(biggest).copy()

    # Node degree threshold
    if min_degree and min_degree > 0:
        deg = dict(H.degree())
        keep = {n for n, d in deg.items() if d >= min_degree}
        H = H.subgraph(keep).copy()

    # Top-N nodes by degree
    if limit_nodes and limit_nodes > 0 and len(H) > limit_nodes:
        deg = sorted(H.degree, key=lambda kv: kv[1], reverse=True)
        keep = set([n for n, _ in deg[:limit_nodes]])
        H = H.subgraph(keep).copy()

    return H


def _compute_layout(
    H: nx.Graph,
    layout: str,
    iterations: int,
) -> dict:
    """Compute positions with the requested layout (or auto-choose)."""
    if layout == "auto":
        # Prefer fast graphviz sfdp if available
        pos = _try_graphviz_layout(H, prog="sfdp")
        if pos is not None:
            return pos
        layout = "spring"  # fallback

    if layout == "sfdp":
        pos = _try_graphviz_layout(H, prog="sfdp")
        if pos is not None:
            return pos
        print("[info] 'sfdp' requested but pygraphviz/graphviz not found. Falling back to 'spring'.")

    if layout == "spring":
        # k ~ 1/sqrt(n) is a good default scale for large graphs
        n = max(len(H), 1)
        k = 1.0 / (n ** 0.5)
        return nx.spring_layout(H, k=k, iterations=iterations, seed=42)
    elif layout == "kk":
        return nx.kamada_kawai_layout(H)
    elif layout == "random":
        return nx.random_layout(H, seed=42)
    else:
        # default to spring if unknown
        return nx.spring_layout(H, iterations=iterations, seed=42)


def plot_one(
    gexf_path: Path,
    out_png: Path,
    min_weight: Optional[int] = None,
    min_degree: Optional[int] = None,
    largest_component: bool = False,
    limit_nodes: Optional[int] = None,
    layout: str = "auto",
    iterations: Optional[int] = None,
    dpi: int = 300,
) -> None:
    """Load GEXF, optionally filter/limit, lay out and save a PNG."""
    print(f"[load] {gexf_path}")
    G = nx.read_gexf(gexf_path)

    # Auto speed heuristics if the user didn't set filters
    if min_weight is None or min_degree is None or (not largest_component) or (limit_nodes is None) or (iterations is None):
        auto_w, auto_deg, auto_lcc, auto_limit, auto_iter = _auto_heuristics(G)
        min_weight = auto_w if min_weight is None else min_weight
        min_degree = auto_deg if min_degree is None else min_degree
        largest_component = auto_lcc or largest_component
        limit_nodes = auto_limit if limit_nodes is None else limit_nodes
        iterations = auto_iter if iterations is None else iterations

    print(f"[info] nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"| min_weight={min_weight} min_degree={min_degree} lcc={largest_component} "
          f"limit_nodes={limit_nodes} iterations={iterations} layout={layout}")

    H = _filter_graph(
        G,
        min_weight=min_weight,
        min_degree=min_degree,
        largest_component=largest_component,
        limit_nodes=limit_nodes,
    )

    print(f"[info] filtered: nodes={H.number_of_nodes()} edges={H.number_of_edges()}")

    if H.number_of_nodes() == 0:
        print("[warn] Graph is empty after filtering. Skipping draw.")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Empty graph after filtering", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[save] {out_png}")
        return

    # Compute layout
    pos = _compute_layout(H, layout=layout, iterations=iterations)

    # Draw
    plt.figure(figsize=(12, 12))
    edges = list(H.edges(data=True))
    weights = [max(int(d.get("weight", 1)), 1) for _, _, d in edges]
    widths = [0.2 * (w ** 0.5) for w in weights]  # sublinear widths

    nx.draw_networkx_nodes(H, pos, node_size=20, alpha=0.7)
    nx.draw_networkx_edges(H, pos, edgelist=[(u, v) for u, v, _ in edges],
                           width=widths, alpha=0.5)
    plt.axis("off")
    plt.title("Coauthorship Graph")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_png}")
