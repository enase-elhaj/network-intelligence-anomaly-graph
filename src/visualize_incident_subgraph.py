import json
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from graph_model import load_graph


def load_incident_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


def subgraph_from_scope(G: nx.DiGraph, scope_nodes: list[str]) -> nx.DiGraph:
    return G.subgraph(scope_nodes).copy()


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # ---- Inputs
    report_path = repo_root / "data" / "processed" / "incident_report_2026-01-01T02-00-00.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing incident report: {report_path}")

    nodes_path = repo_root / "data" / "raw" / "nodes.csv"
    edges_path = repo_root / "data" / "raw" / "edges.csv"

    # ---- Load graph + report
    G = load_graph(nodes_path, edges_path)
    report = load_incident_report(report_path)

    scope_nodes = report["scope"]["nodes_in_scope"]
    anomalous_nodes = [x["node_id"] for x in report["anomalous_nodes_at_timestamp"]]
    root_cause = report["rankings"]["root_cause_top"][0]["node_id"]

    # ---- Subgraph
    H = subgraph_from_scope(G, scope_nodes)

    # ---- Layout
    # Spring layout is fine for small graphs; deterministic seed helps reproducibility.
    pos = nx.spring_layout(H, seed=42)

    # ---- Node styling rules
    node_sizes = []
    node_linewidths = []
    node_edgecolors = []
    node_colors = []

    for n in H.nodes:
        # Base size
        size = 1400
        lw = 1.5
        edgecolor = "black"

        # Root cause highlight (strongest)
        if n == root_cause:
            size = 2200
            lw = 3.5
            edgecolor = "black"

        # Node color categories
        if n in anomalous_nodes:
            color = "red"            # anomalous at timestamp
        else:
            color = "orange"         # in scope, impacted but not directly anomalous

        node_sizes.append(size)
        node_linewidths.append(lw)
        node_edgecolors.append(edgecolor)
        node_colors.append(color)

    # ---- Draw
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(
        H, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        alpha=0.95
    )

    # Edges
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=2, alpha=0.6)

    # Labels
    nx.draw_networkx_labels(H, pos, font_size=10, font_weight="bold")

    # Edge labels (relationship type)
    edge_labels = {(u, v): H[u][v].get("edge_type", "") for u, v in H.edges}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

    # Title + legend text (simple)
    ts = report["incident"]["timestamp"]
    plt.title(f"Incident Subgraph @ {ts}\nRoot cause: {root_cause} | Anomalous nodes in red", fontsize=12)
    plt.axis("off")

    # ---- Save
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "incident_subgraph_2026-01-01T02-00-00.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved graph visualization: {out_path}")


if __name__ == "__main__":
    main()
