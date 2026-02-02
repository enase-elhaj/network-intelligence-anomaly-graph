import pandas as pd
import networkx as nx
from pathlib import Path


def load_graph(nodes_path: Path, edges_path: Path) -> nx.DiGraph:
    """
    Build a directed graph of the infrastructure.
    Nodes: devices, servers, services
    Edges: CONNECTED_TO, HOSTS, DEPENDS_ON
    """
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in nodes.iterrows():
        G.add_node(
            row["node_id"],
            node_type=row["node_type"],
            site=row["site"],
            criticality=row["criticality"],
        )

    # Add edges with relationship type
    for _, row in edges.iterrows():
        G.add_edge(
            row["src"],
            row["dst"],
            edge_type=row["edge_type"],
        )

    return G


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    G = load_graph(
        repo_root / "data" / "raw" / "nodes.csv",
        repo_root / "data" / "raw" / "edges.csv",
    )

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
