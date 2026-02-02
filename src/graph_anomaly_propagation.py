
import pandas as pd
import networkx as nx
from pathlib import Path
from graph_model import load_graph


def get_incident_sources(anomalies: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Filter anomalies to only those occurring at the incident timestamp.
    Returns a DataFrame of anomalous nodes at that time.
    """
    current = anomalies[
        (anomalies["timestamp"] == timestamp) &
        (anomalies["is_anomaly"])
    ].copy()

    # Ensure node_id exists and anomaly_score is numeric
    if "node_id" not in current.columns or "anomaly_score" not in current.columns:
        raise ValueError("anomalies DataFrame must include 'node_id' and 'anomaly_score' columns.")

    return current


def incident_scope_nodes(G: nx.DiGraph, sources: list[str]) -> set[str]:
    """
    Incident scope = anomalous nodes + all downstream descendants (blast radius).
    """
    scope = set(sources)
    for s in sources:
        if s in G:
            scope.update(nx.descendants(G, s))
    return scope


def propagate_anomalies(
    G: nx.DiGraph,
    current_anomalies: pd.DataFrame,
    decay: float = 0.6,
) -> dict[str, float]:
    """
    Propagate anomaly impact downstream through the graph.
    We seed each source node with its anomaly_score and propagate to descendants
    with distance-based decay.

    Returns: impact[node] = total propagated impact at that node.
    """
    impact = {node: 0.0 for node in G.nodes}

    # Seed sources
    source_strength = {}
    for _, row in current_anomalies.iterrows():
        node = row["node_id"]
        strength = float(row["anomaly_score"])
        if node in G:
            impact[node] += strength
            source_strength[node] = strength  # keep original strength for propagation

    # Propagate from each source using its original strength
    for source, strength in source_strength.items():
        for target in nx.descendants(G, source):
            try:
                distance = nx.shortest_path_length(G, source, target)
            except nx.NetworkXNoPath:
                continue

            impact[target] += strength * (decay ** distance)

    return impact


def rank_impact(G: nx.DiGraph, impact: dict[str, float], scope: set[str]) -> list[tuple[str, float]]:
    """
    Rank nodes by impact severity (optionally scoped).
    """
    scored = [(n, impact.get(n, 0.0)) for n in scope]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def compute_root_cause_scores(
    G: nx.DiGraph,
    impact: dict[str, float],
    scope: set[str],
) -> list[tuple[str, float]]:
    """
    Root-cause score favors nodes that:
    - generate lots of downstream impact (within incident scope)
    - receive little upstream impact (within incident scope)

    score(node) = sum_downstream_impact - sum_upstream_impact
    computed ONLY inside 'scope' to avoid unrelated nodes.
    """
    scores = {}

    for node in scope:
        downstream_nodes = set(nx.descendants(G, node)) & scope
        upstream_nodes = set(nx.ancestors(G, node)) & scope

        downstream = sum(impact.get(t, 0.0) for t in downstream_nodes)
        upstream = sum(impact.get(s, 0.0) for s in upstream_nodes)

        scores[node] = downstream - upstream

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def blast_radius(G: nx.DiGraph, node: str, scope: set[str]) -> int:
    """Number of downstream nodes in scope affected by this node."""
    return len(set(nx.descendants(G, node)) & scope)

def top_downstream_impacted(G: nx.DiGraph, node: str, impact: dict[str, float], scope: set[str], k: int = 5):
    """Top-k downstream nodes by impact within scope."""
    downstream = list(set(nx.descendants(G, node)) & scope)
    ranked = sorted(downstream, key=lambda n: impact.get(n, 0.0), reverse=True)
    return ranked[:k]

def main():
    repo_root = Path(__file__).resolve().parents[1]

    G = load_graph(
        repo_root / "data" / "raw" / "nodes.csv",
        repo_root / "data" / "raw" / "edges.csv",
    )

    anomalies = pd.read_csv(repo_root / "data" / "processed" / "anomalies_isoforest.csv")
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])

    ts = pd.Timestamp("2026-01-01 02:00:00")

    # 1) Get incident sources (only anomalous nodes at this timestamp)
    current = get_incident_sources(anomalies, ts)
    sources = current["node_id"].unique().tolist()

    if len(sources) == 0:
        print(f"No anomalies at {ts}. Nothing to propagate.")
        return

    # 2) Define incident scope (sources + descendants)
    scope = incident_scope_nodes(G, sources)

    # 3) Propagate impacts using original anomaly scores
    impact = propagate_anomalies(G, current, decay=0.6)

    # Optional: keep only nodes with non-zero impact in scope (cleans output)
    scope_nonzero = {n for n in scope if impact.get(n, 0.0) > 0}

    # 4) Impact ranking (who is most affected)
    ranked_impact = rank_impact(G, impact, scope_nonzero)

    print(f"\nImpact ranking (most affected) for incident at {ts}:\n")
    for node, score in ranked_impact[:10]:
        print(f"{node:10s} | impact_score = {score:.3f}")

    # 5) Root-cause ranking (who likely caused it)
    root_causes = compute_root_cause_scores(G, impact, scope_nonzero)

    print(f"\nLikely root causes (incident-scoped) for incident at {ts}:\n")
    for node, score in root_causes[:10]:
        print(f"{node:10s} | root_cause_score = {score:.3f}")


    print(f"\nLikely root causes (incident-scoped) for incident at {ts}:\n")
    for node, score in root_causes[:5]:
        br = blast_radius(G, node, scope_nonzero)
        top5 = top_downstream_impacted(G, node, impact, scope_nonzero, k=5)
        print(f"{node:10s} | root_cause_score = {score:.3f} | blast_radius = {br:2d} | top_downstream = {top5}")


if __name__ == "__main__":
    main()

