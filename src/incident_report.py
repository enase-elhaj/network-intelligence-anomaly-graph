import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import networkx as nx

from graph_model import load_graph


def get_incident_sources(anomalies: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Anomalous nodes at incident timestamp."""
    current = anomalies[
        (anomalies["timestamp"] == timestamp) &
        (anomalies["is_anomaly"])
    ].copy()
    return current


def incident_scope_nodes(G: nx.DiGraph, sources: list[str]) -> set[str]:
    """Scope = sources + all descendants (blast radius)."""
    scope = set(sources)
    for s in sources:
        if s in G:
            scope.update(nx.descendants(G, s))
    return scope


def propagate_anomalies(G: nx.DiGraph, current_anomalies: pd.DataFrame, decay: float = 0.6) -> dict[str, float]:
    """
    Seed each source with its anomaly_score, then propagate downstream with decay^distance.
    Returns impact dict for all nodes in graph.
    """
    impact = {node: 0.0 for node in G.nodes}

    source_strength = {}
    for _, row in current_anomalies.iterrows():
        node = row["node_id"]
        strength = float(row["anomaly_score"])
        if node in G:
            impact[node] += strength
            source_strength[node] = strength

    for source, strength in source_strength.items():
        for target in nx.descendants(G, source):
            try:
                dist = nx.shortest_path_length(G, source, target)
            except nx.NetworkXNoPath:
                continue
            impact[target] += strength * (decay ** dist)

    return impact


def rank_impact(impact: dict[str, float], scope: set[str]) -> list[tuple[str, float]]:
    ranked = [(n, impact.get(n, 0.0)) for n in scope]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def compute_root_cause_scores(G: nx.DiGraph, impact: dict[str, float], scope: set[str]) -> list[tuple[str, float]]:
    """
    Root cause score = downstream_sum(impact) - upstream_sum(impact), computed within scope.
    """
    scores = {}
    for node in scope:
        downstream_nodes = set(nx.descendants(G, node)) & scope
        upstream_nodes = set(nx.ancestors(G, node)) & scope

        downstream = sum(impact.get(t, 0.0) for t in downstream_nodes)
        upstream = sum(impact.get(s, 0.0) for s in upstream_nodes)

        scores[node] = downstream - upstream

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def blast_radius(G: nx.DiGraph, node: str, scope: set[str]) -> int:
    """How many downstream nodes in scope are affected by this candidate."""
    return len(set(nx.descendants(G, node)) & scope)


def top_downstream_impacted(G: nx.DiGraph, node: str, impact: dict[str, float], scope: set[str], k: int = 5) -> list[str]:
    downstream = list(set(nx.descendants(G, node)) & scope)
    downstream.sort(key=lambda n: impact.get(n, 0.0), reverse=True)
    return downstream[:k]


def safe_ts_for_filename(ts: pd.Timestamp) -> str:
    # Example: 2026-01-01T02-00-00
    return ts.strftime("%Y-%m-%dT%H-%M-%S")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    G = load_graph(
        repo_root / "data" / "raw" / "nodes.csv",
        repo_root / "data" / "raw" / "edges.csv",
    )

    # Load node metadata for report enrichment
    nodes_df = pd.read_csv(repo_root / "data" / "raw" / "nodes.csv").set_index("node_id")

    # Load anomalies
    anomalies = pd.read_csv(processed_dir / "anomalies_isoforest.csv")
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])

    # Pick incident timestamp
    ts = pd.Timestamp("2026-01-01 02:00:00")

    current = get_incident_sources(anomalies, ts)
    sources = current["node_id"].unique().tolist()

    if len(sources) == 0:
        print(f"No anomalies at {ts}. No report generated.")
        return

    scope = incident_scope_nodes(G, sources)

    # Propagate
    impact = propagate_anomalies(G, current, decay=0.6)

    # Keep only nodes in scope with non-zero impact (clean report)
    scope_nonzero = {n for n in scope if impact.get(n, 0.0) > 0}

    # Rankings
    impact_rank = rank_impact(impact, scope_nonzero)
    root_rank = compute_root_cause_scores(G, impact, scope_nonzero)

    # Build report structure
    report = {
        "incident": {
            "timestamp": ts.isoformat(sep=" "),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "decay": 0.6,
            "model": "isolation_forest",
        },
        "anomalous_nodes_at_timestamp": [],
        "scope": {
            "nodes_in_scope": sorted(list(scope_nonzero)),
            "scope_size": len(scope_nonzero),
        },
        "rankings": {
            "impact_top": [],
            "root_cause_top": [],
        },
    }

    # Add anomalous nodes with metadata
    for _, row in current.sort_values("anomaly_score", ascending=False).iterrows():
        node_id = row["node_id"]
        meta = nodes_df.loc[node_id].to_dict() if node_id in nodes_df.index else {}
        report["anomalous_nodes_at_timestamp"].append({
            "node_id": node_id,
            "anomaly_score": float(row["anomaly_score"]),
            "top_features": row.get("top_features", None),
            **meta
        })

    # Impact top N
    TOP_N = 10
    for node_id, score in impact_rank[:TOP_N]:
        meta = nodes_df.loc[node_id].to_dict() if node_id in nodes_df.index else {}
        report["rankings"]["impact_top"].append({
            "node_id": node_id,
            "impact_score": float(score),
            **meta
        })

    # Root-cause top N with blast radius and downstream evidence
    for node_id, score in root_rank[:TOP_N]:
        meta = nodes_df.loc[node_id].to_dict() if node_id in nodes_df.index else {}
        report["rankings"]["root_cause_top"].append({
            "node_id": node_id,
            "root_cause_score": float(score),
            "blast_radius": int(blast_radius(G, node_id, scope_nonzero)),
            "top_downstream_impacted": top_downstream_impacted(G, node_id, impact, scope_nonzero, k=5),
            "explanation": (
            f"{node_id} ranks high because it impacts "
            f"{blast_radius(G, node_id, scope_nonzero)} downstream nodes "
            f"and receives minimal upstream impact within the incident scope."
        ),
            **meta
        })


    report["incident_summary"] = {
    "likely_root_cause": report["rankings"]["root_cause_top"][0]["node_id"],
    "most_affected_node": report["rankings"]["impact_top"][0]["node_id"],
    "anomalous_nodes": [x["node_id"] for x in report["anomalous_nodes_at_timestamp"]],
    "scope_size": report["scope"]["scope_size"],
}

    # Save JSON report
    ts_tag = safe_ts_for_filename(ts)
    json_path = processed_dir / f"incident_report_{ts_tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also save a flat CSV table for root cause ranking
    root_rows = []
    for item in report["rankings"]["root_cause_top"]:
        root_rows.append({
            "timestamp": report["incident"]["timestamp"],
            "node_id": item["node_id"],
            "node_type": item.get("node_type", ""),
            "site": item.get("site", ""),
            "criticality": item.get("criticality", ""),
            "root_cause_score": item["root_cause_score"],
            "blast_radius": item["blast_radius"],
            "top_downstream_impacted": ",".join(item["top_downstream_impacted"]),
        })

    csv_path = processed_dir / f"incident_root_causes_{ts_tag}.csv"
    pd.DataFrame(root_rows).to_csv(csv_path, index=False)

    print(f"✅ Saved JSON report: {json_path}")
    print(f"✅ Saved root-cause CSV: {csv_path}")

    # Quick console preview
    print("\nRoot-cause top 5:")
    for item in report["rankings"]["root_cause_top"][:5]:
        print(
            f"{item['node_id']:10s} | score={item['root_cause_score']:.3f} | "
            f"blast_radius={item['blast_radius']:2d} | downstream={item['top_downstream_impacted']}"
        )


if __name__ == "__main__":
    main()
