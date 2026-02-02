import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest



FEATURES_BY_TYPE = {
    "router": ["latency_ms", "packet_loss_pct", "traffic_mbps"],
    "switch": ["latency_ms", "packet_loss_pct", "traffic_mbps"],
    "server": ["cpu_pct", "mem_pct"],
    "service": ["error_rate_pct"],
}


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Clean and prepare features:
    - sort by node_id, timestamp
    - forward fill per node (telemetry gaps)
    - fill remaining NaNs with 0
    """
    df = df.sort_values(["node_id", "timestamp"]).copy()
    df[feature_cols] = df.groupby("node_id")[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def fit_isoforest_per_node(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.10,
    random_state: int = 42,
    min_rows_per_node: int = 5,
) -> pd.DataFrame:
    """
    Fit Isolation Forest per node (recommended for infra telemetry).
    Returns df with:
      - anomaly_score (higher = more anomalous)
      - is_anomaly (boolean, based on model prediction)
    Notes:
      - IsolationForest.score_samples(): higher = more normal
      - We'll invert to get anomaly_score where higher = more anomalous
    """
    df = df.sort_values(["node_id", "timestamp"]).copy()

    scores = []
    preds = []

    for node_id, g in df.groupby("node_id", sort=False):
        g = g.sort_values("timestamp").copy()
        X = g[feature_cols].to_numpy(dtype=float)

        # If very small sample for this node, fallback to a simple distance-from-mean score.
        # (Useful now with tiny demo data; later with more data, the IF path dominates.)
        if len(g) < min_rows_per_node:
            mu = X.mean(axis=0)
            # Euclidean distance as a fallback anomaly score (scaled data makes this sensible)
            dist = np.linalg.norm(X - mu, axis=1)
            # Normalize to 0..1 for readability (optional)
            if dist.max() > 0:
                dist = dist / dist.max()
            anomaly_score = dist
            is_anomaly = anomaly_score >= 0.8  # conservative fallback threshold

            scores.append(pd.Series(anomaly_score, index=g.index))
            preds.append(pd.Series(is_anomaly, index=g.index))
            continue

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(X)

        # score_samples: higher is more normal, lower is more anomalous
        normality = model.score_samples(X)
        anomaly_score = -normality  # invert so higher = more anomalous

        # predict: -1 anomaly, +1 normal
        pred = model.predict(X)
        is_anomaly = pred == -1

        scores.append(pd.Series(anomaly_score, index=g.index))
        preds.append(pd.Series(is_anomaly, index=g.index))

    df["anomaly_score"] = pd.concat(scores).sort_index().values
    df["is_anomaly"] = pd.concat(preds).sort_index().values
    return df


# def top_contributors(row: pd.Series, feature_cols: list[str], top_k: int = 2) -> str:
#     """
#     Simple heuristic explanation: for a flagged point, show the features with
#     largest absolute values (since scaled features are comparable).
#     """
#     vals = row[feature_cols].astype(float)
#     idx = vals.abs().sort_values(ascending=False).head(top_k).index.tolist()
#     return ",".join(idx)


def top_contributors(row: pd.Series, node_type: str, top_k: int = 2) -> str:
    """
    Explanation helper:
    - selects only the features relevant to this node_type
    - returns top_k features by absolute value (scaled data makes this meaningful)
    """
    feature_cols = FEATURES_BY_TYPE.get(node_type, [])

    # If node_type unknown or no features defined, return empty
    if not feature_cols:
        return ""

    # Keep only columns that actually exist in the row (safety)
    feature_cols = [c for c in feature_cols if c in row.index]

    if not feature_cols:
        return ""

    vals = row[feature_cols].astype(float)
    top = vals.abs().sort_values(ascending=False).head(top_k).index.tolist()
    return ",".join(top)


def main():
    repo_root = Path(__file__).resolve().parents[1]
##############
    nodes_df = pd.read_csv(repo_root / "data" / "raw" / "nodes.csv")
    node_type_map = dict(zip(nodes_df["node_id"], nodes_df["node_type"]))
###########

    in_path = repo_root / "data" / "processed" / "metrics_scaled.csv"
    out_path = repo_root / "data" / "processed" / "anomalies_isoforest.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing: {in_path}\n"
            "Make sure you generated data/processed/metrics_scaled.csv first."
        )

    df = pd.read_csv(in_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Feature columns (use only those present)
    candidate_features = [
        "latency_ms",
        "packet_loss_pct",
        "traffic_mbps",
        "cpu_pct",
        "mem_pct",
        "error_rate_pct",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in input file.")

    df = prepare_features(df, feature_cols)

    # Per-node Isolation Forest
    result = fit_isoforest_per_node(
        df=df,
        feature_cols=feature_cols,
        contamination=0.10,       # reasonable starting point; tune later
        random_state=42,
        min_rows_per_node=5       # with tiny sample, fallback triggers; increase when data grows
    )

    # Add a small explanation column (helpful for demos)
    # result["top_features"] = result.apply(lambda r: top_contributors(r, feature_cols, top_k=2), axis=1)

    def compute_top_features(row):
        node_id = row["node_id"]
        node_type = node_type_map.get(node_id, "")
        return top_contributors(row, node_type=node_type, top_k=2)

    result["top_features"] = result.apply(compute_top_features, axis=1)


    
    # Save
    result.to_csv(out_path, index=False)

    # Console summary
    n_anom = int(result["is_anomaly"].sum())
    print(f"Saved: {out_path}")
    print(f"Total rows: {len(result)} | Anomalies flagged: {n_anom}")

    # Show top anomalies
    anomalies = result[result["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    if len(anomalies) > 0:
        print("\nTop anomalies:")
        print(
            anomalies[["timestamp", "node_id", "anomaly_score", "top_features"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("\nNo anomalies flagged. Try increasing contamination to 0.15 or lowering fallback threshold.")


if __name__ == "__main__":
    main()
