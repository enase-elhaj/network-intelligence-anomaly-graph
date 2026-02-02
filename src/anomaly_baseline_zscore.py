import pandas as pd
import numpy as np
from pathlib import Path


def rolling_zscore_per_node(df: pd.DataFrame, feature_cols: list[str],
    window: int = 3, min_periods: int = 2, z_threshold: float = 3.0, 
) -> pd.DataFrame: 
    """
    Compute rolling z-scores per node per feature using ONLY past information.
    - Uses rolling mean/std over a window
    - Shifts the rolling stats by 1 step to avoid using the current point (no leakage)
    """

    df = df.sort_values(["node_id", "timestamp"]).copy()

    # Prepare output columns
    for col in feature_cols:
        df[f"z_{col}"] = np.nan  #initialize z-score columns

    # Compute per-node
    out_frames = []
    for node_id, g in df.groupby("node_id", sort=False): #group by node_id
        g = g.sort_values("timestamp").copy()  #ensure time order

        for col in feature_cols:
            # If the column doesn't exist, skip
            if col not in g.columns:
                continue

            # Rolling stats using past values only (shifted)
            rolling_mean = g[col].rolling(window=window, min_periods=min_periods).mean().shift(1)
            rolling_std = g[col].rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)

            z = (g[col] - rolling_mean) / rolling_std
            g[f"z_{col}"] = z

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)

    # Overall anomaly score: max absolute z across features (simple + interpretable)
    z_cols = [f"z_{c}" for c in feature_cols if f"z_{c}" in result.columns]
    result["anomaly_score"] = result[z_cols].abs().max(axis=1)

    # Flag anomalies
    result["is_anomaly"] = result["anomaly_score"] >= z_threshold

    # Helpful: which feature caused the max z-score
    def argmax_feature(row):
        if row[z_cols].isna().all():
            return np.nan
        return row[z_cols].abs().idxmax().replace("z_", "")

    result["top_feature"] = result.apply(argmax_feature, axis=1)

    return result


def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    processed_path = repo_root / "data" / "processed" / "metrics_wide.csv"
    out_path = repo_root / "data" / "processed" / "anomalies_baseline_zscore.csv"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {processed_path}\n"
            "Run your preprocessing step first to create data/processed/metrics_wide.csv"
        )

    df = pd.read_csv(processed_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Choose feature columns present in your dataset (safe defaults)
    candidate_features = [
        "latency_ms",
        "packet_loss_pct",
        "traffic_mbps",
        "cpu_pct",
        "mem_pct",
        "error_rate_pct",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    # Fill missing feature values (because some metrics don't apply to some node types)
    # Forward-fill per node, then remaining NaNs -> 0
    df = df.sort_values(["node_id", "timestamp"])
    df[feature_cols] = df.groupby("node_id")[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0)

    # Run baseline detector
    result = rolling_zscore_per_node(
        df=df,
        feature_cols=feature_cols,
        window=3,          # for your small dataset; later you can use 24 for hourly/day
        min_periods=2,
        z_threshold=3.0,
    )

    # Save
    result.to_csv(out_path, index=False)

    # Quick console summary
    n_anom = int(result["is_anomaly"].sum())
    print(f"Saved: {out_path}")
    print(f"Total rows: {len(result)} | Anomalies flagged: {n_anom}")

    # Show the anomalies (top few)
    anomalies = result[result["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    if len(anomalies) > 0:
        print("\nTop anomalies:")
        print(anomalies[["timestamp", "node_id", "anomaly_score", "top_feature"]].head(10).to_string(index=False))
    else:
        print("\nNo anomalies flagged. Try lowering z_threshold to 2.5 or increasing window.")


if __name__ == "__main__":
    main()
