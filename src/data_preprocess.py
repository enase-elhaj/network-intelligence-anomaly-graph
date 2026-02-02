import pandas as pd
from sklearn.preprocessing import StandardScaler


#Load data
metrics = pd.read_csv("data/raw/metrics_long.csv")
metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])


#Pivot long → wide format

metrics_wide = (
    metrics.pivot_table( index=["timestamp", "node_id"], columns="metric_name",
        values="value").reset_index()
)


#Sort by node_id and timestamp
metrics_wide = metrics_wide.sort_values(["node_id", "timestamp"])

#Handle missing values
metrics_wide = metrics_wide.ffill()  #ffill = use last known value

metrics_wide = metrics_wide.fillna(0) #if no previous value, fill with 0

metrics_wide.to_csv("data/processed/metrics_wide.csv", index=False)



#Per-node, per-feature standardization over time
feature_cols = [
    "latency_ms",
    "packet_loss_pct",
    "traffic_mbps",
    "cpu_pct",
    "mem_pct",
    "error_rate_pct"
]

scaled_frames = []

# Scale features for each node individually
for node_id, df_node in metrics_wide.groupby("node_id"): #group by node_id
    scaler = StandardScaler()
    df_node = df_node.sort_values("timestamp")  #ensure time order

    df_node[feature_cols] = scaler.fit_transform(
        df_node[feature_cols].fillna(0)
    ) #scale features

    scaled_frames.append(df_node)

metrics_scaled = pd.concat(scaled_frames)

#Save scaled data
metrics_scaled.to_csv(
    "data/processed/metrics_scaled.csv",
    index=False
)
