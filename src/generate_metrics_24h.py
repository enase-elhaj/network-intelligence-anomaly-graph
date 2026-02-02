import pandas as pd
import numpy as np
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "raw" / "metrics_long.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 24 hourly timestamps
    start = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = pd.date_range(start, periods=24, freq="h")

    rows = []

    # ---- Helper to add one metric row
    def add(ts, node_id, metric_name, value):
        rows.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "node_id": node_id,
                "metric_name": metric_name,
                "value": float(value),
            }
        )

    # We keep it small but realistic:
    # - Router R1: latency_ms, packet_loss_pct, traffic_mbps
    # - Switch SW1: latency_ms, packet_loss_pct, traffic_mbps
    # - Server SV1: cpu_pct, mem_pct
    # - Service AUTH: error_rate_pct

    rng = np.random.default_rng(42)

    for i, ts in enumerate(timestamps):
        # --------- Normal baselines with mild variation
        # Router R1
        r1_latency = 12 + 1.0*np.sin(i/24*2*np.pi) + rng.normal(0, 0.3)
        r1_loss    = 0.2 + rng.normal(0, 0.03)
        r1_traffic = 520 + 40*np.sin((i-6)/24*2*np.pi) + rng.normal(0, 10)

        # Switch SW1
        sw1_latency = 8 + 0.8*np.sin(i/24*2*np.pi) + rng.normal(0, 0.25)
        sw1_loss    = 0.1 + rng.normal(0, 0.02)
        sw1_traffic = 410 + 30*np.sin((i-6)/24*2*np.pi) + rng.normal(0, 8)

        # Server SV1
        sv1_cpu = 35 + 3*np.sin((i-4)/24*2*np.pi) + rng.normal(0, 1.0)
        sv1_mem = 61 + 2*np.sin((i-5)/24*2*np.pi) + rng.normal(0, 0.8)

        # Service AUTH
        auth_err = 0.4 + rng.normal(0, 0.05)

        # --------- Inject a clear anomaly "incident" at hour 2 (02:00)
        if i == 2:
            r1_latency += 40
            r1_loss    += 1.5
            r1_traffic += 140

            sw1_latency += 12
            sw1_loss    += 0.7
            sw1_traffic += 40

            sv1_cpu += 30
            sv1_mem += 18

            auth_err += 2.8

        # Keep percentages non-negative
        r1_loss = max(r1_loss, 0)
        sw1_loss = max(sw1_loss, 0)
        auth_err = max(auth_err, 0)

        # ---- Add rows
        add(ts, "R1", "latency_ms", r1_latency)
        add(ts, "R1", "packet_loss_pct", r1_loss)
        add(ts, "R1", "traffic_mbps", r1_traffic)

        add(ts, "SW1", "latency_ms", sw1_latency)
        add(ts, "SW1", "packet_loss_pct", sw1_loss)
        add(ts, "SW1", "traffic_mbps", sw1_traffic)

        add(ts, "SV1", "cpu_pct", sv1_cpu)
        add(ts, "SV1", "mem_pct", sv1_mem)

        add(ts, "AUTH", "error_rate_pct", auth_err)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
