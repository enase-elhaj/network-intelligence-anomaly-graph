import json
from pathlib import Path
from openai import OpenAI

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fallback_narrative(report: dict) -> str:
    """
    Deterministic fallback narrative when LLM is unavailable.
    Uses the incident report structure to produce a clean explanation.
    """
    inc = report["incident"]
    summary = report.get("incident_summary", {})
    anomalies = report["anomalous_nodes_at_timestamp"]
    impact = report["rankings"]["impact_top"]
    root = report["rankings"]["root_cause_top"][0]

    lines = []

    lines.append("## Summary")
    lines.append(
        f"- An infrastructure anomaly was detected at **{inc['timestamp']}** "
        f"using an **Isolation Forest** model."
    )
    lines.append(
        f"- The most likely root cause is **{root['node_id']}**, "
        f"with a blast radius of **{root['blast_radius']} nodes**."
    )

    lines.append("\n## Detection")
    for a in anomalies:
        lines.append(
            f"- **{a['node_id']}** ({a['node_type']}): anomalous behavior detected "
            f"in `{a['top_features']}`."
        )

    lines.append("\n## Impact (Blast Radius)")
    for i in impact[:5]:
        lines.append(
            f"- **{i['node_id']}** affected (impact_score ≈ {i['impact_score']:.2f})"
        )

    lines.append("\n## Root-Cause Hypothesis")
    lines.append(
        f"- **{root['node_id']}** is the leading root-cause candidate because it "
        f"propagates impact to downstream nodes: "
        f"{', '.join(root['top_downstream_impacted'])}."
    )

    lines.append("\n## Recommended Next Checks")
    lines.extend([
        "- Inspect latency and packet loss metrics on the root-cause device.",
        "- Check upstream/downstream interfaces for congestion or errors.",
        "- Validate server resource utilization during the incident window.",
        "- Review service error rates and retry patterns.",
        "- Correlate with recent configuration or deployment changes.",
    ])

    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # --- Input report (update filename if needed)
    report_path = repo_root / "data" / "processed" / "incident_report_2026-01-01T02-00-00.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing incident report: {report_path}")

    report = load_json(report_path)

    # --- Prompt template
    prompt_path = repo_root / "prompts" / "incident_narrative_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    instructions = load_text(prompt_path)

    # --- OpenAI client
    client = OpenAI()

    # We send the report as JSON text (model can read structured input)
    report_json_text = json.dumps(report, indent=2)

    # --- Call LLM to generate narrative
    try:
        response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    "Here is the incident report JSON. Generate the incident narrative.\n\n"
                    f"{report_json_text}"
                ),
            },
        ],
    )
        output_text = response.output_text
        print("✅ LLM narrative generated via OpenAI API")

    except Exception as e:
        print(f"⚠️ LLM unavailable ({type(e).__name__}). Using fallback narrative.")
        output_text = fallback_narrative(report)


    # --- Save narrative next to report
    out_md = report_path.with_suffix(".narrative.md")
    out_md.write_text(output_text, encoding="utf-8")

    print(f"✅ Saved narrative: {out_md}")

if __name__ == "__main__":
    main()
