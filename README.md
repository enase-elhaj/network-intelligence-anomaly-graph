Network Intelligence: Anomaly Detection & Graph-Based Root Cause Analysis

Overview
This project demonstrates an end-to-end Infrastructure Intelligence pipeline for detecting network anomalies, reasoning over system dependencies using graphs, and generating human-readable incident reports.

The system combines:

Statistical and ML-based anomaly detection

Graph-based impact propagation and root-cause analysis

LLM-assisted (with deterministic fallback) incident explanation

It is designed to mirror real-world challenges in network operations, observability, and incident response.

Key Capabilities

**Anomaly Detection**

Rolling Z-score baseline (per node, per metric)

Isolation Forest (unsupervised ML)

🕸️ Graph-Based System Intelligence

Dependency graph of routers, switches, servers, and services

Downstream impact propagation with decay

Blast radius estimation

Root-cause vs symptom separation

Explainability & Reasoning:

Feature-level anomaly attribution

Incident-scoped subgraphs

Structured JSON incident reports

Human-readable incident narratives (LLM-assisted with fallback):

Production-Oriented Design

Deterministic pipelines

Clear ETL stages:

Reproducible artifacts (CSV / JSON / Markdown)

Project Structure
network-intelligence-anomaly-graph/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│       ├── nodes.csv              # Node metadata (type, site, criticality)
│       ├── edges.csv              # Directed dependency edges
│       └── metrics_wide.csv       # Raw telemetry (wide format)
│   
├── prompts/    
│    └── incident_narrative_prompt
│
├── src/
│   ├── data_preprocess.py               # Scaling and preprocessing
│   ├── anomaly_baseline_zscore.py         # Rolling z-score anomaly baseline
│   ├── anomaly_isolation_forest.py# ML-based anomaly detection
│   ├── graph_model.py             # Dependency graph construction
│   ├── graph_anomaly_propagation.py       # Impact propagation & root cause logic
|   ├── incident_report.py 
|   ├── visualize_incident_subgraph.py 
│   └── llm_incident_explainer.py  # Incident narrative generation
│
├── reports/
│   └── incident_subgraph_2026-01-01T02-00-00.png
│     
│
└── notebooks/
    └── 01_eda.ipynb         

Data Model:
Nodes

Each node represents an infrastructure or service component:

routers

switches

servers

application services

Node metadata includes:

node_type

site

vendor

criticality

Edges


Telemetry

Per-node time-series metrics such as:

latency

packet loss

CPU / memory utilization

error rates

traffic volume

Methodology
1. Anomaly Detection

Two complementary approaches are used:

Rolling Z-Score (Baseline)

Per-node, per-metric rolling mean & standard deviation

Explainable statistical baseline

High sensitivity, useful for early detection

Isolation Forest

Unsupervised ML model

Captures multivariate interactions

Lower false positives than pure statistical thresholds

Each anomalous point includes:

anomaly score

top contributing features (metric-level attribution)

2. Graph-Based Impact Propagation

Anomalies are injected into a directed dependency graph and propagated downstream using a decay factor:

Impact weakens with graph distance

Nodes accumulate impact from upstream anomalies

Produces an incident-scoped subgraph

This allows the system to distinguish:

Highly impacted nodes (symptoms)

High-leverage nodes (likely root causes)

3. Root Cause Analysis

Root-cause scoring favors nodes that:

generate large downstream impact

receive minimal upstream impact

This separates cause vs effect, a common challenge in observability systems.

4. Incident Reporting & Explanation

For a detected incident:

Structured JSON report is generated

Includes anomalous nodes, impact ranking, root-cause ranking, and blast radius

A human-readable narrative is produced:

This mirrors how agentic AI systems assist on-call engineers.

Example Findings (Incident: 2026-01-01 02:00:00)

Detected Anomalies

R1 (router): elevated latency and packet loss

SV1 (server): high CPU and memory utilization

SW1 (switch): packet loss and latency spikes

AUTH (service): increased error rates

Impact Ranking (Top Affected)

AUTH

SV1

SW1

BILLING

CATALOG

Root Cause Hypothesis

R1 identified as the most likely root cause

Large downstream blast radius affecting 6 nodes

Propagation path consistent with dependency graph


This project demonstrates:

Practical anomaly detection (not just modeling)

System-level reasoning using graphs

Explainability and human-in-the-loop design

Production-aware data science workflows


Technologies Used

Python

pandas, NumPy

scikit-learn

NetworkX

OpenAI API 

PyTorch (conceptual extension via autoencoder discussion)

OpenAI API 

Future Extensions

Neo4j / graph database integration

Streaming telemetry ingestion

Real-time incident detection


Author

Enas Elhaj
Data Scientist | Applied ML
Graduate Student — Applied Artificial Intelligence & Data Science
University of Denver | Ritchie School of Engineering & Computer Science

Former telecom & computer engineering Lecturer & Assistant Professor
📍 United States