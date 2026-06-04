# Delaydar 🚇

**Predicting cascading delay propagation across Chicago's CTA L train network using Temporal Graph Neural Networks.**

---

## Overview

Delaydar models the CTA L train network as a dynamic graph and predicts how a delay at any station propagates across the network over a 30–60 minute horizon. Rather than treating delays as isolated events, it treats them as contagions spreading through a graph — capturing the spatial and temporal structure of real transit disruptions.

Built end-to-end: from a custom data collection pipeline deployed on AWS EC2, through graph construction and feature engineering, to a trained GNN with a live FastAPI inference endpoint.

---

## Pipeline

```
CTA Train Tracker API (every 2 min, AWS EC2)
        ↓
Raw snapshots (5,291 CSVs, ~5.7M arrival records)
        ↓
GTFS static feed → Graph construction (143 stations, 298 edges)
        ↓
Feature engineering → Delay matrix (3,055 snapshots × 143 stations)
        ↓
Temporal GNN (GCNConv + GRU) → Binary delay prediction
        ↓
FastAPI inference endpoint → POST /predict
```

---

## Model Architecture

**DelaydarGNN** combines spatial and temporal components:

- **GCNConv (×2)** — graph convolution layers that propagate delay signals across neighboring stations
- **GRU** — recurrent layer that captures how delay patterns evolve over time
- **Linear** — output layer producing per-station delay logits

Input: 4 timestep window of delay status across all 143 stations `(4 × 143 × 1)`  
Output: delay probability for each station `(143 × 1)`

---

## Results

Evaluated on a held-out temporal test set (611 samples, 20% of data):

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| **DelaydarGNN** | 0.9386 | 0.7250 | **0.9086** |
| Persistence Baseline | 0.9423 | 0.7497 | — |
| Majority Class Baseline | 0.8847 | 0.0000 | — |

> TransitGNN achieves **AUC-ROC of 0.91**, meaning it correctly ranks delayed stations above non-delayed ones 91% of the time.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data collection | Python, CTA Train Tracker API, AWS EC2 |
| GTFS parsing | gtfs-kit, pandas |
| Graph construction | NetworkX → PyTorch Geometric |
| Model | PyTorch, GCNConv + GRU |
| Experiment tracking | MLflow |
| API | FastAPI, uvicorn |

---

## Project Structure

```
delaydar/
├── data/
│   ├── raw/gtfs_static/          # GTFS static feed
│   └── processed/                # Processed features, graph, dataset
├── pipeline/
│   ├── parse_gtfs.py             # Extract stops, routes, edges from GTFS
│   ├── collect_arrivals.py       # CTA API collector (runs on AWS EC2)
│   ├── build_graph.py            # NetworkX graph (143 nodes, 298 edges)
│   ├── process_snapshots.py      # Per-station delay features
│   └── build_dataset.py          # Temporal windows (X, y tensors)
├── model/
│   ├── gnn.py                    # DelaydarGNN architecture
│   ├── train.py                  # Training loop with MLflow tracking
│   ├── evaluate.py               # Test set evaluation + baselines
│   └── best_model.pth            # Saved model weights
└── api/
    └── main.py                   # FastAPI inference endpoint
```

---

## API

Start the server:

```bash
uvicorn api.main:app --reload
```

**GET /health**
```json
{"status": "ok"}
```

**POST /predict**

Request:
```json
{
  "stations": [41450, 40900, 41320]
}
```

Response:
```json
{
  "predictions": [
    {"station_id": 41450, "delay_probability": 0.020},
    {"station_id": 40900, "delay_probability": 0.746},
    {"station_id": 41320, "delay_probability": 0.023}
  ]
}
```

Interactive docs available at `http://127.0.0.1:8000/docs`

---

## Setup

```bash
# Clone the repo
git clone https://github.com/MohitKhera/Delaydar
cd Delaydar

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add CTA API key
echo CTA_API_KEY=your_key_here > .env
```

---

## Frontend
A live map visualization is available at `frontend/index.html`. 
Open it in a browser with the API running to see per-station delay 
probabilities across all 143 CTA L stations, color-coded by risk level 
and auto-refreshing every 2 minutes.

---

## Future Work

- **Temporal features** — add time-of-day and day-of-week as node features; delays behave very differently at rush hour vs. overnight
- **Larger dataset** — 5 days of collection captures limited weekly patterns; 3–4 weeks would expose Monday morning vs. weekend dynamics
- **Batch training** — current sample-by-sample training is slow; PyTorch Geometric DataLoader would enable proper mini-batch training
- **Live inference** — connect `/predict` to the real-time CTA API instead of historical snapshots

---

## Data Sources

- [CTA Train Tracker API](https://www.transitchicago.com/developers/ttdocs/)
- [CTA GTFS Static Feed](https://www.transitchicago.com/developers/gtfs/)
- [Chicago Data Portal — L Station Entries](https://data.cityofchicago.org/)
