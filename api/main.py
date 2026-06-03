from fastapi import FastAPI
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
from pydantic import BaseModel
from typing import List

app = FastAPI()
sys.path.append(str(Path(__file__).parent.parent / "model"))
from gnn import TransitGNN

BASE_DIR = Path(__file__).parent.parent

# Load model
model = TransitGNN(in_channels=1, hidden_channels=64, out_channels=1)
model.load_state_dict(torch.load(BASE_DIR / "model" / "best_model.pth"))
model.eval()
path = BASE_DIR / "data" / "processed" / "dataset.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)
station_ids = data["station_ids"]
df = pd.read_csv(BASE_DIR / "data" / "processed" / "processed_snapshots.csv")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
edge_df = pd.read_csv(PROCESSED_DIR  / "edge_list.csv")
stops_df = pd.read_csv(PROCESSED_DIR  / "stops.csv")
rail_children = stops_df[stops_df["parent_station"] >= 40000]
child_to_parent = {row["stop_id"]: int(row["parent_station"]) for _, row in rail_children.iterrows()}

station_to_idx = {stop_id: idx for idx, stop_id in enumerate(station_ids)}

edges = []
for _, row in edge_df.iterrows():
    src_child = row["stop_id"]
    dst_child = row["next_stop_id"]
    src_parent = child_to_parent.get(src_child)
    dst_parent = child_to_parent.get(dst_child)
    src = station_to_idx.get(src_parent)
    dst = station_to_idx.get(dst_parent)
    if src is not None and dst is not None:
        edges.append([src, dst])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

class PredictRequest(BaseModel):
    stations: List[int]

class StationPrediction(BaseModel):
    station_id: int
    delay_probability: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Get last 4 snapshots from df
    last_four_times = df['snapshot_time'].unique()[-4:]
    last_four = df[df['snapshot_time'].isin(last_four_times)]
    pivoted = last_four.pivot(index='snapshot_time', columns='staId', values='is_delayed')
    pivoted = pivoted.reindex(columns=station_ids)
    snapshot_values = pivoted.values  # shape: (4, 143)
    x = torch.tensor(snapshot_values, dtype=torch.float32).unsqueeze(-1)  # shape: [4, 143, 1]
    # 3. Run model
    with torch.no_grad():
        output = model(x, edge_index)
        probs = torch.sigmoid(output)
    # 4. Filter results for requested stations
    # 5. Return predictions
    results = []
    for station_id in request.stations:
        idx = station_to_idx.get(station_id)
        if idx is not None:
            prob = probs[idx].item()
            results.append(StationPrediction(station_id=station_id, delay_probability=prob))
    return {"predictions": results}