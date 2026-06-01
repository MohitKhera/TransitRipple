import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from gnn import TransitGNN
import mlflow
import mlflow.pytorch

path = Path("data/processed/dataset.pkl")
with open(path, "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]
station_ids = data["station_ids"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Stations: {len(station_ids)}")

PROCESSED_DIR = Path("data/processed")
edge_df = pd.read_csv(PROCESSED_DIR / "edge_list.csv")
stops_df = pd.read_csv(PROCESSED_DIR / "stops.csv")
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
print(f"Edge index shape: {edge_index.shape}")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 80/20 train/test split
split = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

model = TransitGNN(in_channels=1, hidden_channels=32, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 10
LR = 0.001
HIDDEN = 32

with mlflow.start_run():
    mlflow.log_param("lr", LR)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("hidden_channels", HIDDEN)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in range(len(X_train)):
            x = X_train[i].unsqueeze(-1)
            target = y_train[i].unsqueeze(-1)
            optimizer.zero_grad()
            output = model(x, edge_index)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(X_train)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    mlflow.pytorch.log_model(model, "model")
    print("Model logged to MLflow")