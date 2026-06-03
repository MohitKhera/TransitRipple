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

model = TransitGNN(in_channels=1, hidden_channels=64, out_channels=1)
model.load_state_dict(torch.load("model/best_model.pth"))
model.eval()

all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for i in range(len(X_test)):
        # get one sample, reshape it
        # run through model
        # collect output and target
        x = X_test[i].unsqueeze(-1)
        target = y_test[i].unsqueeze(-1)
        all_targets.append(target)
        output = model(x, edge_index)
        all_preds.append(output)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)
probs = torch.sigmoid(all_preds)
preds = (probs > 0.5).float()

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

preds_np = preds.numpy().flatten()
targets_np = all_targets.numpy().flatten()
probs_np = probs.numpy().flatten()

accuracy = accuracy_score(targets_np, preds_np)
f1 = f1_score(targets_np, preds_np)
auc = roc_auc_score(targets_np, probs_np)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC:  {auc:.4f}")

persistence_preds = X_test[:, 3, :]

persistence_preds_np = persistence_preds.numpy().flatten()

p_accuracy = accuracy_score(targets_np, persistence_preds_np)
p_f1 = f1_score(targets_np, persistence_preds_np)

print(f"\n--- Persistence Baseline ---")
print(f"Accuracy: {p_accuracy:.4f}")
print(f"F1 Score: {p_f1:.4f}")

maj_class_baseline = [0] * len(targets_np)

m_accuracy = accuracy_score(targets_np, maj_class_baseline)
m_f1 = f1_score(targets_np, maj_class_baseline)

print(f"\n--- Majority Class Baseline ---")
print(f"Accuracy: {m_accuracy:.4f}")
print(f"F1 Score: {m_f1:.4f}")