import pandas as pd
import numpy as np
from pathlib import Path
import pickle

WINDOW_SIZE = 4
HORIZON = 1

PROCESSED_DIR = Path("data/processed")
df = pd.read_csv(PROCESSED_DIR / "processed_snapshots.csv")
print(f"Loaded {len(df)} rows, {df['staId'].nunique()} unique stations")

df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], format="%Y%m%d_%H%M%S")
df = df.sort_values("snapshot_time")

snapshots = sorted(df["snapshot_time"].unique())
print(f"Total snapshots: {len(snapshots)}")
print(f"First: {snapshots[0]}")
print(f"Last: {snapshots[-1]}")

delay_matrix = df.pivot_table(
    index="snapshot_time",
    columns="staId",
    values="is_delayed",
    aggfunc="max"
).fillna(0)

print(f"Delay matrix shape: {delay_matrix.shape}")

X, y = [], []
matrix = delay_matrix.values  # convert to numpy array

for i in range(len(matrix) - WINDOW_SIZE - HORIZON + 1):
    X.append(matrix[i:i+WINDOW_SIZE])      # 4 consecutive snapshots
    y.append(matrix[i+WINDOW_SIZE])           # 1 snapshot ahead

X = np.array(X)
y = np.array(y)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

with open(PROCESSED_DIR / "dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y, "station_ids": delay_matrix.columns.tolist()}, f)
print("Dataset saved.")