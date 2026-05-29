import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/gtfs_rt")
OUTPUT_DIR = Path("data/processed")

snapshot_files = sorted(RAW_DIR.glob("snapshot_*.csv"))
print(f"Found {len(snapshot_files)} snapshots")

all_records = []
for filepath in tqdm(snapshot_files):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            continue
        timestamp = "_".join(filepath.stem.split("_")[1:])
        df["arrT"] = pd.to_datetime(df["arrT"], format="%Y-%m-%dT%H:%M:%S")
        df["prdt"] = pd.to_datetime(df["prdt"], format="%Y-%m-%dT%H:%M:%S")
        df["wait_minutes"] = (df["arrT"] - df["prdt"]).dt.total_seconds() / 60
        station_stats = df.groupby("staId").agg(
            is_delayed=("isDly", "max"),
            num_trains=("rn", "count"),
            avg_wait=("wait_minutes", "mean")
        ).reset_index()
        station_stats["snapshot_time"] = timestamp
        all_records.append(station_stats)
    except Exception as e:
        continue

results_df = pd.concat(all_records, ignore_index=True)
results_df.to_csv(OUTPUT_DIR / "processed_snapshots.csv", index=False)
print(f"Saved {len(results_df)} rows")