import gtfs_kit as gk 
import pandas as pd
from pathlib import Path

GTFS_ZIP = Path("data/raw/gtfs_static/google_transit.zip")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading GTFS feed...")
feed = gk.read_feed(GTFS_ZIP, dist_units="km")

stops = feed.stops[["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"]].copy()
stops.to_csv(OUTPUT_DIR / "stops.csv", index=False)
print(f"  Stops: {len(stops)} rows")

routes = feed.routes[["route_id", "route_short_name", "route_long_name", "route_color"]].copy()
routes.to_csv(OUTPUT_DIR / "routes.csv", index=False)
print(f"  Routes: {len(routes)} rows")

stop_times = feed.stop_times[["trip_id", "stop_id", "stop_sequence"]].copy()
stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
stop_times["next_stop_id"] = stop_times.groupby("trip_id")["stop_id"].shift(-1)
stop_times = stop_times.dropna(subset=["next_stop_id"])

edge_list = stop_times[["stop_id", "next_stop_id"]].copy().drop_duplicates()
edge_list.to_csv(OUTPUT_DIR / "edge_list.csv", index=False)
print(f"  Edges: {len(edge_list)} rows")

print("\nCheck data/processed/")