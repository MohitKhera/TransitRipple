import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import zipfile
from dotenv import load_dotenv
import os

load_dotenv()

GTFS_ZIP = Path("data/raw/gtfs_static/google_transit.zip")

API_KEY = os.getenv("CTA_API_KEY")
OUTPUT_DIR = Path("data/raw/gtfs_rt")
INTERVAL_SECONDS = 120
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_IDS = []
with zipfile.ZipFile(GTFS_ZIP) as z:
    df = pd.read_csv(z.open("stops.txt"))
    stations = df[df["location_type"] == 1]
    STATION_IDS = stations["stop_id"].tolist()

def fetch_arrivals(station_id):
    url = f"http://lapi.transitchicago.com/api/1.0/ttarrivals.aspx?key={API_KEY}&mapid={station_id}&outputType=JSON"
    try:
        req = requests.get(url)
        return req.json()
    except Exception as e:
        print(f"Error fetching station {station_id}: {e}")
        return None

def save_snapshot(records):
    df = pd.DataFrame(records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(OUTPUT_DIR / f"snapshot_{timestamp}.csv", index=False)

if __name__ == "__main__":
    while True:
        all_records = []
        for station_id in STATION_IDS:
            response = fetch_arrivals(station_id)
            if response is None:
                continue
            arrivals = response["ctatt"].get("eta", [])
            all_records.extend(arrivals)
        save_snapshot(all_records)
        print(f"Saved {len(all_records)} records at {datetime.now()}")
        time.sleep(INTERVAL_SECONDS)

