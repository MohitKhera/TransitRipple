import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

PROCESSED_DIR = Path("data/processed")
STOPS = PROCESSED_DIR / "stops.csv"
EDGE_LIST = PROCESSED_DIR / "edge_list.csv"

stops_df = pd.read_csv(STOPS)
edges_df = pd.read_csv(EDGE_LIST)

rail_children = stops_df[stops_df["parent_station"] >= 40000]
child_to_parent = {row["stop_id"]: row["parent_station"] for _, row in rail_children.iterrows()}

edges_df["parent_stop_id"] = edges_df["stop_id"].map(child_to_parent)
edges_df["parent_next_stop_id"] = edges_df["next_stop_id"].map(child_to_parent)

rail_stops_df = stops_df[stops_df["stop_id"] >= 40000]
rail_edges_df = edges_df.dropna(subset=["parent_stop_id", "parent_next_stop_id"])

node_mapping = {stop_id: idx for idx, stop_id in enumerate(rail_stops_df["stop_id"].tolist())}

G = nx.DiGraph()

for _, row in rail_stops_df.iterrows():
    G.add_node(node_mapping[row["stop_id"]], 
               stop_id=row["stop_id"],
               name=row["stop_name"],
               lat=row["stop_lat"],
               lon=row["stop_lon"])
    
for _, row in rail_edges_df.iterrows():
    G.add_edge(node_mapping[int(row["parent_stop_id"])], node_mapping[int(row["parent_next_stop_id"])])

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Is connected: {nx.is_weakly_connected(G)}")
print(f"Connected components: {nx.number_weakly_connected_components(G)}")

largest_component = max(nx.weakly_connected_components(G), key=len)
G = G.subgraph(largest_component).copy()
print(f"\nUsing largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with open(PROCESSED_DIR / "graph.pkl", "wb") as f:
    pickle.dump(G, f)