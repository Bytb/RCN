import os
import csv
import pickle

# <<< Change this to point to your .gpickle file >>>
base_dir = os.path.dirname(__file__)
path_file = os.path.join(base_dir,"Graphs","ppi_graph.gpickle")

# how many edges you want to sample
edge_sample = 1000

def safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return repr(x)

# Load the graph
with open(path_file, "rb") as f:
    G = pickle.load(f)

# Output directory = same folder as input
outdir = os.path.dirname(path_file)
if outdir == "":
    outdir = "."

# Export nodes
node_attrs = set()
for _, attrs in G.nodes(data=True):
    node_attrs.update(attrs.keys())
node_attrs = sorted(node_attrs)

with open(os.path.join(outdir, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["node_id"] + node_attrs)
    writer.writeheader()
    for n, attrs in G.nodes(data=True):
        row = {"node_id": safe_str(n)}
        for k in node_attrs:
            row[k] = safe_str(attrs.get(k, ""))
        writer.writerow(row)

# Export edges (sample)
edge_attrs = set()
for _, _, attrs in list(G.edges(data=True))[:edge_sample]:
    edge_attrs.update(attrs.keys())
edge_attrs = sorted(edge_attrs)

with open(os.path.join(outdir, "edges_sample.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["u", "v"] + edge_attrs)
    writer.writeheader()
    for count, (u, v, attrs) in enumerate(G.edges(data=True)):
        row = {"u": safe_str(u), "v": safe_str(v)}
        for k in edge_attrs:
            row[k] = safe_str(attrs.get(k, ""))
        writer.writerow(row)
        if count + 1 >= edge_sample:
            break

print("Export complete. Files written:")
print(" -", os.path.join(outdir, "nodes.csv"))
print(" -", os.path.join(outdir, "edges_sample.csv"))
