import pickle
import networkx as nx
import numpy as np
from rnbrw.utils import assign_rnbrw_weights

# --- Read the graph ---
with open("path-to-graph/facebook_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# --- Assign weights ---
T_total = np.load("path-to-weights-folder/weights/T_total.npy")

G = assign_rnbrw_weights(G, T_total)

# --- Save the graph ---
with open("path-to-graph/facebook_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
