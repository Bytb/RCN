import sys
import numpy as np
import pickle
from rnbrw.weights import walk_hole_csr   # <-- fastest configurable CSR primitive

# ---------------------------------------------------------------------
# Load CSR data ONCE
# ---------------------------------------------------------------------
csr_data = np.load("path-to-CSR/Data/raw_data/PPI_csr.npz")
indptr = csr_data["indptr"]
indices = csr_data["indices"]
edge_list = csr_data["edge_list"]
m = int(csr_data["m"])

with open("path-to-edge-lookup/Data/raw_data/edge_lookup.pkl", "rb") as f:
    edge_lookup = pickle.load(f)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
job_id = int(sys.argv[1])
walks_per_job = 300

# Independent seeds for reproducibility
seeds = [1000 + job_id * walks_per_job + i for i in range(walks_per_job)]

# Preallocate accumulator
T = np.zeros(m, dtype=int)

# Run 300 independent walks (CSR backend is fastest)
for s in seeds:
    T += walk_hole_csr(indptr, indices, edge_lookup, edge_list, m, S=1, seed=s)

# Save partial results
out_path = f"path-to-weights/weights/T_partial_{job_id}.npy"
np.save(out_path, T)

print(f"Job {job_id} finished: {walks_per_job} walks → {T.sum()} hits, results saved to {out_path}")
