import numpy as np
import os

num_jobs = 100  # Match --array size
weight_dir = "weights"
T_partial_files = [f for f in os.listdir(weight_dir) if f.startswith("T_partial_") and f.endswith(".npy")]
T_total = sum(np.load(os.path.join(weight_dir, f)) for f in T_partial_files)
np.save("weights/T_total.npy", T_total)
