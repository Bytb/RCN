# Contents of run_rnbrw_batch.py 

import sys 

import numpy as np 

import networkx as nx 

from rnbrw.weights import compute_weights 

from multiprocessing import Pool 

import os 

import pickle

def compute_single_walk(args): 

    graph_path, seed = args 

    with open(graph_path, 'rb') as f:
        G = pickle.load(f) 

    G = compute_weights(G, nsim=1, seed=seed, only_walk=True) 

    m = G.number_of_edges() 

    T = np.zeros(m) 

    for u, v in G.edges(): 
        T[G[u][v]['enum']] = G[u][v]['ret']

    return T 

 

job_id = int(sys.argv[1]) 

graph_path = "path-to-graph/facebook_graph.gpickle"

walks_per_job = 1760

seeds = [1000 + job_id * walks_per_job + i for i in range(walks_per_job)] 

 

with Pool(processes=32) as pool: 

    T_partial = pool.map(compute_single_walk, [(graph_path, seed) for seed in seeds]) 

 

T = np.sum(T_partial, axis=0) 

np.save(f"path-to-save-weights/weights/T_partial_{job_id}.npy", T)
