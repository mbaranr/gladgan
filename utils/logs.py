import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

_buffer = []


def log_graph(adj_mat: np.ndarray, gloss: float, dloss: float, epoch: int, n_samples: int = 3):
    """
    Buffer a few generated graphs (adjacency matrices) from a batch. Call flush_graph_log() to write.
    """
    import jsonpickle  # only needed with --log_graphs
    batch_size = adj_mat.shape[0]
    for i in np.random.choice(batch_size, size=min(n_samples, batch_size), replace=False):
        _buffer.append({
            "epoch": epoch,
            "gloss": gloss,
            "dloss": dloss,
            "graph_id": int(i),
            "graph": jsonpickle.encode(nx.from_numpy_array(adj_mat[i])),
        })


def flush_graph_log(out_file: str = "bin/graphs.parquet"):
    global _buffer
    if not _buffer:
        return
    df = pd.DataFrame(_buffer)
    if Path(out_file).exists():
        df = pd.concat([pd.read_parquet(out_file), df], ignore_index=True)
    df.to_parquet(out_file, index=False)
    _buffer = []
