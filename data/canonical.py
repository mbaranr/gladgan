import networkx as nx
import numpy as np


def _node_label(G, u):
    lab = G.nodes[u].get('label')
    return int(np.argmax(lab)) if lab is not None else 0


def canonical_order(G, use_labels=True):
    """
    Deterministic node order, so that a graph's adjacency matrix has a (nearly) fixed layout,
    the way an image has a fixed pixel grid.

    1. colour refinement (1-WL) starting from (degree, node label) gives structural colours;
    2. components are taken largest first; within each, BFS from the lowest-colour node,
       visiting neighbours by colour. BFS also keeps the adjacency matrix banded.
    Ties between nodes with identical colours (often automorphic) fall back to input order.
    """
    nodes = list(G.nodes())
    col = {u: (G.degree(u), _node_label(G, u) if use_labels else 0) for u in nodes}
    n_colours = len(set(col.values()))
    for _ in range(len(nodes)):
        sig = {u: (col[u], tuple(sorted(col[v] for v in G[u]))) for u in nodes}
        rank = {k: i for i, k in enumerate(sorted(set(sig.values())))}
        new = {u: rank[sig[u]] for u in nodes}
        stable = len(set(new.values())) == n_colours
        col, n_colours = new, len(set(new.values()))
        if stable:
            break
    if not all(isinstance(c, int) for c in col.values()):
        rank = {k: i for i, k in enumerate(sorted(set(col.values())))}
        col = {u: rank[col[u]] for u in nodes}

    pos = {u: i for i, u in enumerate(nodes)}
    key = lambda u: (col[u], pos[u])
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(col[u] for u in c)))
    order = []
    for comp in comps:
        root = min(comp, key=key)
        seen, queue = {root}, [root]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in sorted(G[u], key=key):
                if v not in seen:
                    seen.add(v); queue.append(v)
    return order


def reorder(G, use_labels=True):
    """Copy of G whose node iteration order (and therefore adjacency matrix layout) is canonical."""
    order = canonical_order(G, use_labels)
    idx = {u: i for i, u in enumerate(order)}
    H = nx.Graph()
    H.graph.update(G.graph)
    for u in order:
        H.add_node(idx[u], **G.nodes[u])
    H.add_edges_from((idx[u], idx[v]) for u, v in G.edges())
    return H
