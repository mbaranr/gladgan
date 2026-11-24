"""
Runtime shim so the official GLADC / GLocalKD code runs unmodified on Apple silicon (or CPU)
with current library versions. Import this before their modules.

- .cuda() on tensors/modules -> .to(DEVICE)   (BASELINE_DEVICE env var: mps | cpu | cuda; default mps if available)
- networkx 3 removed to_numpy_matrix; their code also parses nx.__version__ as a float
- GLocalKD's main.py imports tdc (PyTDC) but does not use it for TU datasets
- both util.py files import community (python-louvain) without using it
- GLADC imports torch_geometric without using it
"""
import os
import sys
import types

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(os.environ.get('BASELINE_DEVICE') or
                      ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))

if DEVICE.type != 'cuda':
    torch.Tensor.cuda = lambda self, *a, **k: self.to(DEVICE)
    nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)

if not hasattr(nx, 'to_numpy_matrix'):
    nx.to_numpy_matrix = lambda G, *a, **k: np.asmatrix(nx.to_numpy_array(G, *a, **k))
nx.__version__ = '2.5'   # their node_iter/node_dict do float(nx.__version__); '2.5' selects the modern API

if 'tdc' not in sys.modules:
    tdc = types.ModuleType('tdc'); tdc_utils = types.ModuleType('tdc.utils')
    tdc_utils.retrieve_label_name_list = None
    tdc.utils = tdc_utils
    sys.modules['tdc'] = tdc; sys.modules['tdc.utils'] = tdc_utils

if 'community' not in sys.modules:
    sys.modules['community'] = types.ModuleType('community')   # imported but unused by both repos

if 'torch_geometric' not in sys.modules:
    # GLADC imports GINConv/global_add_pool from torch_geometric but never uses them (the call is commented out)
    tg = types.ModuleType('torch_geometric'); tg_nn = types.ModuleType('torch_geometric.nn')
    tg_nn.GINConv = tg_nn.global_add_pool = None
    tg.nn = tg_nn
    sys.modules['torch_geometric'] = tg; sys.modules['torch_geometric.nn'] = tg_nn
