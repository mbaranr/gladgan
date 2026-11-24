import networkx as nx
import torch
import torch.nn.functional as F

_NX_MAJOR = int(nx.__version__.split('.')[0])

# legacy names used by older experiment scripts
_ADJ_MODE_ALIASES = {'hard-gumbell': 'hard', 'no-gumbell': 'sigmoid'}


def node_iter(G):
    return G.nodes() if _NX_MAJOR < 2 else G.nodes


def node_dict(G):
    return G.nodes if _NX_MAJOR >= 2 else G.node


def process_adj(adj_logits, method='hard', temperature=1.0, deterministic=False):
    """
    Turn symmetric edge logits [B, N, N] into a symmetric adjacency matrix with an empty diagonal.

    method:
      'sigmoid' -> edge probabilities, no sampling
      'soft'    -> relaxed Bernoulli (Gumbel-sigmoid) sample
      'hard'    -> binary Gumbel-sigmoid sample with a straight-through gradient
      'threshold' -> binary edges (logit > 0) with a straight-through gradient, no noise.
                   With noise, a generator can match edge counts with independent coin flips
                   (a random graph) and ignore z; without it, structure must come from z.
    deterministic: skip the noise (used at test time), 'hard' then thresholds the logits at 0.
    """
    method = _ADJ_MODE_ALIASES.get(method, method)
    if method == 'threshold':
        # binary edges with no sampling noise: all variety has to come from z
        method, deterministic = 'hard', True
    if method not in ('sigmoid', 'soft', 'hard'):
        raise ValueError(f"Unknown adjacency mode '{method}'")

    n = adj_logits.size(-1)
    triu = torch.triu(torch.ones(n, n, device=adj_logits.device), diagonal=1)
    off_diag = triu + triu.T

    logits = adj_logits
    if method != 'sigmoid' and not deterministic:
        # logistic noise, mirrored so the sample stays symmetric
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        noise = (torch.log(u) - torch.log1p(-u)) * triu
        logits = logits + noise + noise.transpose(1, 2)

    probs = torch.sigmoid(logits / temperature)
    if method == 'hard':
        hard = (probs > 0.5).float()
        probs = hard - probs.detach() + probs

    return probs * off_diag


def generate_graphs(G, z, adj_mode, feat_mode, deterministic=False, n=None):
    """
    Sample graphs from the generator. Returns (adj [B,N,N], x [B,N,F]).
    n: node counts [B], required when G is size-conditioned.

    Nodes without edges are treated as padding: real padded nodes have no edges and zero
    features, so generated isolated nodes get zero features too. With degree features
    ('deg') the features are computed from the generated adjacency instead of being a
    free generator output, which keeps them consistent with the edges.
    """
    adj_logits, x_logits = G(z, n) if getattr(G, 'size_cond', False) else G(z)
    adj = process_adj(adj_logits, adj_mode, deterministic=deterministic)
    if getattr(G, 'size_cond', False):
        # with canonical ordering a graph's real nodes are the first n slots
        slot = (torch.arange(adj.size(-1), device=adj.device).unsqueeze(0) < n.to(adj.device).unsqueeze(1)).float()
        adj = adj * slot.unsqueeze(1) * slot.unsqueeze(2)
    if feat_mode == 'deg':
        return adj, adj.sum(-1, keepdim=True)

    node_mask = (adj.detach().sum(-1, keepdim=True) > 0).float()
    if feat_mode == 'label':
        # categorical node labels (e.g. atom types): one-hot sample, straight-through gradient
        if deterministic or adj_mode == 'threshold':
            probs = F.softmax(x_logits, -1)
            hard = F.one_hot(x_logits.argmax(-1), x_logits.size(-1)).float()
            x = hard - probs.detach() + probs
        else:
            x = F.gumbel_softmax(x_logits, tau=1.0, hard=True)
    else:
        x = x_logits
    x = x * node_mask
    return adj, x


def normalize_adj(adj, eps=1e-9):
    """
    Row normalize adjacency matrix with epsilon to prevent division by zero.
    """
    rowsum = adj.sum(-1, keepdim=True)
    return adj / (rowsum + eps)
