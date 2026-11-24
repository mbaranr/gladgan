import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConv


class Encoder(nn.Module):
    """
    Maps a graph to a latent vector z (inverse of the generator, as in f-AnoGAN).

    Permutation invariant: two graph convolutions, per-node MLP, then sum pooling over real
    (non-isolated) nodes. Like the critic, it also receives node degree as an input, since
    row-normalized aggregation cannot see it. The output is unbounded to cover the N(0, I) prior.
    """

    def __init__(self,
                 input_dim: int,
                 linear_dim: list,
                 z_dim: int,
                 dropout: float=.0,
                 device: torch.device=None,
                 num_nodes: int=None):
        super(Encoder, self).__init__()

        h = linear_dim[0]
        self.gcn1 = GraphConv(input_dim + 1, h, add_self=False, normalize_embedding=False, device=device)
        self.gcn2 = GraphConv(h, h, add_self=False, normalize_embedding=False, device=device)

        layers = []
        for c0, c1 in zip(linear_dim[:-1], linear_dim[1:]):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], z_dim)

        # optional order-aware branch over the flattened (canonically ordered) graph
        self.flat_layer = None
        if num_nodes is not None:
            self.flat_layer = nn.Sequential(nn.Linear(num_nodes * num_nodes + num_nodes * input_dim, 512), nn.LeakyReLU(),
                                            nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, z_dim))

    def forward(self, x, adj):
        deg = adj.sum(-1, keepdim=True)
        node_mask = (deg > 0).float()

        hidden = F.leaky_relu(self.gcn1(torch.cat((x, deg), -1), adj))
        hidden = F.leaky_relu(self.gcn2(hidden, adj))
        hidden = self.multidense_layer(hidden)
        hidden = self.output_layer(hidden)

        z = (hidden * node_mask).sum(dim=1)
        if self.flat_layer is not None:
            z = z + self.flat_layer(torch.cat((adj.flatten(1), x.flatten(1)), 1))
        return z
