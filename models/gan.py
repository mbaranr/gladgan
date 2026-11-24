from models.layers import *
import torch
from typing import Callable, Optional
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, 
                 conv_dim: list, 
                 z_dim: int, 
                 num_nodes: int, 
                 num_features: int, 
                 dropout: float=.0,
                 size_cond: bool=False):
        
        super(Generator, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
        # size conditioning: the node count n is observed, so G gets it as input instead of having to
        # sample it (and drop the rare large graphs); generate_graphs masks every slot past n
        self.size_cond = size_cond
 
        layers = []
        for c0, c1 in zip([z_dim + int(size_cond)] + conv_dim[:-1], conv_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.BatchNorm1d(c1))  # do not apply to critic
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.adj_layer = nn.Linear(conv_dim[-1], num_nodes * num_nodes)
        self.x_layer = nn.Linear(conv_dim[-1], num_nodes * num_features)

        self.dropout = dropout

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, z, n=None):
        if self.size_cond:
            z = torch.cat((z, (n.float() / self.num_nodes).unsqueeze(-1)), -1)

        output = self.multidense_layer(z)

        # shape [batch_size, num_nodes, num_nodes]
        adj_logits = self.adj_layer(output).view(-1, self.num_nodes, self.num_nodes)

        # make the adjacency matrix symmetric by averaging it with its transpose
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2
        if self.dropout > 0.001:
            adj_logits = self.dropout_layer(adj_logits)
            
        # shape [batch_size, num_nodes, num_features]
        x_logits = self.x_layer(output).view(-1, self.num_nodes, self.num_features)
        if self.dropout > 0.001:
            x_logits = self.dropout_layer(x_logits)

        return adj_logits, x_logits



class Discriminator(nn.Module):
    def __init__(self, 
                 conv_dim: list,
                 aggr_dim: int, 
                 linear_dim: list,
                 device: torch.device=None,
                 dropout: float=.0,
                 activation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                 num_nodes: Optional[int]=None):
        # WGAN critic: output must be unbounded, so no activation by default
    
        super(Discriminator, self).__init__()

        gc_input_dim, gc_output_dim = conv_dim    # gc_output_dim is a list
        # the critic also sees each node's degree: row-normalized (mean) aggregation is blind to
        # how many neighbours a node has, so without it the generator gets no signal on density
        gc_input_dim = gc_input_dim + 1

        # no per-node L2 normalization in the critic: padded nodes are exact zero vectors and
        # normalizing them has an unbounded gradient, which makes the gradient penalty inf
        self.gcn_layer = GraphConv_(gc_input_dim, gc_output_dim, dropout=dropout, device=device, normalize_embedding=False)
        self.agg_layer = GraphAggr(gc_output_dim[-1]+gc_input_dim, aggr_dim, dropout=dropout)

        # optional order-aware branch: an MLP over the flattened (canonically ordered) adjacency and
        # features. The GNN branch cannot see node layout, so without this branch the generator is free
        # to store information in where nodes are placed, which an encoder then cannot read back.
        self.flat_layer = None
        dense_in = aggr_dim
        if num_nodes is not None:
            self.flat_layer = nn.Sequential(nn.Linear(num_nodes * num_nodes + num_nodes * gc_input_dim, 512), nn.LeakyReLU(),
                                            nn.Linear(512, aggr_dim), nn.LeakyReLU())
            dense_in = 2 * aggr_dim

        # multi dense layer
        layers = []
        for c0, c1 in zip([dense_in]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            # layers.append(nn.LayerNorm(c1))     # layer norm -> drop-in replacement to batch norm
            layers.append(nn.Dropout(dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

        self.activation = activation

    def forward(self, x, adj):

        x = torch.cat((x, adj.sum(-1, keepdim=True)), -1)
        hidden = self.gcn_layer(x, adj)
        hidden = torch.cat((hidden, x), -1)
        hidden = self.agg_layer(hidden, torch.tanh)
        if self.flat_layer is not None:
            hidden = torch.cat((hidden, self.flat_layer(torch.cat((adj.flatten(1), x.flatten(1)), 1))), -1)

        hidden = self.multidense_layer(hidden)

        output = self.output_layer(hidden)
        if self.activation is not None:
            output = self.activation(output)

        return output, hidden

