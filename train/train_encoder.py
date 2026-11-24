import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_processing import generate_graphs
from utils.operations import save_checkpoint, sample_z


def encoder_loss(G, D, E, x, adj, kappa, args, z_dim, n=None):
    """
    izi_f    : real graph -> E -> G -> fake graph; adjacency + feature MSE + kappa * critic-feature MSE
               (original f-AnoGAN; the MSE terms assume both graphs share a node ordering)
    izi_feat : same path, critic-feature MSE only (permutation invariant)
    ziz      : z ~ N(0, I) -> G -> E -> z_hat; latent MSE (never needs real graphs, permutation invariant)
    izi_f+ziz: sum of izi_f and ziz (reconstruction training alone degrades E as an inverse of G)
    """
    def ziz():
        z = sample_z(adj.shape[0], z_dim).to(adj.device)
        with torch.no_grad():
            adj_fake, x_fake = generate_graphs(G, z, args.gumbell_type, args.feat, n=n)
        return F.mse_loss(E(x_fake, adj_fake), z)

    if args.encoder_mode == 'ziz':
        return ziz()

    z_hat = E(x, adj)
    adj_tilde, x_tilde = generate_graphs(G, z_hat, args.gumbell_type, args.feat, n=n)
    with torch.no_grad():
        _, real_emb = D(x, adj)
    _, fake_emb = D(x_tilde, adj_tilde)

    # relative to the scale of the real features, so this term cannot swamp the residuals
    loss = kappa * F.mse_loss(fake_emb, real_emb) / (real_emb.pow(2).mean() + 1e-8)
    if args.encoder_mode in ('izi_f', 'izi_f+ziz'):
        loss = loss + F.mse_loss(adj_tilde, adj) + F.mse_loss(x_tilde, x)
    if args.encoder_mode == 'izi_f+ziz':
        loss = loss + ziz()
    return loss


def train_encoder(G: nn.Module,
                  D: nn.Module,
                  E: nn.Module,
                  e_optimizer: torch.optim.Optimizer,
                  epochs: int,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  kappa: float,
                  checkpoint_dir: str,
                  args: argparse.Namespace,
                  z_dim: int = 8
                  ):

    G.eval()
    D.eval()
    E.train()

    # G and D stay fixed; only E's parameters are in e_optimizer
    for p in list(G.parameters()) + list(D.parameters()):
        p.requires_grad_(False)

    avg_eloss = 0.0
    try:
        for epoch in range(epochs):
            total_eloss = 0.0
            batch_count = 0

            for batch in data_loader:
                adj = batch['adj'].float().to(device)
                x = batch['feat'].float().to(device)

                eloss = encoder_loss(G, D, E, x, adj, kappa, args, z_dim, batch['num_nodes'].to(device))

                e_optimizer.zero_grad()
                eloss.backward()
                e_optimizer.step()

                total_eloss += eloss.item()
                batch_count += 1

            avg_eloss = total_eloss / max(batch_count, 1)
            if not args.quiet:
                print(f'\rEncoder Training ({args.encoder_mode}) | Epoch {epoch + 1}/{epochs} | Encoder Loss: {avg_eloss:.4f}', end='')
    finally:
        for p in list(G.parameters()) + list(D.parameters()):
            p.requires_grad_(True)

    if not args.quiet:
        print()
    save_checkpoint(E, e_optimizer, epochs, avg_eloss, os.path.join(checkpoint_dir, 'E_checkpoint_final.pth'))
