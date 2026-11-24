import copy
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_processing import generate_graphs
from utils.operations import sample_z, update_lr, save_checkpoint, load_checkpoint
from train.train_encoder import train_encoder
from train.anomaly_detection import detect_anomalies
from utils.logs import log_graph, flush_graph_log


def train_wgan(G: nn.Module,
               D: nn.Module,
               E: nn.Module,
               g_optimizer: torch.optim.Optimizer,
               d_optimizer: torch.optim.Optimizer,
               e_optimizer: torch.optim.Optimizer,
               start_epoch: int,
               args: argparse.Namespace,
               data_loader_train: torch.utils.data.DataLoader,
               data_loader_test: torch.utils.data.DataLoader,
               device: torch.device,
               z_dim: int,
               save_iters: list,
               checkpoint_dir: str,
               lambda_gp: float,
               kappa: float
               ):

    losses = {'G': [], 'D': []}

    best_gloss = np.inf
    epochs_no_improve = 0
    avg_dloss = 0.0
    avg_gloss = 0.0
    step = 0

    for epoch in range(start_epoch, args.wgan_epochs):

        G.train()
        D.train()
        E.train()

        total_dloss, total_gloss = 0.0, 0.0
        d_steps, g_steps = 0, 0
        logged_this_epoch = False

        for batch in data_loader_train:
            adj = batch['adj'].float().to(device)
            x = batch['feat'].float().to(device)
            n = batch['num_nodes'].to(device)
            current_batch_size = adj.shape[0]

            # critic: one update per batch
            z = sample_z(current_batch_size, z_dim).to(device)
            dloss = train_d_step(G, D, d_optimizer, x, adj, z, args, device, lambda_gp, n)
            total_dloss += dloss
            d_steps += 1

            # generator: one update every n_critic critic updates
            if step % args.n_critic == 0:
                z = sample_z(current_batch_size, z_dim).to(device)
                gloss, adj_hat = train_g_step(G, D, g_optimizer, z, args, E, e_optimizer, n[torch.randperm(len(n), device=n.device)])
                total_gloss += gloss
                g_steps += 1
                if args.log_graphs and not logged_this_epoch:
                    log_graph(adj_hat, gloss, dloss, epoch)
                    logged_this_epoch = True

            step += 1

        avg_dloss = total_dloss / max(d_steps, 1)
        if g_steps > 0:
            avg_gloss = total_gloss / g_steps

        if not args.quiet:
            print(f'\rWGAN Training | Epoch {epoch + 1}/{args.wgan_epochs} | Generator Loss: {avg_gloss:.4f} | Discriminator Loss: {avg_dloss:.4f}', end='')

        losses['G'].append(avg_gloss)
        losses['D'].append(avg_dloss)

        # early stopping (note: WGAN generator loss is not a reliable convergence signal)
        if args.early_stop:
            if avg_gloss < best_gloss:
                best_gloss = avg_gloss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                if not args.quiet:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in generator loss.", end='')
                break

        # decay learning rate
        if (epoch + 1) % args.lr_update_step == 0 and (epoch + 1) > (args.wgan_epochs - args.epochs_decay):
            args.g_lr -= args.g_lr / float(args.epochs_decay)
            args.d_lr -= args.d_lr / float(args.epochs_decay)
            update_lr(g_optimizer, d_optimizer, args.g_lr, args.d_lr)

        # checkpoint + intermediate evaluation
        if (epoch + 1) in save_iters:
            save_checkpoint(G, g_optimizer, epoch + 1, avg_gloss, os.path.join(checkpoint_dir, 'G_checkpoint_{}.pth'.format(epoch + 1)))
            save_checkpoint(D, d_optimizer, epoch + 1, avg_dloss, os.path.join(checkpoint_dir, 'D_checkpoint_{}.pth'.format(epoch + 1)))
            if args.log_graphs:
                flush_graph_log()

            if args.no_intermediate_eval:
                continue

            E_initial_state = copy.deepcopy(E.state_dict())
            e_optimizer_initial_state = copy.deepcopy(e_optimizer.state_dict())

            if not args.quiet:
                print()
            train_encoder(G, D, E, e_optimizer, args.encoder_epochs, data_loader_train, device, kappa, checkpoint_dir, args, z_dim)
            load_checkpoint(E, e_optimizer, os.path.join(checkpoint_dir, 'E_checkpoint_final.pth'))
            detect_anomalies(G, E, D, data_loader_test, kappa, device, args, epoch + 1, train_loader=data_loader_train)

            # roll the encoder back so the final encoder training starts fresh
            E.load_state_dict(E_initial_state)
            e_optimizer.load_state_dict(e_optimizer_initial_state)

    if not args.quiet:
        print()
    if args.log_graphs:
        flush_graph_log()
    save_checkpoint(G, g_optimizer, args.wgan_epochs, avg_gloss, os.path.join(checkpoint_dir, 'G_checkpoint_final.pth'))
    save_checkpoint(D, d_optimizer, args.wgan_epochs, avg_dloss, os.path.join(checkpoint_dir, 'D_checkpoint_final.pth'))
    if args.info_weight > 0:
        # the jointly trained encoder; the encoder stage starts from it
        save_checkpoint(E, e_optimizer, args.wgan_epochs, 0.0, os.path.join(checkpoint_dir, 'E_checkpoint_wgan.pth'))

    return losses


def gradient_penalty(D, x, adj, x_hat, adj_hat):
    """
    WGAN-GP penalty on random interpolations between real and fake graphs. The critic sees
    (x, adj) jointly, so the gradient norm is taken over both inputs together.
    """
    eps = torch.rand(x.shape[0], 1, 1, device=x.device)
    x_int = (eps * x + (1. - eps) * x_hat).requires_grad_(True)
    adj_int = (eps * adj + (1. - eps) * adj_hat).requires_grad_(True)

    score, _ = D(x_int, adj_int)
    grad_x, grad_adj = torch.autograd.grad(outputs=score.sum(),
                                           inputs=[x_int, adj_int],
                                           create_graph=True)
    sq_norm = grad_x.flatten(1).pow(2).sum(1) + grad_adj.flatten(1).pow(2).sum(1)
    grad_norm = torch.sqrt(sq_norm + 1e-12)
    return ((grad_norm - 1.) ** 2).mean()


def train_d_step(G, D, d_optimizer, x, adj, z, args, device, lambda_gp, n=None):

    with torch.no_grad():
        adj_hat, x_hat = generate_graphs(G, z, args.gumbell_type, args.feat, n=n)

    real_score, _ = D(x, adj)
    fake_score, _ = D(x_hat, adj_hat)

    gp = gradient_penalty(D, x, adj, x_hat, adj_hat)
    # drift penalty (Karras et al. 2018) keeps the critic's output, and so its features, from drifting in scale
    dloss = fake_score.mean() - real_score.mean() + lambda_gp * gp + args.drift * (real_score ** 2).mean()

    d_optimizer.zero_grad()
    dloss.backward()
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
    d_optimizer.step()

    if torch.isnan(dloss):
        raise FloatingPointError("NaN in critic loss")

    return dloss.item()


def train_g_step(G, D, g_optimizer, z, args, E=None, e_optimizer=None, n=None):

    adj_hat, x_hat = generate_graphs(G, z, args.gumbell_type, args.feat, n=n)
    fake_score, _ = D(x_hat, adj_hat)
    gloss = -fake_score.mean()
    loss = gloss

    # latent regression (InfoGAN / BiGAN style): G and E jointly keep z recoverable from G(z).
    # Without it a graph generator can become a hash of z (edge logits near 0, so tiny changes
    # in z flip many edges), which no encoder can invert afterwards.
    use_info = args.info_weight > 0 and E is not None
    if use_info:
        loss = loss + args.info_weight * F.mse_loss(E(x_hat, adj_hat), z)
        e_optimizer.zero_grad()

    g_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
    g_optimizer.step()
    if use_info:
        e_optimizer.step()

    # D's grads from this step are discarded by d_optimizer.zero_grad() in the next critic step
    return gloss.item(), adj_hat.detach().cpu().numpy()
