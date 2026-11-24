import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.graph_processing import generate_graphs
from utils.operations import save_scores_to_file

COMPONENTS = ['adj', 'x', 'feat', 'z', 'critic', 'size']

# combined scores: each component is standardized with mean/std over the (normal) training graphs
COMBINED = {
    'feat': ['feat'],
    'critic': ['critic'],
    'critic+feat': ['critic', 'feat'],
    'critic+feat+z': ['critic', 'feat', 'z'],
    'recon': ['adj', 'x'],                       # f-AnoGAN residual (meaningful with --canonical)
    'recon+critic': ['adj', 'x', 'critic'],
    'recon+critic+z': ['adj', 'x', 'critic', 'z'],
}
# size-adjusted variants: each component minus what node count predicts for it (fit on training normals),
# i.e. "unusual for a graph of this size". Without this, residuals largely measure graph size.
ADJUSTED = {f'{k}|size': v for k, v in COMBINED.items()}

# size added back in as an explicit, label-free component: how unusual the node count is for a normal
# training graph, in either direction (|log n - mean| / std over training normals). The direction of
# the size effect differs between datasets, so it must be two-sided to need no labels.
SIZEDEV_WEIGHTS = [0., 0.5, 1., 2.]


def select_sizedev_weight(val_aucs):
    """Weight of the size term with the best validation AUC (ties -> smaller weight)."""
    return max(SIZEDEV_WEIGHTS, key=lambda w: (round(val_aucs[f'score/recon+critic|size+sizedev@{w:g}'], 6), -w))


def size_dev(size, train_size):
    ln, lt = np.log(np.maximum(size, 1)), np.log(np.maximum(train_size, 1))
    return np.abs(ln - lt.mean()) / (lt.std() + 1e-8)


def score_components(G, E, D, data_loader, device, args):
    """
    Per-graph anomaly score components (higher = more anomalous):
      adj, x  : reconstruction squared error of G(E(x)) vs x per real node of x
                (meaningful with --canonical, which fixes the node order)
      feat    : MSE between critic features of x and G(E(x))      (permutation invariant)
      z       : MSE between E(x) and E(G(E(x)))                    (permutation invariant)
      critic  : negative critic score of x; the critic is trained to score normal graphs high
    """
    G.eval(); E.eval(); D.eval()
    out = {k: [] for k in COMPONENTS}
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            adj = batch['adj'].float().to(device)
            x = batch['feat'].float().to(device)

            z_hat = E(x, adj)
            adj_tilde, x_tilde = generate_graphs(G, z_hat, args.gumbell_type, args.feat, deterministic=True,
                                                 n=batch['num_nodes'].to(device))

            real_score, real_emb = D(x, adj)
            _, fake_emb = D(x_tilde, adj_tilde)
            z_tilde = E(x_tilde, adj_tilde)

            # squared errors per real node of x (averaging over the padded matrix made the residual a proxy
            # for graph size)
            n_real = (adj.sum(-1) > 0).sum(-1).clamp(min=1).float()
            out['adj'] += ((adj_tilde - adj) ** 2).sum(dim=(1, 2)).div(n_real).tolist()
            out['x'] += ((x_tilde - x) ** 2).sum(dim=(1, 2)).div(n_real).tolist()
            out['feat'] += F.mse_loss(fake_emb, real_emb, reduction='none').mean(dim=1).tolist()
            out['z'] += F.mse_loss(z_tilde, z_hat, reduction='none').mean(dim=1).tolist()
            out['critic'] += (-real_score.squeeze(-1)).tolist()
            out['size'] += batch['num_nodes'].float().tolist()   # baseline: AUC of node count alone
            labels += batch['label'].tolist()
    return {k: np.array(v) for k, v in out.items()}, np.array(labels) != 0


def size_fit(train_comps):
    """Per component, least-squares fit of value ~ a + b*log(n) + c*log(n)^2 on training normals."""
    ln = np.log(np.maximum(train_comps['size'], 1))
    Xd = np.stack([np.ones_like(ln), ln, ln ** 2], 1)
    fits = {}
    for k, v in train_comps.items():
        if k == 'size':
            continue
        coef, *_ = np.linalg.lstsq(Xd, v, rcond=None)
        resid = v - Xd @ coef
        fits[k] = (coef, resid.std() + 1e-8)
    return fits


def adjust(comps, fits):
    ln = np.log(np.maximum(comps['size'], 1))
    Xd = np.stack([np.ones_like(ln), ln, ln ** 2], 1)
    return {k: (comps[k] - Xd @ coef) / sd for k, (coef, sd) in fits.items()}


def combine(comps, kappa, name, train_stats=None):
    if name == 'izi_f':
        # original f-AnoGAN style score, on raw scales
        return comps['adj'] + comps['x'] + kappa * comps['feat'] + comps['z']
    total = 0.
    for c in COMBINED[name]:
        mu, sd = train_stats[c]
        total = total + (comps[c] - mu) / sd
    return total


def detect_anomalies(G, E, D, data_loader, kappa, device, args, epoch=None,
                     file_path='scores.pkl', train_loader=None, aucs_path='bin/aucs.json', split='test'):

    comps, labels = score_components(G, E, D, data_loader, device, args)

    train_stats = adj_comps = None
    if train_loader is not None:
        train_comps, _ = score_components(G, E, D, train_loader, device, args)
        train_stats = {k: (v.mean(), v.std() + 1e-8) for k, v in train_comps.items()}
        adj_comps = adjust(comps, size_fit(train_comps))   # already standardized on training residuals
    elif args.score != 'izi_f':
        raise ValueError(f"score '{args.score}' needs train_loader for standardization")

    # AUC must be computed on continuous scores, not on thresholded predictions
    aucs = {f'component/{k}': roc_auc_score(labels, v) for k, v in comps.items()}
    aucs['score/izi_f'] = roc_auc_score(labels, combine(comps, kappa, 'izi_f'))
    if train_stats is not None:
        for name in COMBINED:
            aucs[f'score/{name}'] = roc_auc_score(labels, combine(comps, kappa, name, train_stats))
        unit = {k: (0., 1.) for k in adj_comps}
        for k, v in adj_comps.items():
            aucs[f'component/{k}|size'] = roc_auc_score(labels, v)
        for name, parts in COMBINED.items():
            aucs[f'score/{name}|size'] = roc_auc_score(labels, sum(adj_comps[c] for c in parts))
        sd = size_dev(comps['size'], train_comps['size'])
        aucs['component/sizedev'] = roc_auc_score(labels, sd)
        # recon+critic|size has 3 unit-variance parts; sizedev is put on the same scale
        base = sum(adj_comps[c] for c in COMBINED['recon+critic'])
        sd_unit = (sd - size_dev(train_comps['size'], train_comps['size']).mean()) / \
                  (size_dev(train_comps['size'], train_comps['size']).std() + 1e-8)
        for w in SIZEDEV_WEIGHTS:
            aucs[f'score/recon+critic|size+sizedev@{w:g}'] = roc_auc_score(labels, base + w * sd_unit)

    if args.score == 'recon+critic|size+sizedev':
        w = getattr(args, 'sizedev_w', 0.)
        auc_score = aucs[f'score/recon+critic|size+sizedev@{w:g}']
        if split == 'test':
            aucs['score/recon+critic|size+sizedev'] = auc_score
            aucs['sizedev_w'] = w
    else:
        auc_score = aucs[f'score/{args.score}']
    key = epoch if epoch is not None else 'final'

    if not args.quiet:
        prefix = (f"AUC score at epoch {epoch}" if epoch is not None else "Final AUC score") + f" [{split}]"
        print(f"{prefix} ({args.score}): {auc_score:.4f} ({int(labels.sum())} anomalous / {len(labels)} test graphs)")
        print("  " + "  ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in aucs.items()))

    if split == 'test':
        save_scores_to_file(file_path, key, auc_score)

    all_aucs = {}
    if os.path.exists(aucs_path):
        with open(aucs_path) as f:
            all_aucs = json.load(f)
    all_aucs[f'{split}/{key}/{args.encoder_mode}'] = aucs
    os.makedirs(os.path.dirname(aucs_path) or '.', exist_ok=True)
    with open(aucs_path, 'w') as f:
        json.dump(all_aucs, f, indent=1)
    return aucs

    return auc_score
