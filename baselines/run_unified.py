"""
Run GLocalKD or GLADC under gladGAN's evaluation protocol: identical data pipeline (folds / Tox21 split,
majority class = normal, validation split, node features) via gladgan's prepare_data, and the AUC of the
final epoch. Models, losses, optimisers, schedules and anomaly scores are the official ones; the training
loops below are copied from the official main.py files, with only data handling changed.

    python baselines/run_unified.py glocalkd --DS BZR --feat label --cv_fold 0 --out result.json
    python baselines/run_unified.py gladc --DS Tox21_MMP --feat deg --seed 1 --out result.json
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import compat  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('method', choices=['glocalkd', 'gladc'])
p.add_argument('--DS', required=True)
p.add_argument('--feat', default='label', choices=['label', 'deg', 'default'])
p.add_argument('--cv_fold', type=int, default=None)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--out', required=True)
p.add_argument('--no_val', action='store_true', help='skip scoring the validation split')
p.add_argument('--lr', type=float, default=None, help='override the official learning rate')
a = p.parse_args()

# ---------------- data: gladGAN's pipeline (same splits/labels/features as gladGAN runs)
sys.path.insert(0, REPO)
sys_argv = sys.argv
sys.argv = ['main.py', '--DS', a.DS, '--feat', a.feat, '--seed', str(a.seed), '--no-canonical', '--quiet', '--batch_size', '300'] + \
           (['--cv_fold', str(a.cv_fold)] if a.cv_fold is not None else [])
from utils.parser import parse_arguments  # noqa: E402
import main as gg  # noqa: E402
gargs = parse_arguments()
sys.argv = sys_argv
# official repos import their own util/load_data; drop gladgan's from the module cache first
for m in [m for m in list(sys.modules) if m in ('util', 'load_data', 'utils', 'model', 'loss') or m.startswith('utils.')]:
    del sys.modules[m]

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
N, S, train_loader_gg, test_loader_gg, val_loader_gg = gg.prepare_data(os.path.join(REPO, 'datasets/'), gargs)
DEV = compat.DEVICE


def batches(loader_gg, batch_size, shuffle):
    """Re-batch gladGAN's datasets in the official format ('adj' raw padded, 'feats', 'adj_label' = adj + I)."""
    ds = loader_gg.dataset
    idx = list(range(len(ds)))
    if a.method == 'gladc':
        # Tox21 training files contain a few single-atom molecules that load as 0-node graphs; GLADC's
        # contrastive loss divides by their zero norm and its parameters go NaN. No test set contains any.
        idx = [i for i in idx if ds[i]['num_nodes'] > 0]
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        items = [ds[j] for j in idx[i:i + batch_size]]
        adj = torch.tensor(np.stack([it['adj'] for it in items]), dtype=torch.float32)
        yield {'adj': adj, 'feats': torch.tensor(np.stack([it['feat'] for it in items]), dtype=torch.float32),
               'adj_label': adj + torch.eye(adj.shape[-1]), 'label': torch.tensor([it['label'] for it in items])}


repo_dir = os.path.join(HERE, a.method)
sys.path.insert(0, repo_dir)
feat_dim = S.feat_dim


def evaluate(score_fn, loader_gg):
    """Official evaluation uses test batch size 1 (their BatchNorm layers are created per forward pass)."""
    scores, labels = [], []
    for data in batches(loader_gg, 1, False):
        scores.append(float(score_fn(data)))
        labels.append(int(data['label'][0] != 0))
    return roc_auc_score(labels, scores)


if a.method == 'glocalkd':
    import GCN_embedding
    oargs = argparse.Namespace(hidden_dim=512, output_dim=256, num_gc_layers=3, bn=True, dropout=0.3, bias=True,
                               clip=0.1, num_epochs=150, batch_size=300)
    teacher = GCN_embedding.GcnEncoderGraph_teacher(feat_dim, oargs.hidden_dim, oargs.output_dim, 2, oargs.num_gc_layers,
                                                    bn=oargs.bn, args=oargs).cuda()
    for prm in teacher.parameters():
        prm.requires_grad = False
    student = GCN_embedding.GcnEncoderGraph_student(feat_dim, oargs.hidden_dim, oargs.output_dim, 2, oargs.num_gc_layers,
                                                    bn=oargs.bn, args=oargs).cuda()
    optimizer = torch.optim.Adam(filter(lambda q: q.requires_grad, student.parameters()), lr=a.lr or 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    def score(data):
        adj, h0 = data['adj'].cuda(), data['feats'].cuda()
        embed_node, embed = student(h0, adj)
        embed_teacher_node, embed_teacher = teacher(h0, adj)
        loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
        loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
        return (loss_graph + loss_node).item()

    for epoch in range(oargs.num_epochs):          # official loop (main.py: train)
        student.train()
        for data in batches(train_loader_gg, oargs.batch_size, True):
            student.zero_grad()
            adj, h0 = data['adj'].cuda(), data['feats'].cuda()
            embed_node, embed = student(h0, adj)
            embed_teacher_node, embed_teacher = teacher(h0, adj)
            embed_teacher, embed_teacher_node = embed_teacher.detach(), embed_teacher_node.detach()
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0) + loss_node
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(student.parameters(), oargs.clip)
            optimizer.step()
            scheduler.step()
    student.eval()
    with torch.no_grad():
        res = {'test': evaluate(score, test_loader_gg)}
        if val_loader_gg is not None and not a.no_val:
            res['val'] = evaluate(score, val_loader_gg)

else:  # gladc
    from graph_autoencoder import NetGe, Encoder
    from loss import loss_func, loss_cal
    hidden_dim, output_dim, dropout, batch_size, lr, num_epochs = 256, 128, 0.1, 300, 0.0001, 100
    if a.DS.startswith('Tox21'):
        batch_size, lr = 2000, 0.00001   # GLADC's Tox21 settings (main_1.py); main.py's diverge to NaN on Tox21
    lr = a.lr or lr
    NetG = NetGe(feat_dim, hidden_dim, output_dim, dropout, batch_size).cuda()
    noise_NetG = Encoder(feat_dim, hidden_dim, output_dim, dropout, batch_size).cuda()
    optimizerG = torch.optim.Adam(NetG.parameters(), lr=lr)

    def gen_ran_output(h0, adj, model, vice_model):   # official (main.py)
        for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
            if name.split('.')[0] == 'proj_head':
                adv_param.data = param.data
            else:
                adv_param.data = param.data + 1.0 * torch.normal(0, torch.ones_like(param.data) * param.data.std()).cuda()
        return vice_model(h0, adj)

    def score(data):
        adj, h0 = data['adj'].cuda(), data['feats'].cuda()
        x1_r, Feat_0 = NetG.shared_encoder(h0, adj)
        x_fake, s_fake, x2, Feat_1 = NetG(x1_r, adj)
        loss_node = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
        loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
        return (loss_node + loss_graph).item()

    for epoch in range(num_epochs):                # official loop (main.py: train)
        NetG.train()
        for data in batches(train_loader_gg, batch_size, True):
            adj, h0, adj_label = data['adj'].cuda(), data['feats'].cuda(), data['adj_label'].cuda()
            x1_r, Feat_0 = NetG.shared_encoder(h0, adj)
            x1_r_1, Feat_0_1 = gen_ran_output(h0, adj, NetG.shared_encoder, noise_NetG)
            x_fake, s_fake, x2, Feat_1 = NetG(x1_r, adj)
            err_g_con_s, err_g_con_x = loss_func(adj_label, s_fake, h0, x_fake)
            node_loss = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1).mean(dim=0)
            err_g_enc = loss_cal(Feat_0_1, Feat_0)
            lossG = err_g_con_s + err_g_con_x + node_loss + graph_loss + err_g_enc
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
    NetG.eval()
    with torch.no_grad():
        res = {'test': evaluate(score, test_loader_gg)}
        if val_loader_gg is not None and not a.no_val:
            res['val'] = evaluate(score, val_loader_gg)

res.update(method=a.method, DS=a.DS, feat=a.feat, cv_fold=a.cv_fold, seed=a.seed,
           lr=a.lr or (0.0001 if a.method == 'glocalkd' or not a.DS.startswith('Tox21') else 0.00001))
json.dump(res, open(a.out, 'w'))
print(json.dumps(res))
