"""
Health check for a trained run (forward passes only, no training).

    python tools/diagnose.py <run_dir> --DS COX2 --feat label [--cv_fold 0]

Answers the questions that loss curves don't:
  generator : do samples look like real graphs (size, edges)? are they diverse up to isomorphism?
              are edge decisions confident (probabilities near 0/1) or coin flips?
  encoder   : does E invert G on generated graphs (corr(E(G(z)), z))?
              is G(E(x)) closer to x than a random G(z) is (in critic-feature space)?
  scores    : AUC of every score component on the test set
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--DS', required=True)
p.add_argument('--feat', default='label')
p.add_argument('--gumbell_type', default='sigmoid')
p.add_argument('--device', default='auto')
p.add_argument('--n', type=int, default=512)
a, rest = p.parse_known_args()

sys.argv = ['main.py', '--DS', a.DS, '--feat', a.feat, '--gumbell_type', a.gumbell_type, '--quiet'] + rest
import main as M
from models.encoder import Encoder
from models.gan import Discriminator, Generator
from train.anomaly_detection import score_components
from utils.graph_processing import generate_graphs
from utils.operations import sample_z
from utils.parser import parse_arguments
from sklearn.metrics import roc_auc_score

args = parse_arguments()
dev = torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
                   if a.device == 'auto' else a.device)
torch.manual_seed(0)

N, S, train, test, val = M.prepare_data(os.path.join(os.path.dirname(M.__file__), 'datasets/'), args)
fd = S.feat_dim
ck = lambda name: torch.load(os.path.join(a.run, 'checkpoints', f'{name}_checkpoint_final.pth'), weights_only=True)['model_state_dict']
gsd = ck('G')
zd = gsd['multidense_layer.0.weight'].shape[1] - int(args.size_cond)   # first layer also takes n when size-conditioned
G, D, E = M.build_models(args, N, fd, zd, dev)
for m, name in ((G, 'G'), (D, 'D'), (E, 'E')):
    m.load_state_dict(ck(name)); m.eval()
mode = args.gumbell_type
gen = lambda z, n=None: generate_graphs(G, z, mode, a.feat, deterministic=True, n=n)
d2 = lambda u, v: ((u - v) ** 2).mean(-1)

with torch.no_grad():
    # ---------------- generator
    real_n, real_e = [], []
    for b in train:
        adj = b['adj'].float()
        real_n += (adj.sum(-1) > 0).sum(-1).tolist(); real_e += (adj.sum((1, 2)) / 2).tolist()
    z = sample_z(a.n, zd).to(dev)
    n_s = torch.tensor(np.random.choice(real_n, a.n), device=dev)   # node counts from the real distribution
    logits, _ = G(z, n_s) if G.size_cond else G(z)
    adj_g, x_g = gen(z, n_s)
    iu = np.triu_indices(N, 1)
    pr = torch.sigmoid(logits).cpu().numpy()[:, iu[0], iu[1]]
    A, X = adj_g.cpu().numpy(), x_g.cpu().numpy()
    hashes = set()
    for ai, xi in zip(A, X):
        g = nx.from_numpy_array((ai > 0.5).astype(float)); g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
        if a.feat == 'label':
            nx.set_node_attributes(g, {n: str(int(xi[n].argmax())) for n in g}, 'atom')
        hashes.add(nx.weisfeiler_lehman_graph_hash(g, node_attr='atom' if a.feat == 'label' else None))
    gn = ((A > 0.5).sum(-1) > 0).sum(-1); ge = (A > 0.5).sum((1, 2)) / 2
    print(f"[generator] nodes {np.mean(gn):.1f}±{np.std(gn):.1f} (real {np.mean(real_n):.1f}±{np.std(real_n):.1f}) | "
          f"edges {np.mean(ge):.1f}±{np.std(ge):.1f} (real {np.mean(real_e):.1f}±{np.std(real_e):.1f}) | "
          f"unique up to isomorphism {len(hashes)}/{a.n}")
    print(f"[generator] edge confidence: {(np.minimum(pr, 1 - pr) < 0.1).mean():.3f} of edge probs within 0.1 of 0/1 "
          f"(frac in 0.2-0.8: {((pr > .2) & (pr < .8)).mean():.3f})")

    # ---------------- encoder
    zr = E(x_g, adj_g)
    corr = np.mean([np.corrcoef(zr[:, i].cpu(), z[:, i].cpu())[0, 1] for i in range(zd)])
    print(f"[encoder]   E(G(z)) vs z: MSE {d2(zr, z).mean():.3f} (chance {z.var(0).mean():.3f}) | mean per-dim corr {corr:.2f}")
    rec, rnd, pair, lab = [], [], [], []
    for b in test:
        adj = b['adj'].float().to(dev); x = b['feat'].float().to(dev)
        _, fr = D(x, adj)
        nb = b['num_nodes'].to(dev)
        at, xt = gen(E(x, adj), nb); _, ft = D(xt, at)
        ag, xg = gen(sample_z(len(adj), zd).to(dev), nb); _, fg = D(xg, ag)
        rec += d2(ft, fr).tolist(); rnd += d2(fg, fr).tolist(); pair += d2(fr, fr.roll(1, 0)).tolist(); lab += b['label'].tolist()
    rec, rnd, pair, lab = map(np.array, (rec, rnd, pair, lab))
    m = lab == 0
    print(f"[encoder]   test normals, critic-feature distance: to G(E(x)) {rec[m].mean():.4f} | to random G(z) {rnd[m].mean():.4f} | "
          f"to another real graph {pair[m].mean():.4f} -> recon/random ratio {np.median(rec[m] / rnd[m]):.2f} (want << 1)")

# ---------------- scores
comps, labels = score_components(G, E, D, test, dev, args)
print("[scores]    AUC " + "  ".join(f"{k}={roc_auc_score(labels, v):.3f}" for k, v in comps.items()) +
      f"  ({int(labels.sum())} anomalous / {len(labels)})")
