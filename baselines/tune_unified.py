"""
Tuning parity for the baselines: learning-rate grid, chosen per fold on the labelled validation split
(the same split gladGAN has), atom-type features. Reports test AUC at the chosen learning rate.

    python baselines/tune_unified.py --out runs/tuned --datasets BZR COX2 DHFR --reuse results/unified
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [1e-3, 1e-4, 1e-5]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--methods', nargs='+', default=['glocalkd', 'gladc'])
    p.add_argument('--reuse', default=None, help='dir with existing default-lr (1e-4) unified runs that include val')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    if a.reuse:
        for f in glob.glob(os.path.join(a.reuse, '*.json')):
            r = json.load(open(f))
            if r['feat'] == 'label' and 'val' in r and r['cv_fold'] is not None:
                runs[(r['method'], r['DS'], r['cv_fold'], 1e-4)] = r
    for ds in a.datasets:
        for k in range(5):
            for method in a.methods:
                for lr in GRID:
                    key = (method, ds, k, lr)
                    out = os.path.join(a.out, f'{method}_{ds}_fold{k}_lr{lr:g}.json')
                    if key in runs:
                        continue
                    if not os.path.exists(out):
                        cmd = [sys.executable, os.path.join(HERE, 'run_unified.py'), method, '--DS', ds, '--feat', 'label',
                               '--cv_fold', str(k), '--lr', str(lr), '--out', out]
                        with open(out.replace('.json', '.log'), 'w') as f:
                            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                    if os.path.exists(out):
                        runs[key] = json.load(open(out))
                        print(method, ds, k, lr, 'val', round(runs[key]['val'], 3), flush=True)
    summary = {}
    for ds in a.datasets:
        for method in a.methods:
            chosen = []
            for k in range(5):
                cands = [(runs[(method, ds, k, lr)]['val'], lr) for lr in GRID if (method, ds, k, lr) in runs]
                if not cands:
                    continue
                v, lr = max(cands)
                chosen.append((lr, runs[(method, ds, k, lr)]['test']))
            tests = [t for _, t in chosen]
            summary[f'{method}:{ds}'] = {'test_mean': float(np.mean(tests)), 'test_std': float(np.std(tests)),
                                         'chosen_lr': [lr for lr, _ in chosen], 'test': tests}
            print(f'{ds:<6} {method:<9} tuned test {np.mean(tests):.3f} ± {np.std(tests):.3f}  lr per fold {[f"{lr:g}" for lr, _ in chosen]}')
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
