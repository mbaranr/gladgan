"""
GLocalKD and GLADC under gladGAN's protocol, on the same datasets/folds/seeds/features as gladGAN's runs.

    python baselines/reproduce_unified.py --out runs/unified
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = ([(ds, 'label', ('cv_fold', k)) for ds in ['BZR', 'COX2', 'DHFR', 'AIDS'] for k in range(5)] +
          [('NCI1', 'deg', ('cv_fold', k)) for k in range(5)] +
          [(ds, 'deg', ('seed', s)) for ds in ['Tox21_HSE', 'Tox21_MMP', 'Tox21_p53'] for s in range(3)])

def native_feat(ds):
    name = ds + '_training' if ds.startswith('Tox21') else ds
    attrs = os.path.join(os.path.dirname(HERE), 'datasets', name, f'{name}_node_attributes.txt')
    return 'default' if os.path.exists(attrs) else 'deg'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--methods', nargs='+', default=['glocalkd', 'gladc'])
    p.add_argument('--native', action='store_true',
                   help="each baseline's native features: node attributes if the dataset has them, else degree")
    p.add_argument('--no_val', action='store_true', help='skip scoring the validation split')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='run these datasets with atom-type features (CV folds, or seeds 0-2 for Tox21) instead of MATRIX')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    matrix = MATRIX if not a.datasets else \
        [(ds, native_feat(ds) if a.native else 'label', ('seed', s) if ds.startswith('Tox21') else ('cv_fold', s))
         for ds in a.datasets for s in (range(3) if ds.startswith('Tox21') else range(5))]
    for ds, feat, (kind, v) in matrix:
        for method in a.methods:
            tag = f'{method}_{ds}_{feat}_{kind}{v}'
            out = os.path.join(a.out, tag + '.json')
            if os.path.exists(out):
                continue
            cmd = [sys.executable, os.path.join(HERE, 'run_unified.py'), method, '--DS', ds, '--feat', feat,
                   f'--{kind}', str(v), '--out', out] + (['--no_val'] if a.no_val else [])
            with open(os.path.join(a.out, tag + '.log'), 'w') as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(tag, open(out).read() if os.path.exists(out) else f'FAILED rc={r.returncode}', flush=True)
