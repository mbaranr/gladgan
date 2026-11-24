"""
Train one WGAN per (dataset, seed), then train each encoder mode on top of it and collect the
AUC of every score component. Results go to <out>/<dataset>/seed<k>/bin/aucs.json and a summary
table is printed at the end.

    python exp/sweep.py --out runs/sweep --seeds 0 1 2 --jobs 3
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEV = ['BZR', 'COX2', 'DHFR', 'AIDS', 'NCI1', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_p53']
HELDOUT = ['MUTAG', 'PTC_MR', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'Mutagenicity', 'NCI109', 'MCF-7', 'PC-3',
           'Tox21_AhR', 'Tox21_ARE', 'Tox21_ATAD5', 'Tox21_ER', 'Tox21_ER-LBD', 'Tox21_PPAR-gamma', 'Tox21_aromatase']
DATASETS = {ds: 'label' for ds in DEV + HELDOUT}   # all molecules: one-hot atom types
MODES = ['izi_f', 'izi_feat', 'ziz']
DEFAULT_MODES = ['izi_f+ziz']


def run(cmd, cwd, log):
    with open(log, 'a') as f:
        f.write(' '.join(cmd) + '\n'); f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'failed ({r.returncode}): see {log}')


def job(ds, seed, a, fold=None):
    d = Path(a.out) / ds / (f'fold{fold}' if fold is not None else f'seed{seed}')
    if (d / 'bin' / 'aucs.json').exists() and len(json.load(open(d / 'bin' / 'aucs.json'))) >= len(a.modes):
        return ds, seed, 'cached'
    d.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(REPO / 'main.py'), '--DS', ds, '--feat', a.feat or DATASETS[ds], '--seed', str(seed),
            *(['--cv_fold', str(fold)] if fold is not None else []),
            '--quiet', '--encoder_steps', str(a.encoder_steps), '--device', a.device] + a.extra.split()
    for i, mode in enumerate(a.modes):
        extra = ['--wgan_steps', str(a.wgan_steps), '--no_intermediate_eval'] if i == 0 else ['--wgan_pretrained']
        run(base + extra + ['--encoder_mode', mode], d, d / 'log.txt')
    return ds, (f'fold{fold}' if fold is not None else seed), 'done'


def summarize(a):
    rows = {}
    for f in sorted(list(Path(a.out).glob('*/seed*/bin/aucs.json')) + list(Path(a.out).glob('*/fold*/bin/aucs.json'))):
        ds = f.parts[-4]
        for k, aucs in json.load(open(f)).items():
            parts = k.split('/')
            split, mode = (parts[0] if len(parts) == 3 else 'test'), parts[-1]
            for metric, v in aucs.items():
                rows.setdefault((ds, split, mode, metric), []).append(v)
    metrics = a.metrics
    print(f"{'dataset':<10} {'split':<5} {'encoder':<9} " + ' '.join(f'{m.split("/")[-1]:>15}' for m in metrics) + '   n')
    for ds in DATASETS:
        for split in ('val', 'test'):
            for mode in MODES + ['izi_f+ziz']:
                vals = [rows.get((ds, split, mode, m), []) for m in metrics]
                if not vals[0]:
                    continue
                print(f'{ds:<10} {split:<5} {mode:<9} ' + ' '.join(f'{np.mean(v):.3f} ± {np.std(v):.3f}'.rjust(15) for v in vals) + f'   {len(vals[0])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/sweep')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--folds', nargs='+', type=int, default=None, help='5-fold CV folds to run (one seed each) instead of seeds')
    p.add_argument('--wgan_steps', type=int, default=10000)
    p.add_argument('--encoder_steps', type=int, default=1500)
    p.add_argument('--modes', nargs='+', default=DEFAULT_MODES, choices=MODES + ['izi_f+ziz'])
    p.add_argument('--extra', default='', help='extra arguments passed to main.py')
    p.add_argument('--feat', default=None, help='override node feature type for all datasets')
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--device', default='auto')
    p.add_argument('--summary_only', action='store_true')
    p.add_argument('--auto', action='store_true', help='protocol: CV folds for TU datasets, seeds 0-2 for Tox21')
    p.add_argument('--metrics', nargs='+', default=['component/size', 'component/sizedev', 'score/recon+critic', 'score/recon+critic|size', 'score/recon+critic|size+sizedev'])
    a = p.parse_args()
    a.out = os.path.abspath(a.out)
    if not a.summary_only:
        with ThreadPoolExecutor(a.jobs) as ex:
            if a.auto:   # protocol: 5-fold CV for TU datasets, 3 seeds on the official split for Tox21
                futs = [ex.submit(job, ds, s, a) if ds.startswith('Tox21') else ex.submit(job, ds, 0, a, s)
                        for ds in a.datasets for s in ((0, 1, 2) if ds.startswith('Tox21') else range(5))]   # in listed order
            elif a.folds:
                futs = [ex.submit(job, ds, a.seeds[0], a, k) for ds in a.datasets for k in a.folds]
            else:
                futs = [ex.submit(job, ds, s, a) for ds in a.datasets for s in a.seeds]
            for fu in futs:
                try:
                    print(*fu.result(), flush=True)
                except Exception as e:
                    print('ERROR', e, flush=True)
    summarize(a)
