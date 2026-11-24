"""
Assemble the comparison table from raw results.

    python exp/final_table.py --gladgan runs/final_cv runs/final_plain --unified runs/unified --released runs/released
"""
import argparse
import glob
import json
import os

import numpy as np

DATASETS = ['BZR', 'COX2', 'DHFR', 'AIDS', 'NCI1', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_p53']
SHORT = {'Tox21_HSE': 'HSE', 'Tox21_MMP': 'MMP', 'Tox21_p53': 'p53'}

# project page (blog) "Plain Graphs" table: best of dummy/heuristic/DT/RF100/RF400, and GLADC
BLOG_BEST_SIMPLE = {'Tox21_MMP': ('0.500', 'dummy'), 'Tox21_HSE': ('0.557', 'RF100'),
                    'Tox21_p53': ('0.589', 'DT'), 'NCI1': ('0.693', 'RF')}
BLOG_GLADC = {'Tox21_MMP': '0.696 ± 0.042', 'Tox21_HSE': '0.618 ± 0.110', 'Tox21_p53': '0.649 ± 0.216', 'NCI1': '0.683 ± 0.011'}
# GLocalKD paper (Ma et al., WSDM 2022), Table 1: best of InfoGraph/WL/PK x iForest/LESINN and OCGCN; GLocalKD
PAPER_BEST_BASE = {'BZR': '0.775 ± 0.063', 'COX2': '0.671 ± 0.039', 'DHFR': '0.625 ± 0.028', 'AIDS': '0.955 ± 0.023',
                   'NCI1': '0.743 ± 0.015', 'Tox21_HSE': '0.657 ± 0.051', 'Tox21_MMP': '0.571 ± 0.037', 'Tox21_p53': '0.520 ± 0.025'}
PAPER_GLOCALKD = {'BZR': '0.679 ± 0.065', 'COX2': '0.589 ± 0.045', 'DHFR': '0.558 ± 0.030', 'AIDS': '0.992 ± 0.004',
                  'NCI1': '0.683 ± 0.015', 'Tox21_HSE': '0.591 ± 0.001', 'Tox21_MMP': '0.676 ± 0.001', 'Tox21_p53': '0.639 ± 0.002'}


def ms(v):
    return f'{np.mean(v):.3f} ± {np.std(v):.3f}' if len(v) else '—'


def gladgan(dirs, score):
    out, size = {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '*', '*', 'bin', 'aucs.json')):
            ds = f.split(os.sep)[-4]
            j = json.load(open(f))
            k = [x for x in j if x.startswith('test')]
            if not k:
                continue
            out.setdefault(ds, []).append(j[k[0]][score])
            s = j[k[0]]['component/size']
            size.setdefault(ds, []).append(max(s, 1 - s))
    return out, size


def unified(d):
    out = {}
    for f in glob.glob(os.path.join(d, '*.json')):
        r = json.load(open(f))
        out.setdefault((r['method'], r['DS']), []).append(r['test'])
    return out


def released(d):
    out = {}
    path = os.path.join(d, 'released.json')
    if not os.path.exists(path):
        return out
    for k, r in json.load(open(path)).items():
        method, ds, _ = k.split(':')
        if 'folds' in r:
            out[(method, ds)] = r['folds']
        elif 'final' in r:
            out.setdefault((method, ds), []).append(r['final'])
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gladgan', nargs='+', required=True)
    p.add_argument('--unified', required=True)
    p.add_argument('--released', required=True)
    p.add_argument('--score', default='score/recon+critic|size')
    a = p.parse_args()
    gg, size = gladgan(a.gladgan, a.score)
    un, rel = unified(a.unified), released(a.released)
    cols = ['Dataset', 'Features', 'Size only*',
            'Best simple (blog)', 'GLADC (blog)', 'Best baseline (GLocalKD paper)', 'GLocalKD (paper)',
            'GLocalKD (repro, as released)', 'GLADC (repro, as released)',
            'GLocalKD (unified)', 'GLADC (unified)', 'gladGAN (unified)']
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for ds in DATASETS:
        feat = 'atom types' if ds in ('BZR', 'COX2', 'DHFR', 'AIDS') else 'degree'
        bs = BLOG_BEST_SIMPLE.get(ds)
        row = [SHORT.get(ds, ds), feat, ms(size.get(ds, [])).split(' ±')[0] if ds in size else '—',
               f'{bs[0]} ({bs[1]})' if bs else '—', BLOG_GLADC.get(ds, '—'),
               PAPER_BEST_BASE.get(ds, '—'), PAPER_GLOCALKD.get(ds, '—'),
               ms(rel.get(('glocalkd', ds), [])), ms(rel.get(('gladc', ds), [])),
               ms(un.get(('glocalkd', ds), [])), ms(un.get(('gladc', ds), [])), ms(gg.get(ds, []))]
        print('| ' + ' | '.join(row) + ' |')
