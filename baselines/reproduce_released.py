"""
Reproduce GLocalKD and GLADC with their official scripts and defaults (as released), parse the AUCs.

    python baselines/reproduce_released.py --out runs/released
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'datasets')

JOBS = (
    # (method, script, dataset, extra args, seeds)
    [('glocalkd', 'main.py', ds, ['--feature', 'default'], [1]) for ds in ['BZR', 'COX2', 'DHFR', 'AIDS']] +
    [('glocalkd', 'main.py', 'NCI1', ['--feature', 'deg-num'], [1])] +
    [('glocalkd', 'main_Tox.py', ds, [], [1, 2, 3, 4, 5]) for ds in ['Tox21_HSE', 'Tox21_MMP', 'Tox21_p53']] +
    [('gladc', 'main.py', ds, [], [1]) for ds in ['BZR', 'COX2', 'DHFR', 'AIDS', 'NCI1']]
)


def parse(method, script, text):
    if script == 'main.py':   # 5-fold CV: final line "auroc[...], average: m, std: s"
        m = re.search(r'auroc\[([^\]]*)\], average: ([\d.]+), std: ([\d.]+)', text)
        return {'folds': [float(x) for x in m.group(1).split()], 'mean': float(m.group(2)), 'std': float(m.group(3))}
    vals = re.findall(r'auroc_ab: ([\d.]+)', text)   # main_Tox: AUC every 10 epochs; reported value = last
    return {'final': float(vals[-1])}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--only', nargs='*', default=None, help='subset of "method:dataset"')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, 'released.json')
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for method, script, ds, extra, seeds in JOBS:
        key = f'{method}:{ds}'
        if a.only and key not in a.only:
            continue
        for seed in seeds:
            k = f'{key}:seed{seed}'
            if k in res:
                continue
            log = os.path.join(a.out, f'{method}_{ds}_seed{seed}.log')
            cmd = [sys.executable, os.path.join(HERE, 'run_released.py'), method, script, '--DS', ds,
                   '--datadir', DATA, '--seed', str(seed)] + extra
            with open(log, 'w') as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            text = open(log).read()
            try:
                res[k] = parse(method, script, text)
            except Exception as e:
                res[k] = {'error': f'{type(e).__name__} (rc={r.returncode}), see {log}'}
            json.dump(res, open(res_path, 'w'), indent=1)
            print(k, res[k], flush=True)
