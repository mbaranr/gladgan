"""
Run an official baseline script exactly as released (models, losses, splits, label handling, metric),
only with the compat shim applied.

    python baselines/run_released.py glocalkd main.py --DS BZR --feature default --datadir datasets
    python baselines/run_released.py glocalkd main_Tox.py --DS Tox21_MMP --datadir datasets
    python baselines/run_released.py gladc main.py --DS BZR --datadir datasets
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compat  # noqa: E402,F401

method, script, *rest = sys.argv[1:]
repo = os.path.join(HERE, method)
rest = [os.path.abspath(a) if prev == '--datadir' else a for prev, a in zip([''] + rest[:-1], rest)]
os.chdir(repo)
sys.path.insert(0, repo)
sys.argv = [script] + rest
print(f"[run_released] {method}/{script} {' '.join(rest)} on {compat.DEVICE}", flush=True)
runpy.run_path(script, run_name='__main__')
