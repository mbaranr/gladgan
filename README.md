# gladGAN

[f-AnoGAN](https://doi.org/10.1016/j.media.2019.01.010) adapted from images to graph-level anomaly detection.
It is trained on normal molecules only and flags the ones it can't reconstruct.

## Results

Test AUC, mean ± std over 5 stratified folds. The majority class is treated as normal, and every method gets atom-type features.

| Dataset | Size only | WL-kNN | [GLADC](https://github.com/XuexiongLuoMQ/GLADC) | [GLocalKD](https://github.com/RongrongMa/GLocalKD) | gladGAN |
|---|---|---|---|---|---|
| BZR | 0.574 ± 0.077 | 0.779 ± 0.064 | 0.406 ± 0.147 | 0.601 ± 0.153 | **0.778 ± 0.060** |
| COX2 | 0.551 ± 0.096 | 0.689 ± 0.093 | 0.522 ± 0.062 | 0.559 ± 0.111 | **0.714 ± 0.048** |
| DHFR | 0.452 ± 0.035 | 0.588 ± 0.050 | 0.502 ± 0.043 | 0.415 ± 0.048 | **0.659 ± 0.037** |
| AIDS | 0.990 ± 0.007 | 0.298 ± 0.040 | 0.976 ± 0.003 | 0.365 ± 0.042 | **0.995 ± 0.007** |
| MUTAG | 0.848 ± 0.094 | 0.893 ± 0.021 | 0.345 ± 0.065 | **0.896 ± 0.045** | 0.807 ± 0.070 |
| NCI109 | 0.549 ± 0.032 | **0.815 ± 0.019** | 0.414 ± 0.012 | 0.640 ± 0.023 | 0.624 ± 0.024 |

## Usage

The datasets come from the [TUDataset collection](https://chrsmrrs.github.io/datasets/docs/datasets/)
(BZR, COX2, DHFR, AIDS, MUTAG, NCI1, NCI109, Tox21). Download them into `datasets/` by hand or with the script:

```bash
conda env create -f environment.yml && conda activate gladgan
bash exp/fetch_molecular.sh BZR          # or no arguments for every dataset used
mkdir -p runs/BZR && cd runs/BZR
python ../../main.py --DS BZR --cv_fold 0
```

Results are written to `bin/aucs.json`.

GLADC and GLocalKD run from their official code, [XuexiongLuoMQ/GLADC](https://github.com/XuexiongLuoMQ/GLADC)
(`3fe8921`) and [RongrongMa/GLocalKD](https://github.com/RongrongMa/GLocalKD) (`1c8c15f`). `baselines/setup.sh`
clones both at those commits, and `baselines/reproduce_unified.py` runs them under the protocol above.
`baselines/wl_knn.py` computes the WL-kNN and size-only columns.

## License

MIT
