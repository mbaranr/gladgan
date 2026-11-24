#!/bin/bash
# Download the datasets (TUDataset collection, https://chrsmrrs.github.io/datasets/) into datasets/.
# Pass dataset names to fetch only those, e.g.: bash exp/fetch_molecular.sh BZR COX2
set -e
mkdir -p "$(dirname "$0")/../datasets" && cd "$(dirname "$0")/../datasets"
ALL="BZR COX2 DHFR AIDS NCI1 Tox21_HSE Tox21_MMP Tox21_p53 MUTAG PTC_MR PTC_FM PTC_FR PTC_MM NCI109 Mutagenicity MCF-7 PC-3 \
         Tox21_AhR Tox21_ARE Tox21_ATAD5 Tox21_ER Tox21_ER-LBD Tox21_PPAR-gamma Tox21_aromatase"
for n in ${@:-$ALL}; do
  case $n in Tox21_*) parts="${n}_training ${n}_testing";; *) parts=$n;; esac
  for p in $parts; do
    [ -d "$p" ] && continue
    curl -sSL -o "$p.zip" "https://www.chrsmrrs.com/graphkerneldatasets/$p.zip"
    unzip -q "$p.zip" && rm "$p.zip"
  done
done
