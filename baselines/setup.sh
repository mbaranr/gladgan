#!/bin/bash
# Fetch the official baseline implementations at the commits used for the reproduction.
# They are not vendored here: GLocalKD ships without a licence.
set -e
cd "$(dirname "$0")"
[ -d gladc ]    || git clone https://github.com/XuexiongLuoMQ/GLADC.git gladc
[ -d glocalkd ] || git clone https://github.com/RongrongMa/GLocalKD.git glocalkd
git -C gladc checkout -q 3fe8921
git -C glocalkd checkout -q 1c8c15f
echo "baselines ready"
