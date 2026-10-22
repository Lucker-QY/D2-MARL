#!/usr/bin/env bash
# Non-grid networks Ingolstadt21 and Cologne8 (SUMO only). D2-MARL uses the base settings
# with the adaptation to networks with short links of Section IV-C (-spillback).
# The paper reports the mean of the test episodes of rounds 79-98.
#
#   bash scripts/run_nongrid.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

for net in ingolstadt21 cologne8; do
  tag=${net/ingolstadt21/ing21}; tag=${tag/cologne8/col8}
  for seed in "${SEEDS[@]}"; do
    python -u run_d2marl.py -"$net" -sumo -spillback -seed "$seed" \
        -memo "ng-d2marl-${tag}-seed${seed}" -single_process
    for m in AdvancedColight EfficientColight CoLight Attend MPLight PressLight; do
      python -u run_baseline.py -mod "$m" -"$net" -sumo -seed "$seed" \
          -memo "ng-base-${m}-${tag}-seed${seed}" -single_process
    done
    for m in Fixedtime MaxPressure Actuated Random; do
      python -u run_baseline.py -mod "$m" -"$net" -sumo -seed "$seed" \
          -memo "ng-base-${m}-${tag}-seed${seed}" -single_process
    done
  done
done

# rounds 79-98 for the learned methods; rule-based methods have one episode
python -m d2marl_sumo.collect_results --records records --include "ng-" \
    --at-round 99 --last-n 20 --csv summary_nongrid.csv
