#!/usr/bin/env bash
# Train D2-MARL on the six grid scenarios with three seeds and re-evaluate the
# round-99 weights (the protocol of the main tables).
#
#   bash scripts/run_d2marl.sh cityflow [variant]
#   bash scripts/run_d2marl.sh sumo     [variant]
#
# variant: one of configs/ablations.json (default: full). Runs are sequential;
# on a cluster submit one job per (traffic file, seed) instead.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

SIM=${1:?usage: run_d2marl.sh cityflow|sumo [variant]}
VARIANT=${2:-full}
SIMFLAG=""; [ "$SIM" = "sumo" ] && SIMFLAG="-sumo"
VFLAGS=$(python -c "import json,sys; print(' '.join(json.load(open('configs/ablations.json'))['variants'][sys.argv[1]]))" "$VARIANT")

for tf in "${GRID_FLOWS[@]}"; do
  net=${NET[$tf]}
  for seed in "${SEEDS[@]}"; do
    memo="d2marl-${SIM}-${VARIANT}-${TAG[$tf]}-seed${seed}"
    python -u run_d2marl.py $net $(scenario_flags "$net") $SIMFLAG $VFLAGS \
        -traffic_file "$tf" -seed "$seed" -memo "$memo" -single_process
    python -u run_eval.py -mod D2MARL -memo "$memo" -round 99 $net -traffic_file "$tf" \
        -seed "$seed" $SIMFLAG -no_ssm -out_memo "eval-$memo"
  done
done

# Summary of the round-99 re-evaluations (mean +- std over seeds)
python -m d2marl_sumo.collect_results --records records --include "eval-d2marl-${SIM}-${VARIANT}-" \
    --last-n 1 --csv "summary_d2marl_${SIM}_${VARIANT}.csv"
