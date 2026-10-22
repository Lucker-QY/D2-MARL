#!/usr/bin/env bash
# SUMO baselines in the same environment as D2-MARL: train the learned baselines
# (100 rounds) and re-evaluate their round-99 weights; rule-based baselines run one episode.
#
#   bash scripts/run_baselines_sumo.sh
#
# CityFlow baselines: see baselines/llmtscs/README.md. CoordLight: baselines/coordlight/README.md.
# LLMLight-13B needs a separate environment with vLLM, see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

LEARNED=(AdvancedColight EfficientColight CoLight Attend MPLight PressLight)
RULE=(Fixedtime MaxPressure Actuated Random)

for tf in "${GRID_FLOWS[@]}"; do
  net=${NET[$tf]}
  for seed in "${SEEDS[@]}"; do
    for m in "${LEARNED[@]}"; do
      memo="base-sumo-${m}-${TAG[$tf]}-seed${seed}"
      python -u run_baseline.py -mod "$m" $net -sumo -traffic_file "$tf" -seed "$seed" \
          -memo "$memo" -single_process
      python -u run_eval.py -mod "$m" -memo "$memo" -round 99 $net -traffic_file "$tf" \
          -seed "$seed" -sumo -no_ssm -out_memo "eval-$memo"
    done
    for m in "${RULE[@]}"; do
      python -u run_eval.py -mod "$m" $net -traffic_file "$tf" -seed "$seed" -sumo -no_ssm \
          -out_memo "eval-base-sumo-${m}-${TAG[$tf]}-seed${seed}"
    done
  done
done

python -m d2marl_sumo.collect_results --records records --include "eval-base-sumo-" \
    --last-n 1 --csv summary_baselines_sumo.csv
