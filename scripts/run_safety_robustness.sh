#!/usr/bin/env bash
# Safety, observation noise and inference cost on SUMO, from the round-99 weights of
# scripts/run_d2marl.sh sumo and scripts/run_baselines_sumo.sh.
#
#   bash scripts/run_safety_robustness.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

# 1) Safety: re-evaluate with the SSM device and lane queue recording, then compute
#    the surrogate safety measures (d2marl_sumo/ssm_metrics.py).
for tf in "${GRID_FLOWS[@]}"; do
  net=${NET[$tf]}
  for seed in "${SEEDS[@]}"; do
    for m in D2MARL AdvancedColight EfficientColight CoLight Attend MPLight PressLight; do
      if [ "$m" = D2MARL ]; then memo="d2marl-sumo-full-${TAG[$tf]}-seed${seed}"
      else memo="base-sumo-${m}-${TAG[$tf]}-seed${seed}"; fi
      out="ssm-${m}-${TAG[$tf]}-seed${seed}"
      python -u run_eval.py -mod "$m" -memo "$memo" -round 99 $net -traffic_file "$tf" \
          -seed "$seed" -sumo -out_memo "$out"
      dst="safety_raw/${m}/${TAG[$tf]}/seed${seed}"; mkdir -p "$dst"
      cp records/"$out"/*/test_round/round_99/{ssm.xml,lane_queue.csv,collisions.xml} "$dst"/ 2>/dev/null || true
    done
    for m in Fixedtime MaxPressure Actuated Random; do
      out="ssm-${m}-${TAG[$tf]}-seed${seed}"
      python -u run_eval.py -mod "$m" $net -traffic_file "$tf" -seed "$seed" -sumo -out_memo "$out"
      dst="safety_raw/${m}/${TAG[$tf]}/seed${seed}"; mkdir -p "$dst"
      cp records/"$out"/*/test_round/round_0/{ssm.xml,lane_queue.csv,collisions.xml} "$dst"/ 2>/dev/null || true
    done
  done
done
python -m d2marl_sumo.ssm_metrics --raw_dir safety_raw --out safety_out
python analysis/plot_safety.py --data safety_out --out figures

# 2) Observation noise (six grid scenarios): clean, Gaussian sigma 0.1/0.2/0.4, dropout 0.1/0.2
for tf in "${GRID_FLOWS[@]}"; do
  net=${NET[$tf]}
  for seed in "${SEEDS[@]}"; do
    for lvl in "clean 0 0" "s010 0.1 0" "s020 0.2 0" "s040 0.4 0" "d010 0 0.1" "d020 0 0.2"; do
      set -- $lvl
      python -u -m d2marl_sumo.robustness -mod D2MARL -memo "d2marl-sumo-full-${TAG[$tf]}-seed${seed}" \
          -round 99 $net -traffic_file "$tf" -seed "$seed" -sigma "$2" -dropout "$3" -out_tag "$1"
    done
  done
done

# 3) Inference cost: whole-network decision time on one CPU node, 3600 simulated seconds,
#    the first 20 decisions discarded (median reported), seed-10 weights of every learned method
mkdir -p cost_json
for tf in "${GRID_FLOWS[@]}"; do
  net=${NET[$tf]}
  python -u -m d2marl_sumo.inference_cost -mod D2MARL -memo "d2marl-sumo-full-${TAG[$tf]}-seed10" \
      -round 99 $net -traffic_file "$tf" -steps 3600 -warmup 20 -out "cost_json/${TAG[$tf]}_D2MARL.json"
  for m in AdvancedColight EfficientColight CoLight Attend MPLight PressLight; do
    python -u -m d2marl_sumo.inference_cost -mod "$m" -memo "base-sumo-${m}-${TAG[$tf]}-seed10" \
        -round 99 $net -traffic_file "$tf" -steps 3600 -warmup 20 -out "cost_json/${TAG[$tf]}_${m}.json"
  done
done
# CoordLight and LLMLight: d2marl_sumo/inference_cost_torch.py (see baselines/coordlight/README.md)
python -m d2marl_sumo.cost_summary cost_json
