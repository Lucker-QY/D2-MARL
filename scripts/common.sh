#!/usr/bin/env bash
# Shared definitions for the run scripts. Source it, do not execute it.
# Scenario flags follow Table I (configs/table1.json).

SEEDS=(10 100 1000)

# traffic file -> network flag and scenario flags
declare -A NET=(
  [anon_4_4_hangzhou_real.json]="-hangzhou"
  [anon_4_4_hangzhou_real_5816.json]="-hangzhou"
  [anon_3_4_jinan_real.json]="-jinan"
  [anon_3_4_jinan_real_2000.json]="-jinan"
  [anon_3_4_jinan_real_2500.json]="-jinan"
  [anon_16_3_newyork_real.json]="-newyork163"
)
declare -A TAG=(
  [anon_4_4_hangzhou_real.json]="hz1"
  [anon_4_4_hangzhou_real_5816.json]="hz2"
  [anon_3_4_jinan_real.json]="jn1"
  [anon_3_4_jinan_real_2000.json]="jn2"
  [anon_3_4_jinan_real_2500.json]="jn3"
  [anon_16_3_newyork_real.json]="ny163"
)
GRID_FLOWS=(anon_4_4_hangzhou_real.json anon_4_4_hangzhou_real_5816.json
            anon_3_4_jinan_real.json anon_3_4_jinan_real_2000.json anon_3_4_jinan_real_2500.json
            anon_16_3_newyork_real.json)

scenario_flags() {   # $1 = network flag
  case "$1" in
    -hangzhou)   echo "-weight_max_wait -0.1" ;;
    *)           echo "" ;;
  esac
}

# The paper runs every traffic file and seed as its own single-process job on 4 CPU cores.
export CUDA_VISIBLE_DEVICES=-1
