# CoordLight baseline

[CoordLight](https://github.com/marmotlab/CoordLight) (IEEE T-ITS 2025) is
a PyTorch + PPO + Ray implementation and is not part of this code base. We do not
redistribute it; the two patches contain our changes. CoordLight is MIT-licensed
(`LICENSE-CoordLight`).

| Patch | Used for | Changes |
|---|---|---|
| `coordlight_cityflow.patch` | CityFlow runs | 15 s green / 5 s yellow and 4 phases as in the other methods; 100 episodes, 8 workers; seed setting; waiting-time, severe-delay (> 90 s) and Gini metrics with the definitions of D2-MARL; command-line options of `Tester.py` |
| `coordlight_sumo.patch` | SUMO runs | everything above, plus: engine selection (CityFlow or the TraCI back end `coordlight_sumo/cityflow_api_over_sumo.py`) controlled by environment variables; scenario, seed, number of episodes/workers and output directory configurable through environment variables; AWT and AQL computed with the definitions of D2-MARL (`utils/model_test.py`) so that they are comparable with the other baselines; loading `state_dict.pth` when no `checkpoint.pkl` exists; robust path handling |

The observation (QDSE), reward, phase logic and network of CoordLight are unchanged in both patches.

## Setup

Clone CoordLight into `coordlight_sumo/CoordLight` (the SUMO wrapper expects it there):

```bash
git clone https://github.com/marmotlab/CoordLight.git coordlight_sumo/CoordLight
cd coordlight_sumo/CoordLight
git checkout be41b848a743cfa4a2814792cf3455dacd13bb36
git apply ../../baselines/coordlight/coordlight_cityflow.patch   # CityFlow runs
# or
git apply ../../baselines/coordlight/coordlight_sumo.patch       # SUMO runs
```

For SUMO, the New York net_config shipped with CoordLight lacks `lane_length_dict`;
add it (verified against the Hangzhou and Jinan files) from the repository root:

```bash
python -m coordlight_sumo.fix_net_config --write
```

Install the requirements of CoordLight (`requirements.txt` in its repository).

## Scenarios

| Scenario | `CITYFLOW_MAP_TYPE` | `CITYFLOW_FLOW_TYPE` |
|---|---|---|
| Hangzhou-1 | `DH` | `1` |
| Hangzhou-2 | `DH` | `3` |
| Jinan-1 / 2 / 3 | `DJ` | `1` / `2` / `3` |
| New York | `DN2` | (any) |

## CityFlow

With `coordlight_cityflow.patch`, set `CITYFLOW_MAP_TYPE`, `CITYFLOW_FLOW_TYPE`, `SEED`
(10, 100, 1000) and `EXPERIMENT_NAME` in `Parameters.py`, then train and test:

```bash
cd coordlight_sumo/CoordLight
python Driver.py
python Tester.py --test_path Train_CityFlow_CoordLight/ --experiment_name <experiment name> --seed 10 --num_test 1
```

## SUMO

With `coordlight_sumo.patch`, the settings are environment variables; for example,
Hangzhou-1 with seed 10:

```bash
cd coordlight_sumo/CoordLight
CL_SIMULATOR=sumo \
CL_SUMO_NET=../../sumo_scenarios/anon_4_4_hangzhou_real/anon_4_4_hangzhou_real.net.xml \
CL_SUMO_ROU=../../sumo_scenarios/anon_4_4_hangzhou_real/anon_4_4_hangzhou_real.repaired.rou.xml \
CL_MAP_TYPE=DH CL_FLOW_TYPE=1 CL_SEED=10 CL_EXPERIMENT_NAME=SUMO_CoordLight_hz1_seed10 \
python Driver.py
```

Other variables: `CL_MAX_EPISODES` (default 100), `CL_NUM_META` (Ray workers, default 8),
`CL_TRAIN_ROOT` (output root, default `Train_CityFlow_CoordLight`), `CL_USE_GPU`,
`CL_HORIZON` (default 3600). Evaluate the trained model with the metric pipeline of
this repository (five evaluation episodes, exported as `records/<out_memo>/.../test_round/round_0..4`):

```bash
python -m coordlight_sumo.run_coordlight -traffic_file anon_4_4_hangzhou_real.json \
    -model_path coordlight_sumo/CoordLight/Train_CityFlow_CoordLight/SUMO_CoordLight_hz1_seed10 \
    -seed 10 -out_memo sumo-base-CoordLight-hz1-seed10
```

CoordLight's released pipeline uses eight parallel workers, i.e. eight times the
interactions of the other methods (see the paper).
