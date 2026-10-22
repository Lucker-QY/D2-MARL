# D²-MARL: Dual-Phase Multi-Agent Reinforcement Learning for Extreme Delay Suppression

Code for the paper *D²-MARL: Dual-Phase Multi-Agent Reinforcement Learning for Extreme
Delay Suppression* (IEEE Transactions on Intelligent Transportation Systems).

D²-MARL is a multi-agent traffic signal controller that targets the upper tail of the
waiting-time distribution. A dual-phase attention design pairs an intra-phase branch over
phase and delay descriptors with a spatial graph-attention branch over neighbouring
intersections and fuses them with a learned gate; a bounded delay penalty in the reward
and a stratified replay of severe-delay samples act on the longest waits.

The code runs on **CityFlow** and on **SUMO** with the same agent, and contains the SUMO
versions of all baselines, the CityFlow-to-SUMO converter, and the evaluation of the tail,
safety, robustness and inference-cost results of the paper.

## Repository layout

```
run_d2marl.py            train D2-MARL (CityFlow by default, -sumo for SUMO)
run_baseline.py          train/run a baseline in the same environment
run_eval.py              evaluate stored weights of a given round in a fresh episode
models/                  D2-MARL agent (d2marl_agent.py) and baseline agents
utils/                   training pipeline and CityFlow environment (based on LLMTSCS)
d2marl_sumo/             SUMO environment, CityFlow->SUMO conversion, metrics,
                         safety measures, robustness and inference-cost tools
coordlight_sumo/         TraCI back end and evaluation wrapper for CoordLight on SUMO
baselines/               patches for the external baselines (LLMTSCS, CoordLight)
analysis/                figure scripts (analysis/figures/), safety figure, divergence rule
configs/                 Table I settings, ablation flags, configs of the paper runs
scripts/                 run scripts for the experiments of the paper
data/                    CityFlow networks and demands (Hangzhou, Jinan, New York)
sumo_scenarios/          SUMO versions of the six grid scenarios, Ingolstadt21 and Cologne8
tests/                   configuration tests (no TensorFlow needed)
```

## Installation

The experiments were run on Linux with Python 3.6.15 and TensorFlow 2.4.0 on CPU
(4 cores per run).

```bash
conda env create -f environment.yml
conda activate d2marl
```

**CityFlow** (version 0.1) is built from source, see the
[CityFlow documentation](https://cityflow.readthedocs.io/en/latest/install.html):

```bash
git clone https://github.com/cityflow-project/CityFlow.git
pip install ./CityFlow
```

**SUMO 1.20.0** is needed for the SUMO experiments. Install it from
[eclipse.dev/sumo](https://eclipse.dev/sumo) and set `SUMO_HOME`; `traci` and `sumolib`
are imported from `$SUMO_HOME/tools`:

```bash
export SUMO_HOME=/path/to/sumo-1.20.0
export PATH=$SUMO_HOME/bin:$PATH
```

Optional, for single baselines only: LLMLight-13B needs PyTorch, `transformers` and
`vllm` with a GPU (a separate environment is recommended; the agent registry in
`utils/config.py` loads it without TensorFlow); CoordLight needs the requirements of its
own repository (see `baselines/coordlight/README.md`).

Check the installation without a simulator:

```bash
python -m unittest discover tests
```

## Data

| Scenario | Tag | CityFlow files (`data/`) | SUMO scenario (`sumo_scenarios/`) |
|---|---|---|---|
| Hangzhou-1 | hz1 | `Hangzhou/4_4/anon_4_4_hangzhou_real.json` | `anon_4_4_hangzhou_real/` |
| Hangzhou-2 | hz2 | `Hangzhou/4_4/anon_4_4_hangzhou_real_5816.json` | `anon_4_4_hangzhou_real_5816/` |
| Jinan-1 | jn1 | `Jinan/3_4/anon_3_4_jinan_real.json` | `anon_3_4_jinan_real/` |
| Jinan-2 | jn2 | `Jinan/3_4/anon_3_4_jinan_real_2000.json` | `anon_3_4_jinan_real_2000/` |
| Jinan-3 | jn3 | `Jinan/3_4/anon_3_4_jinan_real_2500.json` | `anon_3_4_jinan_real_2500/` |
| New York | ny163 | `newyork_16_3/16_3/anon_16_3_newyork_real.json` | `anon_16_3_newyork_real/` |
| Ingolstadt21 (non-grid) | ing21 | – | `ingolstadt21/` |
| Cologne8 (non-grid) | col8 | – | `cologne8/` |

The grid datasets come from [traffic-signal-control.github.io](https://traffic-signal-control.github.io).
Their SUMO versions were generated with the converter, for example:

```bash
python -m d2marl_sumo.cityflow2sumo --roadnet data/Hangzhou/4_4/roadnet_4_4.json \
    --flow data/Hangzhou/4_4/anon_4_4_hangzhou_real.json \
    --outdir sumo_scenarios/anon_4_4_hangzhou_real --name anon_4_4_hangzhou_real
python -m d2marl_sumo.validate_conversion \
    --report sumo_scenarios/anon_4_4_hangzhou_real/anon_4_4_hangzhou_real.convert_report.json \
    --cityflow-roadnet data/Hangzhou/4_4/roadnet_4_4.json \
    --cityflow-flow data/Hangzhou/4_4/anon_4_4_hangzhou_real.json
```

Ingolstadt21 and Cologne8 are taken from the [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO)
(Ingolstadt21: GPL-3.0; Cologne8: CC BY-NC-SA 3.0; see the `LICENSE` file in each directory).

## Quick start

A one-round smoke test on CityFlow and on SUMO:

```bash
python run_d2marl.py -hangzhou -traffic_file anon_4_4_hangzhou_real.json -seed 10 \
    -weight_max_wait -0.1 -num_rounds 1 -single_process -memo smoke-cf
python run_d2marl.py -hangzhou -traffic_file anon_4_4_hangzhou_real.json -seed 10 \
    -weight_max_wait -0.1 -num_rounds 1 -single_process -memo smoke-sumo -sumo
```

Outputs go to `records/<memo>/<traffic_file>_<time stamp>/` (test episodes in
`test_round/round_<N>/`) and the weights to `model/<memo>/...`.

## Settings (Table I)

The shared training hyperparameters are the defaults of the LLMTSCS framework
(`utils/config.py`), and the method constants (waiting-time limits W_s / W_h = 60 / 90 s,
over-limit threshold τ = 60 s, transition width Δ = 30 s, penalty cap 2000, replay
threshold τ_p = −50, maximum severe-sample share ρ = 0.05, ω_q = 0.5, ω_o = 0.25) are the
defaults of `run_d2marl.py` and the same in all scenarios. Only the delay weight ω_w
depends on the scenario; the non-grid networks add the adaptation of Section IV-C:

| Scenario | ω_w | Flags for `run_d2marl.py` |
|---|---|---|
| Hangzhou | 0.10 | `-hangzhou -weight_max_wait -0.1` |
| Jinan | 0.05 | `-jinan` |
| New York | 0.05 | `-newyork163` |
| Ingolstadt21 (SUMO) | 0.05 | `-ingolstadt21 -sumo -spillback` |
| Cologne8 (SUMO) | 0.05 | `-cologne8 -sumo -spillback` |

`configs/table1.json` lists all values. `tests/test_config_equivalence.py` checks that
these flags reproduce the configuration of the 18 CityFlow runs of the paper
(`configs/paper_cityflow_full_model.json`).

Seeds are 10, 100 and 1000. Every traffic file and seed was run as its own
single-process job (`-traffic_file ... -single_process`) on four CPU cores.

## Reproducing the paper

### Main comparison (CityFlow) and SUMO replication

Train D²-MARL and re-evaluate the round-99 weights, e.g. Jinan-2 with seed 100:

```bash
python run_d2marl.py -jinan \
    -traffic_file anon_3_4_jinan_real_2000.json -seed 100 -memo d2marl-cf-jn2-seed100 -single_process
python run_eval.py -mod D2MARL -memo d2marl-cf-jn2-seed100 -round 99 -jinan \
    -traffic_file anon_3_4_jinan_real_2000.json -seed 100 -out_memo eval-d2marl-cf-jn2-seed100
```

Add `-sumo` to both commands for SUMO (and `-no_ssm` to `run_eval.py` when no safety
measures are needed). `scripts/run_d2marl.sh cityflow` and `scripts/run_d2marl.sh sumo`
run all six scenarios and three seeds and summarise the results.

Baselines:

* CityFlow: the learned baselines and Fixed-time, MaxPressure and Random were run with
  LLMTSCS, see `baselines/llmtscs/README.md`; ASC with
  `python run_baseline.py -mod Actuated -hangzhou -traffic_file anon_4_4_hangzhou_real.json`.
* SUMO: all baselines run in the same environment as D²-MARL through `run_baseline.py`
  (`-mod` Fixedtime, MaxPressure, Actuated, Random, PressLight, Attend, MPLight, CoLight,
  EfficientColight, AdvancedColight, LLMLight) and are re-evaluated with `run_eval.py`;
  see `scripts/run_baselines_sumo.sh`.
* CoordLight on both simulators: `baselines/coordlight/README.md`.

### Ablations

Each variant (named as in the ablation table of the paper) adds flags to the scenario
flags (`configs/ablations.json`):

| Variant | Flags |
|---|---|
| full | – |
| w/o NA (all mechanisms removed) | `-no_extended_state -no_max_wait_time -no_wait_penalty -no_intra_phase_attention -no_stratified_replay` |
| w/o ES (extended state and over-limit term) | `-no_extended_state` |
| w/o WP (delay penalty) | `-no_wait_penalty` |
| w/o MW (delay-burden feature) | `-no_max_wait_time` |
| w/o IA (intra-phase attention) | `-no_intra_phase_attention` |
| w/o RP (stratified replay) | `-no_stratified_replay` |

```bash
bash scripts/run_d2marl.sh cityflow RP
```

### Non-grid networks

```bash
python run_d2marl.py -ingolstadt21 -sumo -spillback -seed 10 -memo ng-d2marl-ing21-seed10 -single_process
python run_d2marl.py -cologne8 -sumo -spillback -seed 10 -memo ng-d2marl-col8-seed10 -single_process
```

`-spillback` is the adaptation to networks with short links (Section IV-C): the
free-storage ratio of the downstream link enters the spatial branch, the gate and the
intra-phase branch (whose 44 inputs form four tokens of width 11), and the pressure is
discounted by min(1, c / 0.2). Non-grid lanes are aggregated into 12 movement slots and
unavailable phases are masked in action selection and in the TD target. See
`scripts/run_nongrid.sh` for the baselines. Teleporting is disabled (`--time-to-teleport -1`),
as in RESCO, so gridlocked vehicles stay in the network.

### Safety, observation noise and inference cost (SUMO)

`scripts/run_safety_robustness.sh` runs all three parts:

* **Safety**: `run_eval.py -sumo` with the SSM device writes `ssm.xml`, `lane_queue.csv`
  and `collisions.xml`; `python -m d2marl_sumo.ssm_metrics --raw_dir safety_raw --out safety_out`
  computes the measures (definitions in the module docstring) and
  `python analysis/plot_safety.py --data safety_out --out figures` draws the figure.
* **Observation noise**: `python -m d2marl_sumo.robustness` on the six grid scenarios; each
  lane feature x becomes x(1 + ε) with ε ~ N(0, σ²), σ ∈ {0.1, 0.2, 0.4}, or is set to zero
  with probability p ∈ {0.1, 0.2} (dropout); phase and adjacency inputs are not perturbed.
* **Inference cost**: `python -m d2marl_sumo.inference_cost` (TensorFlow agents) and
  `python -m d2marl_sumo.inference_cost_torch` (CoordLight, LLMLight), summarised by
  `python -m d2marl_sumo.cost_summary`. The paper times the actions for the whole network
  at one decision step on one CPU node (3600 s, first 20 decisions discarded, median),
  in the six grid scenarios.

### Replacement of divergent runs

`python analysis/divergence_rule.py per_seed.csv --keys method flow` flags runs that are
Dixon Q outliers (Q > 0.970) and more than 50 % worse than the median of the other two
seeds in A_TT or M_wt. Replacement seeds are tried in the order 20, 200, 2000, 20000,
200000, and all runs are reported (supplementary material, Table S15).

### Figures and tables

Result files are not part of the repository. The tables are produced from the run
directories by the summary tools, and the figures by the scripts in `analysis/`, which
read those summaries (input formats in the docstring of each script):

| Paper | Produced by |
|---|---|
| Tables II, III, IV | `python -m d2marl_sumo.collect_results` (CityFlow baselines: LLMTSCS summaries, see `baselines/llmtscs/README.md`) |
| Table V | `python -m d2marl_sumo.ssm_metrics` |
| Fig. 4(b) | `python analysis/figures/fig_topk_neighbourhood.py --out figures` |
| Fig. 5 | `python analysis/figures/fig_rank_radar.py --table main_table.csv --out figures/rank_radar.png` |
| Fig. 6 | `python analysis/figures/fig_wait_density.py --vehicles vehicles --out figures/density.png` |
| Fig. 7 | `python analysis/figures/fig_gate_alpha.py --roadnet ... --log 100=... --log 1000=... --out figures/gate.pdf` |
| Fig. 8 | `python analysis/figures/fig_noise.py --csv noise_per_seed.csv --out figures` |
| Fig. 9 | `python analysis/plot_safety.py --data safety_out --out figures` |
| Supplement: loss, attention | `analysis/figures/fig_loss.py`, `analysis/figures/fig_attention.py` |
| Supplement: inference cost | `python -m d2marl_sumo.cost_summary cost_json` |
| Supplement: replaced runs | `python analysis/divergence_rule.py` |

The test episode of a round writes `attention_logs/attn_log_ep<round>.json` for D²-MARL;
it contains the gate output, the delay burden and the attention weights of every
intersection at every step and is the input of Fig. 7 and of the attention figure.

## Metrics and evaluation windows

`python -m d2marl_sumo.collect_results` summarises all runs under `records/`
(mean ± standard deviation over seeds):

| Metric | Definition |
|---|---|
| ATT | average travel time (unfinished trips counted up to the end of the episode) |
| Awt | average waiting time of the waiting vehicles, averaged over decision steps |
| AQL | average total queue length |
| M_wt | longest single continuous wait of any vehicle (`Mwt_exact`; `Mwt` is the same value derived from the >90 s events and is censored as `<90` when no wait exceeds 90 s) |
| C_d | number of waiting episodes longer than 90 s |
| G_n | Gini coefficient of the per-vehicle cumulative waiting time |

The paper never mixes evaluation windows within a table:

| Results | Window | `collect_results` options |
|---|---|---|
| D²-MARL on CityFlow; all learned methods on SUMO | fresh episode with the round-99 weights (`run_eval.py -round 99`) | `--include eval-... --last-n 1` |
| CityFlow baselines (LLMTSCS) | round-99 test episode of the training run | – |
| Non-grid networks | mean of the test episodes of rounds 79–98 | `--at-round 99 --last-n 20` |
| Rule-based controllers | one episode | – |

Evaluation of stored weights is deterministic. Training with the same seed on different
hardware or library builds can follow a different trajectory, which is one reason the
paper reports three seeds.

## Trained weights and result files

Following the practice of the baseline repositories (LLMTSCS, CoordLight), this
repository contains code, configurations and scenario data only; trained weights and
raw result files are not included.

## License

The code written for this repository is released under the MIT License (`LICENSE`).
Third-party code and data keep their own licenses, see `THIRD_PARTY_NOTICES.md`.
In particular, the training framework builds on
[LLMTSCS](https://github.com/usail-hkust/LLMTSCS), the Ingolstadt21 scenario is
GPL-3.0 and the Cologne8 scenario is CC BY-NC-SA 3.0.

## Citation

```bibtex
@article{d2marl,
  title   = {D$^2$-MARL: Dual-Phase Multi-Agent Reinforcement Learning for Extreme Delay Suppression},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  note    = {to appear}
}
```
