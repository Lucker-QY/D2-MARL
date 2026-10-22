# CityFlow baselines (LLMTSCS)

The CityFlow baselines of the paper (Fixed-time, MaxPressure, Random, PressLight,
AttendLight, MPLight, CoLight, Eff-CoLight, Adv-CoLight and LLMLight-13B) were run
with the open-source [LLMTSCS](https://github.com/usail-hkust/LLMTSCS) implementation
(the code of LLMLight), on which this repository also builds. We do not redistribute
LLMTSCS; `llmtscs_d2marl.patch` contains our changes to it.

## Changes in the patch

| Change | Files | Why |
|---|---|---|
| Action interval 15 s instead of 30 s (`--duration 15`, `MIN_ACTION_TIME = MEASURE_TIME = 15`) | `run_*.py`, `utils/config.py` | same action interval as D2-MARL (Table I) |
| `--seed` option that seeds `random`, NumPy, TensorFlow and PyTorch | `run_*.py` | three seeds (10, 100, 1000) per method |
| Dataset `newyork_16x3` (New York 16x3) | `run_*.py` | scenario of the paper |
| Gini coefficient, per-vehicle waiting times (`all_vehicles_wait_time.csv`), waiting events above 90 s (`delay_events_over_90s.csv`) | `utils/cityflow_env.py`, `utils/model_test.py`, `utils/oneline.py`, `utils/llm_aft_trainer.py` | the tail metrics of the paper (M_wt, C_d, G_n), with the same definitions as `utils/cityflow_env.py` of this repository |
| Weights & Biases logging disabled | `utils/pipeline.py`, `utils/llm_aft_trainer.py` | runs without an account |
| Sample list initialisation | `utils/construct_sample.py` | `[[] for _ in ...]` instead of `[None] * n` |

## Setup

```bash
git clone https://github.com/usail-hkust/LLMTSCS.git
cd LLMTSCS
git checkout d5d4180f34edb843e1d1b462d5846c75d6d4533a
git apply /path/to/D2MARL/baselines/llmtscs/llmtscs_d2marl.patch
mkdir -p data/NewYork/16_3
cp /path/to/D2MARL/data/newyork_16_3/16_3/* data/NewYork/16_3/
```

The Hangzhou and Jinan files shipped with LLMTSCS are identical to those in `data/` of
this repository. Install the requirements listed in the LLMTSCS README (CityFlow needs Linux).

## Commands

Every baseline is run with seeds 10, 100 and 1000; for example, Adv-CoLight on Hangzhou-1:

```bash
python run_advanced_colight.py --dataset hangzhou --traffic_file anon_4_4_hangzhou_real.json --seed 10
```

| Baseline in the paper | Script |
|---|---|
| Fixed-time | `run_fixedtime.py` |
| MaxPressure | `run_maxpressure.py` |
| Random | `run_random.py` |
| PressLight | `run_presslight.py` |
| AttendLight | `run_attendlight.py` |
| MPLight | `run_mplight.py` |
| CoLight | `run_colight.py` |
| Eff-CoLight | `run_efficient_colight.py` |
| Adv-CoLight | `run_advanced_colight.py` |
| LLMLight-13B | `run_open_LLM_with_vllm.py` with the LightGPT-13B weights from the [LLMLight collection on Hugging Face](https://huggingface.co/collections/usail-hkust/llmlight-lightgpt-673ac5a619cbbe309165b56d) |

`--dataset` is one of `hangzhou` (traffic files `anon_4_4_hangzhou_real.json`,
`anon_4_4_hangzhou_real_5816.json`), `jinan` (`anon_3_4_jinan_real.json`,
`anon_3_4_jinan_real_2000.json`, `anon_3_4_jinan_real_2500.json`) and `newyork_16x3`
(`anon_16_3_newyork_real.json`). The tables use the round-99 test episode of each run.

ASC (actuated control) is not part of LLMTSCS; on CityFlow it is run with this
repository: `python run_baseline.py -mod Actuated -hangzhou -traffic_file anon_4_4_hangzhou_real.json`.
