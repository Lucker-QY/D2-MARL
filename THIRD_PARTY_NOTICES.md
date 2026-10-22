# Third-party code and data

| Component | Where | Origin | License |
|---|---|---|---|
| Training framework (`utils/`), baseline agents (`models/` except `d2marl_agent.py`, `actuated_agent.py`, `phase_util.py`), LLMLight prompt and parsing (`models/llmlight_agent.py`) | this repository, modified | [LLMTSCS](https://github.com/usail-hkust/LLMTSCS) (code of LLMLight) | MIT, as stated by the LLMTSCS authors |
| CityFlow baselines of the paper | `baselines/llmtscs/llmtscs_d2marl.patch` (patch only) | [LLMTSCS](https://github.com/usail-hkust/LLMTSCS), commit `d5d4180f34edb843e1d1b462d5846c75d6d4533a` | MIT, as stated by the LLMTSCS authors |
| CoordLight baseline | `baselines/coordlight/*.patch` (patches only) | [CoordLight](https://github.com/marmotlab/CoordLight), commit `be41b848a743cfa4a2814792cf3455dacd13bb36` | MIT, see `baselines/coordlight/LICENSE-CoordLight` |
| Hangzhou, Jinan and New York datasets | `data/` | [traffic-signal-control.github.io](https://traffic-signal-control.github.io), as distributed with LLMTSCS / CoordLight | see the original source |
| SUMO versions of the grid scenarios | `sumo_scenarios/anon_*` | converted from `data/` with `d2marl_sumo/cityflow2sumo.py` | as the source data |
| Ingolstadt21 network and demand | `sumo_scenarios/ingolstadt21/` | [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO); routes repaired with `duarouter` | GPL-3.0, see `sumo_scenarios/ingolstadt21/LICENSE` |
| Cologne8 network and demand | `sumo_scenarios/cologne8/` | [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO), based on the TAPAS Cologne scenario; routes repaired with `duarouter` | CC BY-NC-SA 3.0, see `sumo_scenarios/cologne8/LICENSE` |

Not distributed here and installed separately: [CityFlow](https://github.com/cityflow-project/CityFlow) (Apache-2.0) and [SUMO](https://eclipse.dev/sumo) (EPL-2.0).

The MIT license in `LICENSE` covers the code written for this repository. The files listed above keep the licenses of their origin.
