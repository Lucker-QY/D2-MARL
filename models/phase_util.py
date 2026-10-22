# -*- coding: utf-8 -*-
"""Single entry point for looking up the phase encoding.

The keys of `dic_traffic_env_conf['PHASE']` are 1..N, but during the all-yellow
transition `cur_phase` is **-1** (all_yellow_phase_index in the environment).
A direct lookup raises `KeyError: -1`, which is logged only as `Error: -1`.

Decision points normally do not fall on a yellow phase, but any mechanism that
changes the phase between decision points can make them do so.


During yellow the encoding of the **last valid phase** is used: it describes the
phase being left, and runs that never reach this branch are unaffected.
"""


def phase_code(agent, phase_id):
    """Return the encoding of phase_id, falling back when phase_id is not in the table.

    agent is only used to store `_last_phase_key`; it needs no interface.
    """
    table = agent.dic_traffic_env_conf["PHASE"]
    if phase_id not in table:
        phase_id = getattr(agent, "_last_phase_key", None)
        if phase_id not in table:
            phase_id = sorted(table)[0]
    agent._last_phase_key = phase_id
    return table[phase_id]
