"""
LLMLight agent (LLMLight-13B in the baseline table) running in the SUMO environment.

Why it can be used unchanged
------------------
The policy input of LLMLight consists of two quantities:
    1. the number of queued vehicles per movement                    queue_len
    2. the number of approaching vehicles per movement, binned by
       distance to the stop line                                       cells[0..3]
They are rendered into an English prompt and the 13B model answers with
<signal>XXX</signal>. The simulator only supplies these two numbers, so the method
itself is unchanged on SUMO as long as both quantities are defined the same way.

Definitions (checked against LLMTSCS/utils/my_utils.py:get_state_detail)
    queue_len  = sum of halting vehicles on the lanes of the movement
                 -> lane_num_waiting_vehicle_in in sumo_env (same definition)
    cells      = lane_pos = lane length - distance travelled (distance to the stop line),
                 bins (0,L/10] (L/10,L/3] (L/3,2L/3] rest,
                 only vehicles with speed > 0.1 m/s
                 -> lane_cells feature in sumo_env, same implementation
    prompt     = copied verbatim from my_utils.getPrompt / state2text
    parsing    = same regex <signal>(.*?)</signal>; on failure the action falls back to 0

Phase numbering
    LLMTSCS: four_phase_list = {'ETWT':0, 'NTST':1, 'ELWL':2, 'NLSL':3}
    movement slots here: [WL,WT,WR, EL,ET,ER, NL,NT,NR, SL,ST,SR];
    actions 0..3 are WT+ET / NT+ST / WL+EL / NL+SL
    (see the phase_1..phase_4 comments in models/maxpressure_agent.py),
    identical to LLMLight, so action2code is used as is.

Batching
    A scenario has 16-48 intersections; one 13B call per intersection and step
    is infeasible (48 x 240 = 11520 generations). utils/oneline.py therefore calls
    prepare_batch once per step before the action loop; this class runs the
    prompts of all intersections as one vLLM batch and each agent picks its own
    result. The original LLMTSCS implementation batches the same way.
"""

import os
import re

from .agent import Agent

# ---- constants copied verbatim from LLMTSCS/utils/my_utils.py; do not change ----
LOCATION_DICT = {"N": "North", "S": "South", "E": "East", "W": "West"}
FOUR_PHASE_LIST = {"ETWT": 0, "NTST": 1, "ELWL": 2, "NLSL": 3}
PHASE_EXPL = {
    "NTST": "- NTST: Northern and southern through lanes.",
    "NLSL": "- NLSL: Northern and southern left-turn lanes.",
    "ETWT": "- ETWT: Eastern and western through lanes.",
    "ELWL": "- ELWL: Eastern and western left-turn lanes.",
}
# 12 movement slots here -> the 8 movement names of LLMLight
SLOT = {"WL": 0, "WT": 1, "EL": 3, "ET": 4, "NL": 6, "NT": 7, "SL": 9, "ST": 10}

SIGNAL_RE = re.compile(r"<signal>(.*?)</signal>", re.S)

# Final prompt format: LLMTSCS does not use a chat template but flattens the
# system/user parts into the Alpaca format used for fine-tuning; another format
# (e.g. Llama [INST]) breaks the <signal> output format of the 13B model.
# See LLMTSCS/utils/llm_aft_trainer.py:1107.
def _flatten(msgs):
    return (msgs[0]["content"] + "\n\n### Instruction:\n"
            + msgs[1]["content"] + "\n\n### Response:\n")


def _state2text(queue, cells):
    """Same as my_utils.state2text: Segment 3 = cells[2] + cells[3]."""
    txt = ""
    for p in FOUR_PHASE_LIST:
        l1, l2 = p[:2], p[2:]
        i1, i2 = SLOT[l1], SLOT[l2]
        q1, q2 = int(queue[i1]), int(queue[i2])
        c1, c2 = cells[i1], cells[i2]
        s1 = (c1[0], c2[0])
        s2 = (c1[1], c2[1])
        s3 = (c1[2] + c1[3], c2[2] + c2[3])
        n1, n2 = LOCATION_DICT[l1[0]], LOCATION_DICT[l2[0]]
        txt += (f"Signal: {p}\n"
                f"Relieves: {PHASE_EXPL[p][8:-1]}\n"
                f"- Early queued: {q1} ({n1}), {q2} ({n2}), {q1 + q2} (Total)\n"
                f"- Segment 1: {s1[0]} ({n1}), {s1[1]} ({n2}), {s1[0] + s1[1]} (Total)\n"
                f"- Segment 2: {s2[0]} ({n1}), {s2[1]} ({n2}), {s2[0] + s2[1]} (Total)\n"
                f"- Segment 3: {s3[0]} ({n1}), {s3[1]} ({n2}), {s3[0] + s3[1]} (Total)\n\n")
    return txt


def _get_prompt(state_txt):
    """Same as my_utils.getPrompt. Changing it changes the evaluated method."""
    return [
        {"role": "system",
         "content": "You are an expert in traffic management. You can use your "
                    "knowledge of traffic commonsense to solve this traffic "
                    "signal control tasks."},
        {"role": "user",
         "content": "A traffic light regulates a four-section intersection with northern, southern, eastern, and western "
                    "sections, each containing two lanes: one for through traffic and one for left-turns. Each lane is "
                    "further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the "
                    "middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching "
                    "vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and "
                    "await passage permission. Approaching vehicles will arrive at the intersection in the future.\n\n"
                    "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                    "specific lanes. The state of the intersection is listed below. It describes:\n"
                    "- The group of lanes relieving vehicles' flow under each signal phase.\n"
                    "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                    "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                    + state_txt +
                    "Please answer:\n"
                    "Which is the most effective traffic signal that will most significantly improve the traffic "
                    "condition during the next phase?\n\n"
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- You can only choose one of the signals listed above.\n"
                    "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                    "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                    "- Your choice can only be given after finishing the analysis.\n"
                    "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."}
    ]


class LLMLightAgent(Agent):

    # the 13B weights are loaded once and shared by all intersections
    _llm = None
    _tok = None
    _plan = {}          # step -> [action of each intersection]
    # Three separate counters: "malformed output" and "the model abstained" are
    # different things (an abstention is a well-formed <signal>NONE</signal>).
    _n_call = 0
    _n_unparsed = 0     # no <signal> tag at all: prompt/weights mismatch, investigate
    _n_abstain = 0      # tag present but not one of the four phases (NONE, N/A, ...)
    _n_abstain_empty = 0  # abstentions at intersections without any vehicle
    _fails = []         # raw outputs without a tag, kept for inspection
    _abstains = []      # raw outputs of abstentions, kept for inspection
    _seed = 0

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 cnt_round, intersection_id):
        super(LLMLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.idx = int(intersection_id)
        self.action = 0
        # sampling seed = environment seed, so a rerun with the same seed gives the same decisions
        LLMLightAgent._seed = int(dic_traffic_env_conf.get("SEED", 0))

    # ---------------- model ----------------
    @classmethod
    def _ensure_llm(cls, conf):
        if cls._llm is not None:
            return
        path = conf.get("LLM_PATH") or os.environ.get("LLMLIGHT_PATH")
        if not path:
            raise RuntimeError(
                "No LLM weight path given. Use -hp LLM_PATH=/abs/path or "
                "the environment variable LLMLIGHT_PATH.")
        import vllm
        from transformers import AutoTokenizer
        import torch
        print("[LLMLight] loading %s ..." % path)
        cls._llm = vllm.LLM(model=path, tokenizer=path, dtype=torch.bfloat16,
                            gpu_memory_utilization=float(
                                conf.get("GPU_MEM_UTIL", 0.90)),
                            max_model_len=int(conf.get("MAX_MODEL_LEN", 4096)))
        cls._tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        cls._tok.pad_token_id = 0
        print("[LLMLight] ready")

    # ---------------- batching hook ----------------
    @classmethod
    def prepare_batch(cls, count, state_list, conf):
        """
        Called by oneline.py once per step before the action loop.
        Runs the prompts of all intersections as one batch; generating them one
        by one would take more than ten hours per episode.
        """
        cls._ensure_llm(conf)
        prompts = []
        for st in state_list:
            q = st.get("lane_num_waiting_vehicle_in")
            c = st.get("lane_cells")
            if q is None or c is None:
                raise RuntimeError(
                    "state lacks lane_num_waiting_vehicle_in / lane_cells; "
                    "LIST_STATE_FEATURE must contain both.")
            prompts.append(_flatten(_get_prompt(_state2text(q, c))))

        import vllm
        # Sampling parameters copied from LLMTSCS/utils/llm_aft_trainer.py:1051.
        # temperature=0.1, not 0: the original samples with temperature, and greedy
        # decoding would be a different baseline.
        params = vllm.SamplingParams(
            top_k=50, top_p=1.0, temperature=0.1,
            max_tokens=int(conf.get("NEW_MAX_TOKENS", 1024)),
            seed=cls._seed + count)   # fixed per step, reproducible
        outs = cls._llm.generate(prompts, params)

        actions = []
        for o in outs:
            txt = o.outputs[0].text
            # As in the original: findall and take the **last** <signal>, not the
            # first; the model may mention several signals while reasoning.
            sigs = SIGNAL_RE.findall(txt)
            cls._n_call += 1
            sig = sigs[-1].strip() if sigs else None
            if sig in FOUR_PHASE_LIST:
                actions.append(FOUR_PHASE_LIST[sig])
            else:
                # In both cases the action is ETWT (=0), as in LLMTSCS; this is the
                # behaviour of the original method, not a fallback of ours.
                # The two cases are counted separately:
                if sig is None:
                    cls._n_unparsed += 1
                    cls._fails.append(txt[-500:])
                else:
                    cls._n_abstain += 1
                    if len(cls._abstains) < 5:
                        cls._abstains.append((sig, txt[-300:]))
                    # An abstention at an empty intersection is harmless: with no
                    # vehicle, the chosen phase has no effect. Counted separately so
                    # that its share can be reported.
                    st = state_list[len(actions)]
                    q = st.get("lane_num_waiting_vehicle_in") or []
                    c = st.get("lane_cells") or []
                    def _tot(x):
                        t = 0.0
                        for v in x:
                            if isinstance(v, (list, tuple)):
                                t += sum(float(u) for u in v)
                            else:
                                t += float(v)
                        return t
                    if _tot(q) + _tot(c) == 0.0:
                        cls._n_abstain_empty += 1
                actions.append(0)
        cls._plan[count] = actions

    @classmethod
    def report(cls):
        if not cls._n_call:
            return
        n = cls._n_call
        r_unparsed = 100.0 * cls._n_unparsed / n
        r_abstain = 100.0 * cls._n_abstain / n
        print("[LLMLight] %d generations: no <signal> tag %d (%.2f%%); "
              "abstentions (tag not one of the four phases) %d (%.2f%%), %d of them at empty intersections. "
              "Both fall back to ETWT as in the original implementation"
              % (n, cls._n_unparsed, r_unparsed,
                 cls._n_abstain, r_abstain, cls._n_abstain_empty))
        # Only missing tags indicate a prompt/weights mismatch.
        if r_unparsed > 5.0:
            print("!! more than 5%% of the outputs have no tag: prompt template and "
                  "weights may not match; this run does not measure LLMLight.")
            for t in cls._fails[:2]:
                print("---- output without tag (last 500 characters) ----\n%s\n" % t)
        # Abstentions are not errors, but many abstentions at non-empty
        # intersections mean the policy degrades to a fixed phase there.
        if cls._n_abstain:
            n_busy = cls._n_abstain - cls._n_abstain_empty
            print("   abstention samples (phase name / last 300 characters):")
            for sg, t in cls._abstains[:2]:
                print("   [%s] %s\n" % (sg, t))
            if n_busy > 0.05 * n:
                print("!! %d abstentions at non-empty intersections (%.2f%%); "
                      "these decisions degrade to a fixed phase"
                      % (n_busy, 100.0 * n_busy / n))

    # ---------------- decision ----------------
    def choose_action(self, count, state):
        if state["cur_phase"][0] == -1:      # yellow transition, keep the phase
            return self.action
        plan = self._plan.get(count)
        if plan is None or self.idx >= len(plan):
            # No batch result: the batching hook did not run. Fail loudly instead
            # of silently falling back to per-intersection inference, which would
            # turn a half-hour evaluation into more than ten hours.
            raise RuntimeError(
                "No batch result for step %s; the prepare_batch hook in "
                "utils/oneline.py did not run." % count)
        self.action = plan[self.idx]
        return self.action
