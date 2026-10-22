# ---------------------------------------------------------------------------
# Agent registry.
#
# Agents are imported one by one. Most agents need TensorFlow, while LLMLight
# needs vLLM/PyTorch, and the two usually live in different environments.
# A plain import of every agent would fail in an environment without TF with
#   run_baseline.py -> utils.utils -> utils.config -> models.colight_agent
#     -> import tensorflow
# raising ModuleNotFoundError although LLMLight itself does not need TF.
#
# An agent whose module cannot be imported is left out of the registry and
# produces a clear error only when it is requested.
# ---------------------------------------------------------------------------
class _Unavailable(object):
    """Placeholder for an agent whose module cannot be imported here; keeps the reason."""

    def __init__(self, cls, why):
        self.cls = cls
        self.why = why


def _reg(module, cls):
    """Import models.<module>.<cls>; on failure return a placeholder with the reason."""
    try:
        mod = __import__("models." + module, fromlist=[cls])
        return getattr(mod, cls)
    except ImportError as e:          # missing TF / torch / vLLM
        return _Unavailable(cls, "%s: %s" % (type(e).__name__, e))


class _AgentRegistry(dict):
    """On a missing key, tell "unknown method" apart from "missing dependency"."""

    def __missing__(self, key):
        bad = _ALL_AGENTS.get(key)
        if isinstance(bad, _Unavailable):
            raise KeyError(
                "%r (%s) cannot be imported in this environment: %s | "
                "TF-based agents need TensorFlow, LLMLight needs torch + vLLM."
                % (key, bad.cls, bad.why))
        raise KeyError(
            "Unknown MODEL_NAME %r. Available: %s"
            % (key, ", ".join(sorted(self))))


FixedtimeAgent = _reg("fixedtime_agent", "FixedtimeAgent")
ActuatedAgent = _reg("actuated_agent", "ActuatedAgent")
RandomAgent = _reg("random_agent", "RandomAgent")
LLMLightAgent = _reg("llmlight_agent", "LLMLightAgent")
MaxPressureAgent = _reg("maxpressure_agent", "MaxPressureAgent")
MPLightAgent = _reg("mplight_agent", "MPLightAgent")
CoLightAgent = _reg("colight_agent", "CoLightAgent")
PressLightAgent = _reg("presslight", "PressLightAgent")
AttendLightAgent = _reg("attendlight_agent", "AttendLightAgent")
D2MARLAgent = _reg("d2marl_agent", "D2MARLAgent")


_ALL_AGENTS = {
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,
    "MPLight": MPLightAgent,
    "EfficientColight": CoLightAgent,
    "PressLight": PressLightAgent,
    "Attend": AttendLightAgent,
    "AdvancedColight": CoLightAgent,
    "D2MARL": D2MARLAgent,
    "Actuated": ActuatedAgent,      # ASC baseline
    "Random": RandomAgent,          # Random baseline
    "CoLight": CoLightAgent,        # original CoLight (not the Efficient/Advanced variants)
    "LLMLight": LLMLightAgent,      # LLMLight-13B
}

# keep only the agents that could be imported in this environment
DIC_AGENTS = _AgentRegistry(
    (k, v) for k, v in _ALL_AGENTS.items()
    if not isinstance(v, _Unavailable))


DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["LLMLight", "Random", "CoLight", "Fixedtime", "MaxPressure", "Actuated", "EfficientColight", "MPLight", "PressLight", "Attend", "AdvancedColight", "D2MARL"],
    "LIST_MODEL_NEED_TO_UPDATE": ["LLMLight", "Random", "CoLight", "Fixedtime", "MaxPressure", "EfficientColight", "MPLight", "PressLight", "Attend", "AdvancedColight", "D2MARL"],

    "SEED": 0,  # default random seed

    "NUM_LANE": 12,
    "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    # neighbourhood size of the graph models (CoLight family, D2-MARL).
    # TOP_K_ADJACENCY=5: each intersection attends to itself + its 4 nearest neighbours (softmax over these 5).
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "DEBUG": False,   # True enables debug output

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix",
    ],
    "DIC_REWARD_INFO": {
        "queue_length": -0.25,
        "max_wait_penalty": -0.25,
        "over_limit_penalty": -0.25,        # weight of the over-limit term (0 disables it)
    },

    # ── extended state and over-limit reward ──
    "USE_EXTENDED_STATE": True,             # extra delay-risk state features
    "WAIT_COUNT_THRESHOLD": 200.0,          # waiting time (s) above which a vehicle counts as over-limit

    # ── bounded delay penalty (overridden by run_d2marl.py) ──
    "WAIT_SOFT_LIMIT": 150.0,
    "WAIT_HARD_LIMIT": 300.0,
    "MAX_PENALTY_CAP": 2000.0,

    # ── Huber transition ──
    "HUBER_DELTA": 100.0,        # seconds beyond the hard limit at which the penalty turns linear
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    

    # attention export
    # topk: export the top-k neighbour attention (softmax over k neighbours per source)
    # full: export N x N (truly fully connected only if TOP_K_ADJACENCY == NUM_INTERSECTIONS)
    "ATTENTION_EXPORT_MODE": "topk",

    # intra-phase attention (D2MARLAgent)
    # True enables intra-phase self-attention
    # False falls back to plain CoLight (IA ablation)
    "INTRA_PHASE_ATTENTION": True,
    # INTRA_PHASE_ATTENTION_D_MODEL (default 8): projection size per token; the D2-MARL
    # input has 32 entries, i.e. 4 tokens of 8, and d_model=8 gives embed_dim=32.
    # stratified replay of severe-delay samples (run_d2marl.py overrides these)
    "USE_STRATIFIED_REPLAY": True,          # enable stratified replay
    "PENALTY_REPLAY_THRESHOLD": -50.0,      # reward below which a sample counts as severe
    "PENALTY_REPLAY_RATIO": 0.20,           # maximum share of severe samples per batch
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}
