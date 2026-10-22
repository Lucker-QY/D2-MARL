"""
D2-MARL agent: dual-phase attention (intra-phase branch + spatial graph-attention
branch fused by a learned gate) with delay-risk state features, a bounded delay
penalty in the reward and stratified replay of severe-delay samples.
"""
import numpy as np
import os
from .agent import Agent
from .phase_util import phase_code
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time
import csv
import pandas as pd


def build_memory():
    return []


class D2MARLAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        self.save_reward_metrics = dic_agent_conf.get('SAVE_REWARD_METRICS', False)
        self.reward_metrics_save_freq = dic_agent_conf.get('REWARD_METRICS_SAVE_FREQ', 100)
        self.debug = dic_agent_conf.get('DEBUG', False)

        if self.save_reward_metrics:
            self.reward_metrics_history = []
            self.episode_count = cnt_round if cnt_round is not None else 0
            print(f"Reward metric logging enabled, every {self.reward_metrics_save_freq} steps")

            self.reward_metrics_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "reward_metrics")
            if not os.path.exists(self.reward_metrics_dir):
                os.makedirs(self.reward_metrics_dir)

        # always initialise episode_count; used by on_episode_end and others
        self.episode_count = cnt_round if cnt_round is not None else 0

        # ================== enable attention logging early ==================
        self.save_attention = dic_agent_conf.get("INTRA_PHASE_ATTENTION", False)
        if self.save_attention:
            self.attention_logs = []
        # ===================================================================

        super(D2MARLAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()

        # ================= action mask =================
        # On grid networks every intersection has four phases, so no mask is needed.
        # On real non-grid networks (e.g. Ingolstadt) a T-junction has only 2-3
        # phases; without a mask argmax could pick a phase that does not exist, and
        # max_a' Q(s',a') in the TD target would maximise over non-existent actions.
        #
        # The environment computes the mask from the topology at reset() and writes it
        # into the config, shape [num_agents, num_actions], 1 = phase exists. Without
        # a mask all entries are 1 and the behaviour is unchanged.
        _mask = self.dic_traffic_env_conf.get("ACTION_MASK", None)
        if _mask is None:
            self.action_mask = np.ones((self.num_agents, self.num_actions),
                                       dtype=np.float32)
            self.has_action_mask = False
        else:
            self.action_mask = np.asarray(_mask, dtype=np.float32)
            assert self.action_mask.shape == (self.num_agents, self.num_actions), \
                "ACTION_MASK must have shape [{0}, {1}], got {2}".format(
                    self.num_agents, self.num_actions, self.action_mask.shape)
            self.has_action_mask = bool((self.action_mask == 0).any())
        # every intersection needs at least one available phase
        assert (self.action_mask.sum(axis=1) > 0).all(), \
            "an intersection has no available phase; check the network conversion"
        # ============================================================

        if cnt_round == 0:
            self.q_network = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)),
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        # bind here only when cnt_round == 0 (for > 0 load_network has already bound it)
        if self.save_attention and cnt_round == 0:
            self._bind_attention_extractor()

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N


    @staticmethod
    def MLP(ins, layers=None, name_suffix=""):
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name=f'Dense_embed_{name_suffix}_{layer_index}')(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name=f'Dense_embed_{name_suffix}_{layer_index}')(h)

        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)
        neighbor_repr = Lambda(lambda x: tf.matmul(x[0], x[1]))([in_nei, neighbor_repr])

        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(agent_repr_head)

        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(
            neighbor_repr_head)

        att = Lambda(lambda x: K.softmax(tf.matmul(x[0], x[1], transpose_b=True)),
                      name=f'spatial_attn_score_{suffix}')([agent_repr_head, neighbor_repr_head])
        att_record = Reshape((self.num_agents, head, self.num_neighbors))(att)

        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(
            neighbor_hidden_repr_head)
        out = Lambda(lambda x: K.mean(tf.matmul(x[0], x[1]), axis=2))([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal', name='MLP_after_relation_%d' % suffix)(
            out)
        return out, att_record

    def IntraPhaseAttention(self, in_feats, num_phases=4, d_model=16, name_suffix=""):
        """
        Intra-phase attention.
        Self-attention among the tokens of one intersection's phase/delay descriptor.

        in_feats shape: [Batch, num_agents, len_feature]
            - D2-MARL input: 8 (cur_phase) + 12 (delay burden) + 12 (max wait) = 32
            - cut into num_phases = 4 consecutive tokens of feat_dim = 32 // 4 = 8 entries

        Steps:
        1. cut the flat len_feature vector into num_phases tokens of feat_dim entries
        2. project each token from feat_dim to d_model with a Dense layer
        3. compute Q, K, V and self-attention
        4. flatten the result and pass it to the MLP
        """
        # 1. per-token size from the actual input size
        feat_dim = int(in_feats.shape[-1]) // num_phases
        # projection size d_model * num_phases, so the flattened output has the input's size
        embed_dim = d_model * num_phases

        # 2. reshape into num_phases tokens: [Batch, num_agents, num_phases, feat_dim]
        x = Reshape((self.num_agents, num_phases, feat_dim))(in_feats)

        # 3. token embedding: [Batch, num_agents, num_phases, embed_dim]
        x = Dense(embed_dim, activation='relu', name=f'intra_embed_{name_suffix}')(x)

        # 4. Q, K, V
        Q = Dense(embed_dim, name=f'intra_q_{name_suffix}')(x)
        K_layer = Dense(embed_dim, name=f'intra_k_{name_suffix}')(x)
        V = Dense(embed_dim, name=f'intra_v_{name_suffix}')(x)

        # 5. attention scores: Softmax(Q * K^T / sqrt(d))
        # K_layer: [Batch, num_agents, num_phases, embed_dim] -> [Batch, num_agents, embed_dim, num_phases]
        K_T = Lambda(lambda t: K.permute_dimensions(t, (0, 1, 3, 2)))(K_layer)

        # 5. attention scores (tf.matmul)
        # Score shape: [Batch, num_agents, num_phases, num_phases]
        score = Lambda(lambda args: tf.matmul(args[0], args[1]))([Q, K_T])
        score = Lambda(lambda t: t / np.sqrt(embed_dim))(score)
        # named so that the scores can be extracted
        score = Lambda(lambda t: K.softmax(t, axis=-1), name=f"intra_attn_score_{name_suffix}")(score)

        # 6. weighted sum of values (tf.matmul)
        # out shape: [Batch, num_agents, num_phases, embed_dim]
        out = Lambda(lambda args: tf.matmul(args[0], args[1]))([score, V])

        # 7. flatten: [Batch, num_agents, num_phases * embed_dim] = [Batch, num_agents, len_feature]
        out = Reshape((self.num_agents, num_phases * embed_dim))(out)

        return out

    def adjacency_index2matrix(self, adjacency_index):
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        # cur_phase is -1 (all_yellow_phase_index) during the all-yellow
                        # transition, while the PHASE keys are 1..N; a direct lookup would
                        # raise KeyError: -1. During yellow the encoding of the previous
                        # phase is used, see models/phase_util.py.
                        tmp.extend(phase_code(self, s[i][feature][0]))
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])

            feats0.append(tmp)
        feats = np.array([feats0])
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    # ============ action mask helpers ============
    def _apply_action_mask(self, q):
        """
        q: [num_agents, num_actions]. Pushes the Q-values of invalid phases to -inf
        so that neither argmax nor max can select them. Unchanged if the mask is all ones.
        """
        if not self.has_action_mask:
            return q
        n = q.shape[0]
        mask = self.action_mask[:n]
        return np.where(mask > 0, q, -np.inf)

    def _random_valid_action(self, n_agents):
        """The random branch of epsilon-greedy also draws only among valid phases."""
        if not self.has_action_mask:
            return np.random.randint(self.num_actions, size=n_agents)
        out = np.empty(n_agents, dtype=np.int64)
        for i in range(n_agents):
            valid = np.flatnonzero(self.action_mask[i] > 0)
            out[i] = np.random.choice(valid)
        return out
    # =====================================================

    def choose_action(self, count, states):
        self.current_step = count

        if self.save_reward_metrics and not hasattr(self, 'reward_metrics_history'):
            self.reward_metrics_history = []

        if self.debug and count % 100 == 0:
            print(f"choose_action, step: {count}")

        xs = self.convert_state_to_input(states)

        # =================
        # call the model directly and convert the EagerTensor to numpy
        q_values = self.q_network(xs, training=False).numpy()

        # Mask non-existent phases before choosing. With an all-ones mask
        # masked_q equals q_values, so grid networks are unaffected.
        masked_q = self._apply_action_mask(q_values[0])

        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = self._random_valid_action(len(masked_q))
        else:
            action = np.argmax(masked_q, axis=1)

        if hasattr(self, 'save_reward_metrics') and self.save_reward_metrics:
            self._collect_reward_metrics(states, action, q_values)

            if count % self.reward_metrics_save_freq == 0 and count > 0:
                self.save_reward_metrics_to_csv()

        # =================
        # log attention weights (only in evaluation/test episodes, EPSILON very small)
        if getattr(self, "save_attention", False) and getattr(self, "has_bound_attention", False) and self.dic_agent_conf["EPSILON"] < 0.05:
            attn_scores = self.attn_extractor(xs, training=False)
            if not isinstance(attn_scores, list):
                attn_scores = [attn_scores]

            extracted_arrays = [t.numpy() for t in attn_scores]

            # unpack the extracted outputs according to the architecture
            if self.use_intra_attention:
                intra_attn_scores = extracted_arrays[0]    # [Batch, N, Phase, Phase]
                gating_alpha_scores = extracted_arrays[1]  # [Batch, N, 1]
                spatial_attn_scores = extracted_arrays[2:] # graph attention of each CNN layer
            else:
                intra_attn_scores = None
                gating_alpha_scores = None
                spatial_attn_scores = extracted_arrays

            intersections_data = []
            for inter_id in range(self.num_agents):
                intra_matrix = intra_attn_scores[0, inter_id].tolist() if intra_attn_scores is not None else []
                alpha_val = float(gating_alpha_scores[0, inter_id, 0]) if gating_alpha_scores is not None else 0.0

                spatial_matrices = {}
                for cnn_idx, sp_scores in enumerate(spatial_attn_scores):
                    sp_matrix = sp_scores[0, inter_id, :, 0, :].tolist()
                    spatial_matrices[f'layer_{cnn_idx}'] = sp_matrix

                # Also log the delay state of this intersection at this step, so that
                # the gate alpha can be paired with the delay burden at the same time.
                # The attention log comes from the evaluation episode (logged only when
                # EPSILON < 0.05); the train_round samples belong to another episode
                # and cannot be paired, so the delay state is recorded here.
                st = states[inter_id] if inter_id < len(states) else {}
                def _sum(key):
                    v = st.get(key)
                    try:
                        return float(np.sum(v)) if v is not None else None
                    except Exception:
                        return None

                intersections_data.append({
                    "inter_id": inter_id,
                    "action_taken": int(action[inter_id]),
                    "q_values": q_values[0][inter_id].tolist(),
                    "gating_alpha": alpha_val,  # gate value
                    "delay_burden": _sum("lane_delay_burden"),
                    "max_wait_norm": _sum("lane_raw_max_wait"),
                    "over_limit": _sum("lane_over_limit_num"),
                    "queue": _sum("traffic_movement_pressure_queue_efficient"),
                    "intra_attention": intra_matrix,
                    "spatial_attention": spatial_matrices
                })

            record = {
                "step": count,
                "intersections": intersections_data
            }
            self.attention_logs.append(record)

        return action

    def _collect_reward_metrics(self, states, actions, q_values):
        if not hasattr(self, 'reward_metrics_history'):
            self.reward_metrics_history = []

        traffic_metrics = self.extract_traffic_metrics(states)

        record = {
            'step': getattr(self, 'current_step', 0),
            'timestamp': time.time(),
            'traffic_metrics': traffic_metrics,
            'actions': actions.tolist() if isinstance(actions, np.ndarray) else actions,
            'q_values': q_values.tolist() if isinstance(q_values, np.ndarray) else q_values,
        }

        self.reward_metrics_history.append(record)

        if len(self.reward_metrics_history) >= 1000:
            self.save_reward_metrics_to_csv()
            self.reward_metrics_history = []

    def extract_traffic_metrics(self, states):
        metrics = {}
        for i in range(min(self.num_agents, len(states))):
            queue_length = 0
            pressure = 0

            if 'lane_num_waiting_vehicle_in' in states[i]:
                queue_length = sum(states[i]['lane_num_waiting_vehicle_in'])

            if 'pressure' in states[i]:
                pressure = np.sum(states[i]['pressure']) if isinstance(states[i]['pressure'], (list, np.ndarray)) else states[i]['pressure']

            phase = states[i].get('cur_phase', [0])[0] if 'cur_phase' in states[i] else 0

            metrics[f'intersection_{i}'] = {
                'queue_length': queue_length,
                'pressure': pressure,
                'phase': phase
            }
        return metrics

    def save_reward_metrics_to_csv(self):
        if not hasattr(self, 'reward_metrics_history') or not self.reward_metrics_history:
            return

        file_name = f"reward_metrics_ep{self.episode_count}_inter{self.intersection_id}.csv"
        file_path = os.path.join(self.reward_metrics_dir, file_name)

        try:
            data = []
            for record in self.reward_metrics_history:
                row = {
                    'step': record['step'],
                    'timestamp': record['timestamp'],
                }

                for inter_id, metrics in record['traffic_metrics'].items():
                    for metric_name, metric_value in metrics.items():
                        row[f'{inter_id}_{metric_name}'] = metric_value

                if isinstance(record['actions'], list) and len(record['actions']) > 0:
                    row['action'] = record['actions'][0]

                if isinstance(record['q_values'], list) and len(record['q_values']) > 0:
                    for i, q_value in enumerate(record['q_values'][0]):
                        row[f'q_value_{i}'] = q_value

                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            if self.debug:
                print(f"Saved reward metrics to {file_path}")
        except Exception as e:
            print(f"Error while saving reward metrics: {e}")

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        slice_size = len(memory[0])
        _adjs = []
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])
                _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
            _adjs.append(_adj)
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)
        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)
        target = self.q_network([_state2, _adjs2])
        next_state_qvalues = self.q_network_bar([_next_state2, _adjs2])
        final_target = np.copy(target)

        # max_a' Q(s',a') in the TD target is masked as well; masking only
        # choose_action is not enough, because a bootstrap target that maximises
        # over invalid phases would still bias the whole Q-function.
        next_q = np.asarray(next_state_qvalues)
        if self.has_action_mask:
            next_q = np.where(self.action_mask[None, :, :] > 0, next_q, -np.inf)

        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * np.max(next_q[i, j])

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

    def build_network(self, MLP_layers=[32, 32]):
        from tensorflow.keras.layers import Lambda, Concatenate

        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        In.append(Input(shape=(self.num_agents, self.len_feature), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        use_intra_attention = self.dic_agent_conf.get("INTRA_PHASE_ATTENTION", False)

        # ================= 1. state layout and feature slices =================
        feature_slices = {}
        start_idx = 0
        used_features = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]  # exclude the trailing adjacency_matrix

        for feat in used_features:
            length = 8 if "phase" in feat else 12
            feature_slices[feat] = Lambda(
                lambda x, s=start_idx, e=start_idx+length: x[:, :, s:e],
                name=f"slice_{feat}"
            )(In[0])
            start_idx += length

        # ================= 2. spatial branch (graph attention over neighbours) =================
        spatial_feats_list = [feature_slices[f] for f in used_features if f != "lane_delay_burden"]
        spatial_input = Concatenate(name="concat_spatial")(spatial_feats_list) if len(spatial_feats_list) > 1 else spatial_feats_list[0]

        branch_spatial = self.MLP(spatial_input, layers=[128, 128], name_suffix="spatial")

        for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
            branch_spatial, _ = self.MultiHeadsAttModel(
                branch_spatial, In[1], d_in=128, h_dim=CNN_layer_size[0],
                dout=128, head=CNN_heads[CNN_layer_index], suffix=CNN_layer_index
            )
        # branch_spatial shape: [Batch, N, 128]

        # ================= 3. intra-phase branch and gating network =================
        if use_intra_attention:
            # base features
            intra_inputs = [feature_slices["cur_phase"]]
            # pressure feature: the first pressure feature present in LIST_STATE_FEATURE
            # (-spillback replaces the efficient pressure by its capacity-discounted version)
            _press = next((k for k in ("traffic_movement_pressure_queue_capacity",
                                       "traffic_movement_pressure_queue_efficient",
                                       "traffic_movement_pressure_queue")
                           if k in feature_slices), None)
            if _press is None:
                raise KeyError(
                    "the gate needs a pressure feature, none in LIST_STATE_FEATURE: %s"
                    % list(feature_slices))
            gate_inputs = [feature_slices[_press]]

            # delay burden feature
            if "lane_delay_burden" in feature_slices:
                intra_inputs.append(feature_slices["lane_delay_burden"])
                gate_inputs.append(feature_slices["lane_delay_burden"])

            # delay-risk features
            if "lane_raw_max_wait" in feature_slices:
                intra_inputs.append(feature_slices["lane_raw_max_wait"])
                gate_inputs.append(feature_slices["lane_raw_max_wait"])
            if "lane_over_limit_num" in feature_slices:
                # the number of over-limit movements is an aggregate indicator for the gate
                gate_inputs.append(feature_slices["lane_over_limit_num"])
            if "lane_downstream_free_capacity" in feature_slices:
                # Non-grid adaptation (Section IV-C, -spillback): the free-storage ratio of the
                # downstream link enters the gate and the intra-phase branch, so the branch that
                # tracks long waits also sees whether the exit can take the released vehicles.
                # The intra-phase input becomes 32 + 12 = 44 values, i.e. four tokens of width 11.
                gate_inputs.append(feature_slices["lane_downstream_free_capacity"])
                intra_inputs.append(feature_slices["lane_downstream_free_capacity"])

            intra_input = Concatenate(name="concat_intra")(intra_inputs)
            gate_input = Concatenate(name="concat_gate")(gate_inputs)

            intra_d_model = self.dic_agent_conf.get("INTRA_PHASE_ATTENTION_D_MODEL", 8)
            intra_feature = self.IntraPhaseAttention(intra_input, num_phases=self.num_actions, d_model=intra_d_model, name_suffix="1")

            branch_intra = self.MLP(intra_feature, layers=[128, 128], name_suffix="intra")
            # branch_intra shape: [Batch, N, 128]

            # 3.2 gating network
            #
            # FIXED_GATING_ALPHA (default None, i.e. the learned gate of the paper) fixes
            # alpha to a constant; it is kept only as a diagnostic option.
            fixed_alpha = self.dic_agent_conf.get("FIXED_GATING_ALPHA", None)
            if fixed_alpha is not None:
                fa = float(fixed_alpha)
                # Build a layer with the same name so that weight files and attention
                # extraction stay compatible; its output is replaced by the constant.
                _ = Dense(16, activation='relu', name='gating_hidden')(gate_input)
                alpha = Lambda(lambda x: tf.ones_like(x[:, :, :1]) * fa,
                               name='gating_alpha')(gate_input)
                print("==> [ablation] gating coefficient fixed to {0}".format(fa))
            else:
                gate_hidden = Dense(16, activation='relu', name='gating_hidden')(gate_input)
                alpha = Dense(1, activation='sigmoid', name='gating_alpha')(gate_hidden)
            # alpha shape: [Batch, N, 1]

            # 3.3 fusion (Keras broadcasting aligns the dimensions)
            fusion_feat = Lambda(
                lambda args: args[0] * args[1] + (1.0 - args[0]) * args[2],
                name="feature_fusion"
            )([alpha, branch_intra, branch_spatial])

        else:
            fusion_feat = branch_spatial

        # ================= 4. output layer =================
        out = Dense(self.num_actions, kernel_initializer='random_normal', name='action_layer')(fusion_feat)
        model = Model(inputs=In, outputs=out)

        model.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]), loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # keep the history returned by fit
        history = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])

        # ================= record and save the loss =================
        train_loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        if train_loss:
            final_loss = train_loss[-1]
            final_val_loss = val_loss[-1] if val_loss else "N/A"
            actual_epochs = len(train_loss)

            # log directory: work_directory/loss_logs
            loss_log_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "loss_logs")
            if not os.path.exists(loss_log_dir):
                os.makedirs(loss_log_dir)

            loss_file = os.path.join(loss_log_dir, f"loss_inter_{self.intersection_id}.csv")
            file_exists = os.path.exists(loss_file)

            with open(loss_file, "a") as f:
                if not file_exists:
                    f.write("round,epochs_run,final_loss,final_val_loss\n")
                # self.episode_count is the current training round (set in __init__)
                f.write(f"{self.episode_count},{actual_epochs},{final_loss},{final_val_loss}\n")
        # ==========================================================

    def build_network_from_copy(self, network_copy):
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def _bind_attention_extractor(self):
        if not getattr(self, "save_attention", False):
            return

        try:
            out_layers = []
            self.use_intra_attention = self.dic_agent_conf.get("INTRA_PHASE_ATTENTION", False)

            # 1. intra-phase attention, if enabled
            if self.use_intra_attention:
                out_layers.append(self.q_network.get_layer("intra_attn_score_1").output)
                # gate alpha
                out_layers.append(self.q_network.get_layer("gating_alpha").output)

            # 2. spatial graph attention (one per CNN layer)
            for i in range(len(self.CNN_layers)):
                out_layers.append(self.q_network.get_layer(f"spatial_attn_score_{i}").output)

            if out_layers:
                self.attn_extractor = Model(inputs=self.q_network.input, outputs=out_layers)
                self.has_bound_attention = True
            else:
                self.has_bound_attention = False
        except Exception as e:
            self.has_bound_attention = False
            print(f"Warning: Failed to create attention extractor (Layers might be missing). Error: {e}")

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

        # ==========================================
        # rebind the extraction model after every load
        if getattr(self, "save_attention", False):
            self._bind_attention_extractor()
        # ==========================================

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

        if hasattr(self, 'save_reward_metrics') and self.save_reward_metrics and self.reward_metrics_history:
            self.save_reward_metrics_to_csv()

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def on_episode_end(self):
        """
        Called at the end of every episode; saves the attention log.
        Triggered through the agent.on_episode_end() hook in generator.py / model_test.py.
        """
        if getattr(self, "save_attention", False) and len(self.attention_logs) > 0:
            import json
            log_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "attention_logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_path = os.path.join(log_dir, f"attn_log_ep{self.episode_count}.json")
            with open(file_path, "w") as f:
                json.dump(self.attention_logs, f, indent=4, ensure_ascii=False)
            print(f"--> attention log saved to: {file_path}")
            self.attention_logs = []  # reset for the next episode


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        base_config.update(config)
        return base_config
