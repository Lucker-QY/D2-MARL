"""
sumo_topology.py
================
Projects any SUMO network (non-grid, different phase counts, not four-legged) onto
the canonical "4 approaches x 3 turns = 12 lanes" observation slots of the D2-MARL
agent, together with a validity mask.

Design (consistent with the conventions of utils/cityflow_env.py):

1. approach order       : ["W", "E", "N", "S"]          (as Intersection.list_approachs)
2. turn order per approach: ["L", "T", "R"]            (derived from PHASE_MAP, see below)
   check: in config.py PHASE_MAP[0] = [1, 4, ...], PHASE_LIST[0] = 'WT_ET'
          slot 1 = second lane of W, slot 4 = second lane of E -> index 1 = through
          PHASE_MAP[2] = [0, 3, ...], PHASE_LIST[2] = 'WL_EL' -> index 0 = left
          hence the canonical order L, T, R.
   Note: SUMO numbers lanes from 0 = rightmost, opposite to this order, so lanes are
         placed **by the turn direction of their connections**, not by lane index;
         this also works on non-grid networks (unequal lane counts, missing turns).
3. control and observation are decoupled:
   - control uses the green phases of SUMO's own tlLogic (2-8 phases),
     action = index of the green phase, with an action mask for different phase counts;
   - observations are projected onto the 12 canonical slots; missing slots are zero
     and 0 in the mask.
   Non-grid topologies therefore need no change to the signal control itself, only
   a mask on the agent side.

Requires sumolib (shipped with SUMO; set SUMO_HOME).
"""

import os
import sys
import math
import json
from collections import OrderedDict, defaultdict

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)

try:
    import sumolib
except ImportError:  # pragma: no cover - required at run time
    raise EnvironmentError(
        "cannot import sumolib; install SUMO and set the SUMO_HOME environment variable."
    )


# ==========================================================================
# canonical slots
# ==========================================================================
APPROACHES = ["W", "E", "N", "S"]
TURNS = ["L", "T", "R"]
NUM_SLOTS = len(APPROACHES) * len(TURNS)          # 12

# compass bearing at the centre of each approach (the direction vehicles come "from")
_APPROACH_BEARING = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

# SUMO connection direction character -> canonical turn
# see https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#connections
_DIR_TO_TURN = {
    "s": "T",   # straight
    "t": "L",   # turnaround (U-turn), counted as left, as U_TURN_AS in converter.py
    "l": "L",   # left
    "r": "R",   # right
    "L": "L",   # partially left
    "R": "R",   # partially right
    "i": "L",   # invalid/internal, conservatively counted as left
}


def slot_index(approach, turn):
    """Canonical slot index: W_L=0, W_T=1, W_R=2, E_L=3, ... S_R=11"""
    return APPROACHES.index(approach) * len(TURNS) + TURNS.index(turn)


SLOT_NAMES = [f"{a}{t}" for a in APPROACHES for t in TURNS]


# ==========================================================================
# geometry helpers
# ==========================================================================
def _bearing(p_from, p_to):
    """Compass bearing from p_from to p_to (degrees, 0 = north, clockwise)."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    ang = math.degrees(math.atan2(dx, dy))
    return ang % 360.0


def _angular_gap(a, b):
    """Smallest angle between two bearings (0-180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _incoming_edge_approach_bearing(edge):
    """
    "Coming-from" bearing of an approach lane.
    The edge shape follows the driving direction and its last two points lead into the
    junction; vehicles come from the opposite direction.
    """
    shape = edge.getShape()
    p_end = shape[-1]
    p_prev = shape[-2] if len(shape) >= 2 else shape[0]
    # looking upstream from the junction = the direction vehicles come from
    return _bearing(p_end, p_prev)


def assign_approaches(incoming_edges):
    """
    Assign N approach edges by their coming-from bearing to the canonical W/E/N/S approaches.

    Two common cases on non-grid networks:
      - fewer than 4 legs (T-junction): the missing approach is masked as a whole;
      - more than 4 legs or two legs in one quadrant: the leg with the closer angle takes
        the approach and the other moves to the nearest free approach; if all four are
        taken, the leg overflows and is merged into the approach with the closest angle
        (recorded in the diagnostics).

    :param incoming_edges: list of sumolib Edges
    :return: (approach_to_edges, overflow, diag)
             approach_to_edges: {"W": [edge, ...], ...}, empty list for a missing approach
    """
    scored = []
    for e in incoming_edges:
        b = _incoming_edge_approach_bearing(e)
        scored.append((e, b))

    approach_to_edges = OrderedDict((a, []) for a in APPROACHES)
    taken = {}          # approach -> (edge, gap)
    overflow = []

    # sort by fit to the best approach first, so that better fits take their approach
    def best_gap(item):
        _, b = item
        return min(_angular_gap(b, _APPROACH_BEARING[a]) for a in APPROACHES)

    for edge, b in sorted(scored, key=best_gap):
        order = sorted(APPROACHES, key=lambda a: _angular_gap(b, _APPROACH_BEARING[a]))
        placed = False
        for a in order:
            if a not in taken:
                taken[a] = (edge, _angular_gap(b, _APPROACH_BEARING[a]))
                approach_to_edges[a].append(edge)
                placed = True
                break
        if not placed:
            overflow.append((edge, b))

    # overflow legs are merged into the approach with the closest angle (no traffic is dropped)
    for edge, b in overflow:
        a = min(APPROACHES, key=lambda x: _angular_gap(b, _APPROACH_BEARING[x]))
        approach_to_edges[a].append(edge)

    diag = {
        "num_legs": len(incoming_edges),
        "num_overflow": len(overflow),
        "missing_approaches": [a for a in APPROACHES if not approach_to_edges[a]],
        "approach_bearings": {
            a: round(taken[a][1], 2) for a in taken
        },
    }
    return approach_to_edges, [e for e, _ in overflow], diag


# ==========================================================================
# intersection topology
# ==========================================================================
class IntersectionTopology:
    """
    Topology projection of one signalised intersection.

    Exposes:
      slot_lanes[i]      : SUMO lane ids of canonical slot i (several or none)
      slot_mask[i]       : 1 = slot exists at this intersection, 0 = missing (masked)
      exit_slot_lanes[i] : downstream exit lane ids of slot i (for the pressure)
      green_phases       : indices of the green phases in the SUMO tlLogic (action space)
      phase_slot_matrix  : [num_green_phases x 12] 0/1 matrix of the slots served by each phase
      action_mask        : 0/1 vector of length MAX_PHASES for different phase counts
    """

    def __init__(self, tls_id, node, net, max_phases=4):
        self.tls_id = tls_id
        self.node_id = node.getID()
        self.net = net
        self.max_phases = max_phases

        self.slot_lanes = [[] for _ in range(NUM_SLOTS)]
        self.exit_slot_lanes = [[] for _ in range(NUM_SLOTS)]
        self.slot_mask = [0] * NUM_SLOTS
        self.lane_to_slot = {}

        self.entering_lanes = []
        self.exiting_lanes = []

        self._build_movements(node)
        self._build_phases()

    # ------------------------------------------------------------------
    def _build_movements(self, node):
        incoming = [e for e in node.getIncoming() if e.getFunction() != "internal"]
        outgoing = [e for e in node.getOutgoing() if e.getFunction() != "internal"]

        approach_to_edges, overflow, diag = assign_approaches(incoming)
        self.diag = diag
        self.diag["node_type"] = node.getType()

        for approach, edges in approach_to_edges.items():
            for edge in edges:
                for lane in edge.getLanes():
                    links = lane.getOutgoing()
                    if not links:
                        continue
                    # a lane may serve through + right; each connection is placed separately,
                    # so the lane appears in several slots (shared lanes as in CityFlow).
                    seen_turns = set()
                    for conn in links:
                        raw_dir = conn.getDirection()
                        turn = _DIR_TO_TURN.get(raw_dir, "T")
                        if turn in seen_turns:
                            continue
                        seen_turns.add(turn)
                        idx = slot_index(approach, turn)
                        if lane.getID() not in self.slot_lanes[idx]:
                            self.slot_lanes[idx].append(lane.getID())
                            self.slot_mask[idx] = 1
                        self.lane_to_slot.setdefault(lane.getID(), []).append(idx)
                        to_lane = conn.getToLane().getID()
                        if to_lane not in self.exit_slot_lanes[idx]:
                            self.exit_slot_lanes[idx].append(to_lane)

        self.entering_lanes = sorted({l for e in incoming for l in
                                      [ln.getID() for ln in e.getLanes()]})
        self.exiting_lanes = sorted({l for e in outgoing for l in
                                     [ln.getID() for ln in e.getLanes()]})

    # ------------------------------------------------------------------
    def _build_phases(self):
        """
        Extract the green phases from the SUMO tlLogic and build the phase -> slot matrix.
        Yellow/all-red phases (no G/g, or <= 5 s and all y) are not actions.
        """
        tls = self.net.getTLS(self.tls_id)
        # controlled links: index -> [(inLane, outLane, viaLane), ...]
        conns = tls.getConnections()          # [(inLane, outLane, linkIndex), ...]
        link_to_slots = defaultdict(set)
        for in_lane, _out_lane, link_index in conns:
            in_id = in_lane.getID()
            for s in self.lane_to_slot.get(in_id, []):
                link_to_slots[link_index].add(s)

        programs = tls.getPrograms()
        prog = programs.get("0") or list(programs.values())[0]

        self.green_phases = []
        rows = []
        for p_idx, phase in enumerate(prog.getPhases()):
            state = phase.state
            # Yellow must be tested for 'y' before testing for 'G': right turns stay 'G'
            # during yellow (as lightphases[0] in CityFlow), and testing 'G' first would
            # turn yellow phases into actions (green0 / yellow0 / green1 / yellow1),
            # a silent error that the training curves would not reveal.
            if "y" in state:
                continue                       # transition phase, not an action
            if not any(c in "Gg" for c in state):
                continue                       # all red
            row = [0] * NUM_SLOTS
            for link_index, ch in enumerate(state):
                if ch in "Gg":
                    for s in link_to_slots.get(link_index, ()):
                        row[s] = 1
            if not any(row):
                continue                       # serves only movements outside the canonical slots
            self.green_phases.append(p_idx)
            rows.append(row)

        self.phase_slot_matrix = rows
        self.num_actions = len(self.green_phases)

        # Self-check: a wrong phase detection (e.g. yellow taken as green) changes the
        # action space silently. Two checks:
        #   1) consecutive green phases must not serve identical movement sets
        #   2) no phase may serve only right turns (almost surely a misread yellow)
        for k in range(1, len(rows)):
            if rows[k] == rows[k - 1]:
                self.diag.setdefault("warnings", []).append(
                    "phases {0} and {1} serve identical movements; a transition phase "
                    "was probably taken as an action".format(k - 1, k))
        right_only = [k for k, r in enumerate(rows)
                      if r and all(self.slot_lanes[i] == [] or
                                   (i % len(TURNS)) == TURNS.index("R")
                                   for i, v in enumerate(r) if v)]
        if right_only:
            self.diag.setdefault("warnings", []).append(
                "phases {0} serve only right turns; almost surely yellow phases read as green".format(
                    right_only))

        # different phase counts -> pad to max_phases and build the action mask
        self.action_mask = [1] * min(self.num_actions, self.max_phases) + \
                           [0] * max(0, self.max_phases - self.num_actions)
        if self.num_actions > self.max_phases:
            self.diag["phases_truncated"] = self.num_actions - self.max_phases

        self.diag["num_green_phases"] = self.num_actions
        self.diag["num_masked_actions"] = self.action_mask.count(0)

    # ------------------------------------------------------------------
    def summary(self):
        return {
            "tls_id": self.tls_id,
            "node_id": self.node_id,
            "num_actions": self.num_actions,
            "valid_slots": sum(self.slot_mask),
            "slot_mask": self.slot_mask,
            "action_mask": self.action_mask,
            **self.diag,
        }


# ==========================================================================
# network level
# ==========================================================================
class NetworkTopology:
    """Topology projection of a whole SUMO network, with top-K adjacency (as in the CityFlow version)."""

    def __init__(self, net_file, top_k=5, max_phases=4):
        self.net_file = net_file
        self.net = sumolib.net.readNet(net_file, withPrograms=True)
        self.top_k = top_k
        self.max_phases = max_phases

        self.inter_ids = []
        self.topologies = OrderedDict()

        # Sorted for a stable order: env and prepare_sumo_conf each build a NetworkTopology,
        # and the intersection order must agree, otherwise ACTION_MASK would be misaligned.
        for tls in sorted(self.net.getTrafficLights(), key=lambda t: t.getID()):
            tls_id = tls.getID()
            node = self._node_for_tls(tls_id)
            if node is None:
                continue
            incoming = [e for e in node.getIncoming() if e.getFunction() != "internal"]
            outgoing = [e for e in node.getOutgoing() if e.getFunction() != "internal"]
            # as _is_node_virtual: nodes with fewer than 3 legs are virtual (no agent)
            if len(set([e.getID() for e in incoming + outgoing])) < 3:
                continue
            topo = IntersectionTopology(tls_id, node, self.net, max_phases=max_phases)
            if topo.num_actions < 2:
                continue                       # a single-phase intersection cannot be controlled
            self.topologies[tls_id] = topo
            self.inter_ids.append(tls_id)

        self._build_adjacency()

    # ------------------------------------------------------------------
    def _node_for_tls(self, tls_id):
        # tls_id usually equals the node id; netconvert may produce a "GS_" prefix or a joined id
        for cand in (tls_id, tls_id.replace("GS_", "", 1)):
            try:
                return self.net.getNode(cand)
            except KeyError:
                continue
        # joined tls: use the first node it controls
        try:
            tls = self.net.getTLS(tls_id)
            nodes = {c[0].getEdge().getToNode().getID() for c in tls.getConnections()}
            if nodes:
                return self.net.getNode(sorted(nodes)[0])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _build_adjacency(self):
        """
        top-K adjacency: the K nearest intersections by Euclidean distance between centres
        (including itself, at position 0). Same definition as
        utils/cityflow_env.py::_adjacency_extraction, so the spatial attention inputs are comparable.
        """
        self.inter_id_to_index = {i: k for k, i in enumerate(self.inter_ids)}
        coords = {}
        for tls_id in self.inter_ids:
            node = self._node_for_tls(tls_id)
            coords[tls_id] = node.getCoord()

        n = len(self.inter_ids)
        self.adjacency = {}
        for i in self.inter_ids:
            xi, yi = coords[i]
            dists = []
            for j in self.inter_ids:
                xj, yj = coords[j]
                dists.append((math.hypot(xi - xj, yi - yj), self.inter_id_to_index[j]))
            dists.sort(key=lambda t: t[0])
            row = [idx for _, idx in dists[:min(self.top_k, n)]]
            # itself first (distance 0; already first after sorting, made explicit here)
            self_idx = self.inter_id_to_index[i]
            row = [self_idx] + [x for x in row if x != self_idx]
            row = row[:self.top_k]
            while len(row) < self.top_k:       # pad with itself if there are fewer than K intersections
                row.append(self_idx)
            self.adjacency[i] = row

    # ------------------------------------------------------------------
    def report(self):
        """Print/export topology diagnostics (heterogeneity of non-grid networks)."""
        rows = [self.topologies[i].summary() for i in self.inter_ids]
        stats = {
            "net_file": os.path.basename(self.net_file),
            "num_agents": len(rows),
            "phase_count_hist": _hist([r["num_actions"] for r in rows]),
            "leg_count_hist": _hist([r["num_legs"] for r in rows]),
            "valid_slot_hist": _hist([r["valid_slots"] for r in rows]),
            "num_with_overflow": sum(1 for r in rows if r["num_overflow"] > 0),
            "num_truncated_phases": sum(1 for r in rows if "phases_truncated" in r),
            "warnings": [w for r in rows for w in r.get("warnings", [])][:10],
            "num_intersections_needing_mask":
                sum(1 for r in rows if r.get("num_masked_actions", 0) > 0),
            "masked_action_hist": _hist(
                [r.get("num_masked_actions", 0) for r in rows]),
        }
        return stats, rows


def _hist(values):
    h = defaultdict(int)
    for v in values:
        h[v] += 1
    return dict(sorted(h.items()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check the topological heterogeneity of a SUMO network")
    ap.add_argument("net_file")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_phases", type=int, default=4)
    ap.add_argument("--dump", type=str, default=None, help="write per-intersection diagnostics to json")
    a = ap.parse_args()

    topo = NetworkTopology(a.net_file, top_k=a.top_k, max_phases=a.max_phases)
    stats, rows = topo.report()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump({"stats": stats, "intersections": rows}, f,
                      indent=2, ensure_ascii=False)
        print(f"\nper-intersection diagnostics written to {a.dump}")
