"""
cityflow2sumo.py
================
High-fidelity CityFlow -> SUMO converter (replaces the c2s path of LibSignal's converter.py).

Issues of the original converter fixed here (each affects the cross-simulator
comparison or the safety measures):

  #  original                                   this implementation
  -- ------------------------------------------ --------------------------------------
  1  <vehicle> never sets a type, so vTypes     every vehicle is bound to its vType
     have no effect
  2  no sigma: SUMO's default driver           sigma=0 (configurable), matching CityFlow's
     imperfection 0.5                           deterministic car following
  3  no speedFactor: SUMO draws a random       speedFactor=1.0, speedDev=0
     desired-speed factor normc(1,0.1,0.2,2)
  4  edge speed fixed at 11.111, ignoring      maxSpeed read per lane, set per lane if needed
     per-lane maxSpeed
  5  intermediate shape points dropped         full shape kept, link lengths match CityFlow
  6  lane widths lost                          width written per lane
  7  one fixed set of vehicle parameters       read from flow.json, de-duplicated into vTypes
  8  each flow entry treated as one vehicle,    expanded by startTime/endTime/interval
     ignoring interval (vehicles silently lost)
  9  link indices of tlLogic guessed by hand;  two netconvert passes: build the network to
     a wrong order breaks every phase           get the real link indices, then build the tll
  10 netconvert shifts the coordinate origin   --offset.disable-normalization,
                                                 coordinates identical to CityFlow

Usage (the scenario directory and --name must equal the CityFlow traffic file
without .json, which is how run_d2marl.py -sumo finds the scenario):
    python -m d2marl_sumo.cityflow2sumo \
        --roadnet data/Hangzhou/4_4/roadnet_4_4.json \
        --flow    data/Hangzhou/4_4/anon_4_4_hangzhou_real.json \
        --outdir  sumo_scenarios/anon_4_4_hangzhou_real \
        --name    anon_4_4_hangzhou_real
"""

import os
import sys
import json
import math
import shutil
import argparse
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)


# ==========================================================================
# helpers
# ==========================================================================
def _indent(elem, level=0):
    """Pretty-print the xml for manual inspection."""
    pad = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "    "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def _write(tree_root, path):
    _indent(tree_root)
    ET.ElementTree(tree_root).write(path, encoding="UTF-8", xml_declaration=True)


def _run(cmd, desc):
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode("utf-8", errors="ignore")
    if r.returncode != 0:
        print(out)
        raise RuntimeError(f"{desc} failed (return code {r.returncode})")
    warn = [l for l in out.splitlines() if "Warning" in l or "Error" in l]
    if warn:
        print(f"  [{desc}] {len(warn)} warnings, first 5:")
        for l in warn[:5]:
            print("    " + l)
    return out


# ==========================================================================
# 1. network: nod / edg / con
# ==========================================================================
def emit_network_inputs(roadnet, outdir, name, right_on_red=True):
    """Generate nod.xml / edg.xml / con.xml from the CityFlow roadnet.json."""
    inters = roadnet["intersections"]
    roads = roadnet["roads"]
    road_by_id = {r["id"]: r for r in roads}

    # ---- nod.xml ----
    nodes = ET.Element("nodes")
    nodes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    nodes.set("xsi:noNamespaceSchemaLocation",
              "http://sumo.dlr.de/xsd/nodes_file.xsd")
    n_signal = 0
    for it in inters:
        n = ET.SubElement(nodes, "node")
        n.set("id", it["id"])
        n.set("x", repr(float(it["point"]["x"])))
        n.set("y", repr(float(it["point"]["y"])))
        if it.get("virtual", False):
            n.set("type", "priority")
        else:
            n.set("type", "traffic_light_right_on_red" if right_on_red
                  else "traffic_light")
            n.set("tl", it["id"])
            n_signal += 1
    _write(nodes, os.path.join(outdir, name + ".nod.xml"))

    # ---- edg.xml ----
    edges = ET.Element("edges")
    edges.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    edges.set("xsi:noNamespaceSchemaLocation",
              "http://sumo.dlr.de/xsd/edges_file.xsd")
    for r in roads:
        e = ET.SubElement(edges, "edge")
        e.set("id", r["id"])
        e.set("from", r["startIntersection"])
        e.set("to", r["endIntersection"])
        lanes = r["lanes"]
        e.set("numLanes", str(len(lanes)))
        e.set("priority", "-1")

        # speed limit: per lane in CityFlow; written on the edge if all lanes agree, per lane otherwise
        speeds = [float(l.get("maxSpeed", 11.111)) for l in lanes]
        widths = [float(l.get("width", 3.2)) for l in lanes]
        e.set("speed", repr(speeds[0]))

        # keep intermediate shape points so that link lengths match CityFlow
        pts = r.get("points", [])
        if len(pts) > 2:
            e.set("shape", " ".join(
                "{0},{1}".format(p["x"], p["y"]) for p in pts))

        # CityFlow orders lanes inner->outer, SUMO outer->inner: reverse for per-lane attributes
        need_per_lane = (len(set(speeds)) > 1) or (len(set(widths)) > 1)
        if need_per_lane:
            n_lane = len(lanes)
            for cf_idx in range(n_lane):
                sumo_idx = n_lane - 1 - cf_idx
                ln = ET.SubElement(e, "lane")
                ln.set("index", str(sumo_idx))
                ln.set("speed", repr(speeds[cf_idx]))
                ln.set("width", repr(widths[cf_idx]))
        else:
            e.set("width", repr(widths[0]))
    _write(edges, os.path.join(outdir, name + ".edg.xml"))

    # ---- con.xml ----
    cons = ET.Element("connections")
    cons.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    cons.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/connections_file.xsd")

    # roadlink_key -> [(fromEdge, toEdge, fromLane, toLane), ...]
    # used later to look up CityFlow's availableRoadLinks when building the tll
    roadlink_conns = {}
    n_conn = 0
    for it in inters:
        for rl_idx, rl in enumerate(it["roadLinks"]):
            se, ee = rl["startRoad"], rl["endRoad"]
            if se not in road_by_id or ee not in road_by_id:
                continue
            n_from = len(road_by_id[se]["lanes"])
            n_to = len(road_by_id[ee]["lanes"])
            tuples = []
            for ll in rl["laneLinks"]:
                f = n_from - 1 - int(ll["startLaneIndex"])
                t = n_to - 1 - int(ll["endLaneIndex"])
                if not (0 <= f < n_from and 0 <= t < n_to):
                    continue
                c = ET.SubElement(cons, "connection")
                c.set("from", se)
                c.set("to", ee)
                c.set("fromLane", str(f))
                c.set("toLane", str(t))
                tuples.append((se, ee, f, t))
                n_conn += 1
            roadlink_conns[(it["id"], rl_idx)] = {
                "tuples": tuples, "type": rl.get("type", "go_straight")}
    _write(cons, os.path.join(outdir, name + ".con.xml"))

    return {"num_nodes": len(inters), "num_signal_nodes": n_signal,
            "num_edges": len(roads), "num_connections": n_conn,
            "roadlink_conns": roadlink_conns}


# ==========================================================================
# 2. two netconvert passes + tlLogic
# ==========================================================================
NETCONVERT_COMMON = [
    "--offset.disable-normalization", "true",   # coordinates identical to CityFlow
    "--no-turnarounds", "true",
    "--check-lane-foes.all", "true",            # complete conflict points (needed for SSM)
    "--junctions.corner-detail", "5",
    "--tls.default-type", "static",
    "--no-internal-links", "false",             # keep internal lanes (needed for crossing conflicts)
    "--lefthand", "false",
]


def build_network(outdir, name, meta, cityflow_roadnet,
                  yellow_time=5, right_on_red=True,
                  num_phases=4, phase_offset=1):
    """
    Two netconvert passes:
      pass 1 builds the network -> sumolib reads the real link index of every signal
      pass 2 rebuilds the network with a tll.xml generated from these indices
    so the link order is never guessed.
    """
    import sumolib

    nod = os.path.join(outdir, name + ".nod.xml")
    edg = os.path.join(outdir, name + ".edg.xml")
    con = os.path.join(outdir, name + ".con.xml")
    tll = os.path.join(outdir, name + ".tll.xml")
    net = os.path.join(outdir, name + ".net.xml")
    net_tmp = os.path.join(outdir, "_pass1.net.xml")

    print("\n[2/5] netconvert pass 1 (real link indices)")
    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--connection-files", con, "--output-file", net_tmp]
         + NETCONVERT_COMMON, "netconvert pass1")

    print("\n[3/5] generating tll.xml from the real link indices")
    net1 = sumolib.net.readNet(net_tmp, withPrograms=True)
    n_phase_total = _emit_tll(net1, cityflow_roadnet, meta, tll,
                              yellow_time=yellow_time,
                              right_on_red=right_on_red,
                              num_phases=num_phases,
                              phase_offset=phase_offset)

    print("\n[4/5] netconvert pass 2 (with signal plans)")
    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--connection-files", con, "--tllogic-files", tll,
          "--output-file", net] + NETCONVERT_COMMON, "netconvert pass2")

    os.remove(net_tmp)
    return net, n_phase_total


def _emit_tll(net, roadnet, meta, out_path, yellow_time=5, right_on_red=True,
              num_phases=4, phase_offset=1):
    """
    Translate CityFlow lightphases into SUMO tlLogic.

    The (inLane, outLane, linkIndex) triples of each tls are read from the built
    network and matched to CityFlow roadLinks by (fromEdge, toEdge, fromLane, toLane),
    which gives the roadLink of every link index.
    """
    roadlink_conns = meta["roadlink_conns"]
    inter_by_id = {it["id"]: it for it in roadnet["intersections"]}

    root = ET.Element("tlLogics")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/tllogic_file.xsd")

    total_phases = 0
    for tls in net.getTrafficLights():
        tls_id = tls.getID()
        inter = inter_by_id.get(tls_id)
        if inter is None:
            # netconvert may rename joined junctions; skip with a warning
            print(f"    [warning] tls {tls_id} has no matching intersection in the CityFlow roadnet; "
                  f"keeping the default plan of netconvert")
            continue

        # quadruple -> link index
        tuple_to_link = {}
        n_links = 0
        for in_lane, out_lane, link_index in tls.getConnections():
            key = (in_lane.getEdge().getID(), out_lane.getEdge().getID(),
                   in_lane.getIndex(), out_lane.getIndex())
            tuple_to_link[key] = link_index
            n_links = max(n_links, link_index + 1)

        # roadLink index -> set of link indices it occupies
        rl_to_links = defaultdict(set)
        rl_type = {}
        for rl_idx in range(len(inter["roadLinks"])):
            info = roadlink_conns.get((tls_id, rl_idx))
            if not info:
                continue
            rl_type[rl_idx] = info["type"]
            for t in info["tuples"]:
                if t in tuple_to_link:
                    rl_to_links[rl_idx].add(tuple_to_link[t])

        right_links = set()
        if right_on_red:
            for rl_idx, typ in rl_type.items():
                if typ == "turn_right":
                    right_links |= rl_to_links[rl_idx]

        # Select the green phases actually used by the agent.
        #
        # The CityFlow benchmark roadnets usually have 9 lightphases (index 0 is
        # yellow, 1-8 are the possible phases), but in utils/cityflow_env.py
        #     next_phase_to_set_index = action + 1
        # maps action a in {0..NUM_PHASES-1} to lightphases[1..NUM_PHASES] only.
        # Exporting all 8 would give SUMO an action space of 8 instead of CityFlow's 4.
        # Hence exactly num_phases phases are taken, starting at phase_offset.
        all_lp = inter.get("trafficLight", {}).get("lightphases", [])
        candidate = all_lp[phase_offset: phase_offset + num_phases]
        green_phases = [lp for lp in candidate if (lp.get("availableRoadLinks") or [])]
        if len(green_phases) != num_phases:
            print("    [warning] {0}: expected {1} green phases, found {2} "
                  "({3} lightphases in total)".format(
                      tls_id, num_phases, len(green_phases), len(all_lp)))

        if not green_phases:
            print(f"    [warning] {tls_id} has no valid green phase, skipped")
            continue

        logic = ET.SubElement(root, "tlLogic")
        logic.set("id", tls_id)
        logic.set("type", "static")
        logic.set("programID", "0")
        logic.set("offset", "0")

        states = []
        for lp in green_phases:
            s = ["r"] * n_links
            for li in right_links:
                s[li] = "s"          # right turns permitted (stop-then-go)
            for rl_idx in lp["availableRoadLinks"]:
                for li in rl_to_links.get(rl_idx, ()):
                    s[li] = "G"
            states.append(s)

        for i, s in enumerate(states):
            ph = ET.SubElement(logic, "phase")
            ph.set("duration", str(int(float(green_phases[i].get("time", 30)))))
            ph.set("state", "".join(s))
            total_phases += 1
            # yellow: links that are G in this phase but not in the next turn yellow
            nxt = states[(i + 1) % len(states)]
            y = ["y" if (a == "G" and b != "G") else a
                 for a, b in zip(s, nxt)]
            phy = ET.SubElement(logic, "phase")
            phy.set("duration", str(int(yellow_time)))
            phy.set("state", "".join(y))

    _write(root, out_path)
    return total_phases


# ==========================================================================
# 3. demand
# ==========================================================================
def emit_routes(flow, outdir, name, sigma=0.0, depart_speed="max",
                depart_lane="best", horizon=3600):
    """
    CityFlow flow.json -> SUMO rou.xml.

    A CityFlow flow entry means one vehicle every interval seconds from startTime
    to endTime. The original converter emitted only one vehicle at startTime and
    silently lost the rest when endTime > startTime; here entries are fully expanded.
    """
    vtypes = OrderedDict()
    vehicles = []

    def vtype_key(v):
        return (
            round(float(v.get("length", 5.0)), 3),
            round(float(v.get("width", 2.0)), 3),
            round(float(v.get("maxPosAcc", 2.0)), 3),
            round(float(v.get("maxNegAcc", 4.5)), 3),
            round(float(v.get("minGap", 2.5)), 3),
            round(float(v.get("maxSpeed", 11.111)), 3),
            round(float(v.get("headwayTime", 1.5)), 3),
        )

    n_expanded = 0
    for entry in flow:
        v = entry.get("vehicle", {})
        key = vtype_key(v)
        if key not in vtypes:
            vtypes[key] = "vt{0}".format(len(vtypes))
        vt = vtypes[key]

        route = entry.get("route", [])
        if len(route) < 2:
            continue

        t0 = float(entry.get("startTime", 0))
        t1 = float(entry.get("endTime", t0))
        iv = float(entry.get("interval", 0) or 0)

        if t1 <= t0 or iv <= 0:
            departs = [t0]
        else:
            departs = []
            t = t0
            while t <= t1 + 1e-9:
                departs.append(t)
                t += iv
            n_expanded += len(departs) - 1

        for d in departs:
            if d > horizon:
                continue
            vehicles.append((d, vt, route))

    vehicles.sort(key=lambda x: (x[0], x[2][0]))

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    for key, vt_id in vtypes.items():
        length, width, accel, decel, min_gap, max_speed, tau = key
        t = ET.SubElement(root, "vType")
        t.set("id", vt_id)
        t.set("vClass", "passenger")
        t.set("carFollowModel", "Krauss")
        t.set("length", repr(length))
        t.set("width", repr(width))
        t.set("accel", repr(accel))
        t.set("decel", repr(decel))
        t.set("emergencyDecel", repr(max(9.0, decel)))
        t.set("minGap", repr(min_gap))
        t.set("maxSpeed", repr(max_speed))
        t.set("tau", repr(tau))
        # -- deterministic behaviour as in CityFlow (missing in the original converter, affects TTC/PET) --
        t.set("sigma", repr(sigma))          # driver imperfection
        t.set("speedFactor", "1.0")          # no randomised desired speed
        t.set("speedDev", "0")
        t.set("departLane", depart_lane)
        t.set("departSpeed", depart_speed)

    for i, (d, vt, route) in enumerate(vehicles):
        veh = ET.SubElement(root, "vehicle")
        veh.set("id", str(i))
        veh.set("type", vt)                  # missing in the original converter
        veh.set("depart", "{0:.2f}".format(d))
        veh.set("departLane", depart_lane)
        veh.set("departSpeed", depart_speed)
        r = ET.SubElement(veh, "route")
        r.set("edges", " ".join(route))

    out = os.path.join(outdir, name + ".rou.xml")
    _write(root, out)
    return out, {"num_vtypes": len(vtypes), "num_vehicles": len(vehicles),
                 "num_from_interval_expansion": n_expanded,
                 "num_flow_entries": len(flow)}


def repair_routes(rou_file, net_file, outdir, name):
    """
    Check and repair the routes with duarouter. If connections are missing after
    conversion a CityFlow route is disconnected in SUMO; the number of repaired
    routes is a fidelity indicator that should be reported.
    """
    out = os.path.join(outdir, name + ".repaired.rou.xml")
    log = _run(["duarouter", "-n", net_file, "-r", rou_file, "-o", out,
                "--repair", "true", "--ignore-errors", "true",
                "--no-step-log", "true", "--xml-validation", "never"],
               "duarouter")
    n_repaired = sum(1 for l in log.splitlines()
                     if "repair" in l.lower() or "disconnected" in l.lower())
    return out, n_repaired


# ==========================================================================
# 4. sumocfg
# ==========================================================================
def emit_sumocfg(outdir, name, net_file, rou_file, horizon=3600, step=1.0):
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file").set("value", os.path.basename(net_file))
    ET.SubElement(inp, "route-files").set("value", os.path.basename(rou_file))

    t = ET.SubElement(root, "time")
    ET.SubElement(t, "begin").set("value", "0")
    ET.SubElement(t, "end").set("value", str(int(horizon)))
    ET.SubElement(t, "step-length").set("value", repr(step))

    p = ET.SubElement(root, "processing")
    # teleporting disabled: SUMO would otherwise remove extremely delayed vehicles and bias Mwt / C_d down
    ET.SubElement(p, "time-to-teleport").set("value", "-1")
    ET.SubElement(p, "waiting-time-memory").set("value", "10000")
    ET.SubElement(p, "lanechange.duration").set("value", "0")
    ET.SubElement(p, "collision.action").set("value", "warn")
    ET.SubElement(p, "collision.check-junctions").set("value", "true")

    r = ET.SubElement(root, "report")
    ET.SubElement(r, "no-step-log").set("value", "true")
    ET.SubElement(r, "no-warnings").set("value", "true")

    out = os.path.join(outdir, name + ".sumocfg")
    _write(root, out)
    return out


# ==========================================================================
# 5. main
# ==========================================================================
def convert(roadnet_path, flow_path, outdir, name,
            sigma=0.0, yellow_time=5, horizon=3600,
            depart_speed="max", depart_lane="best", right_on_red=True,
            num_phases=4, phase_offset=1):
    os.makedirs(outdir, exist_ok=True)

    print("[1/5] reading the CityFlow files, generating nod/edg/con")
    with open(roadnet_path) as f:
        roadnet = json.load(f)
    with open(flow_path) as f:
        flow = json.load(f)

    meta = emit_network_inputs(roadnet, outdir, name, right_on_red=right_on_red)
    print("  intersections {0} (signalised {1})  edges {2}  connections {3}".format(
        meta["num_nodes"], meta["num_signal_nodes"],
        meta["num_edges"], meta["num_connections"]))

    net_file, n_phases = build_network(outdir, name, meta, roadnet,
                                       yellow_time=yellow_time,
                                       right_on_red=right_on_red,
                                       num_phases=num_phases,
                                       phase_offset=phase_offset)

    print("\n[5/5] generating the demand")
    rou_file, flow_stats = emit_routes(flow, outdir, name, sigma=sigma,
                                       depart_speed=depart_speed,
                                       depart_lane=depart_lane,
                                       horizon=horizon)
    print("  {0} vTypes, {1} vehicles ({2} flow entries, {3} added by interval expansion)".format(
        flow_stats["num_vtypes"], flow_stats["num_vehicles"],
        flow_stats["num_flow_entries"], flow_stats["num_from_interval_expansion"]))

    rou_fixed, n_repaired = repair_routes(rou_file, net_file, outdir, name)
    if n_repaired:
        print("  [note] duarouter reported {0} messages about route connectivity; "
              "check {1}".format(n_repaired, os.path.basename(rou_fixed)))

    cfg = emit_sumocfg(outdir, name, net_file, rou_fixed,
                       horizon=horizon)

    report = {
        "name": name,
        "source_roadnet": roadnet_path,
        "source_flow": flow_path,
        "net_file": net_file,
        "rou_file": rou_fixed,
        "sumocfg": cfg,
        "green_phases_total": n_phases,
        "num_phases_per_tls": num_phases,
        "phase_offset": phase_offset,
        "routes_repaired_msgs": n_repaired,
        "sigma": sigma,
        "depart_speed": depart_speed,
        **{k: v for k, v in meta.items() if k != "roadlink_conns"},
        **flow_stats,
    }
    with open(os.path.join(outdir, name + ".convert_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nconversion finished -> {0}".format(cfg))
    print("next: check the conversion with python -m d2marl_sumo.validate_conversion; "
          "the free-flow travel time difference should be < 5%")
    return report


def main():
    ap = argparse.ArgumentParser(description="High-fidelity CityFlow -> SUMO conversion")
    ap.add_argument("--roadnet", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="driver imperfection; 0 = as CityFlow (default), "
                         "0.5 = SUMO default (for sensitivity analysis)")
    ap.add_argument("--yellow", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--depart-speed", default="max")
    ap.add_argument("--depart-lane", default="best")
    ap.add_argument("--no-right-on-red", action="store_true")
    ap.add_argument("--num-phases", type=int, default=4,
                    help="size of the action space. CityFlow roadnets usually define 8 phases, "
                         "but PHASE in config.py has 4; the default 4 matches it")
    ap.add_argument("--phase-offset", type=int, default=1,
                    help="index of the first lightphase to use; default 1 skips the yellow "
                         "phase at index 0 (action+1 in cityflow_env.py)")
    a = ap.parse_args()

    convert(a.roadnet, a.flow, a.outdir, a.name,
            sigma=a.sigma, yellow_time=a.yellow, horizon=a.horizon,
            depart_speed=a.depart_speed, depart_lane=a.depart_lane,
            right_on_red=not a.no_right_on_red,
            num_phases=a.num_phases, phase_offset=a.phase_offset)


if __name__ == "__main__":
    main()
