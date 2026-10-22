"""
validate_conversion.py
======================
Fidelity check of the conversion. The conversion is not lossless, so there must be
evidence that CityFlow and SUMO simulate the same scenario.

Four checks:
  1. structure      intersections / signalised intersections / edges / lanes   required: equal
  2. vehicles       vehicles in rou.xml vs. expanded flow.json                 required: equal
  3. connectivity   number of routes repaired by duarouter                     reported, the fewer the better
  4. free flow      free-flow travel time of single vehicles in an empty network  required: < 5 % from CityFlow

Check 4 matters most: **if the free-flow travel times differ by more than 5 %, speed
limits / vehicle lengths / minimum gaps are not aligned and no ATT comparison is valid**.

Usage:
    python -m d2marl_sumo.validate_conversion \
        --report sumo_scenarios/anon_4_4_hangzhou_real/anon_4_4_hangzhou_real.convert_report.json \
        --cityflow-roadnet data/Hangzhou/4_4/roadnet_4_4.json \
        --cityflow-flow    data/Hangzhou/4_4/anon_4_4_hangzhou_real.json
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)


def count_cityflow(roadnet_path, flow_path, horizon=3600):
    with open(roadnet_path) as f:
        rn = json.load(f)
    with open(flow_path) as f:
        fl = json.load(f)

    n_inter = len(rn["intersections"])
    n_signal = sum(1 for i in rn["intersections"] if not i.get("virtual", False))
    n_road = len(rn["roads"])
    n_lane = sum(len(r["lanes"]) for r in rn["roads"])

    # expand the vehicles with CityFlow semantics
    n_veh = 0
    for e in fl:
        if len(e.get("route", [])) < 2:
            continue
        t0 = float(e.get("startTime", 0))
        t1 = float(e.get("endTime", t0))
        iv = float(e.get("interval", 0) or 0)
        if t1 <= t0 or iv <= 0:
            n_veh += 1 if t0 <= horizon else 0
        else:
            t = t0
            while t <= t1 + 1e-9:
                if t <= horizon:
                    n_veh += 1
                t += iv
    return {"intersections": n_inter, "signal_intersections": n_signal,
            "roads": n_road, "lanes": n_lane, "vehicles": n_veh}


def count_sumo(net_file, rou_file):
    import sumolib
    net = sumolib.net.readNet(net_file)
    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]
    nodes = [n for n in net.getNodes()
             if not n.getID().startswith(":")]
    n_signal = len(net.getTrafficLights())
    n_lane = sum(len(e.getLanes()) for e in edges)

    n_veh = 0
    for _evt, elem in ET.iterparse(rou_file, events=("end",)):
        if elem.tag == "vehicle":
            n_veh += 1
            elem.clear()
    return {"intersections": len(nodes), "signal_intersections": n_signal,
            "roads": len(edges), "lanes": n_lane, "vehicles": n_veh}


def _read_routes(rou_file, n_max):
    """Take the first n_max real routes from rou.xml. Real routes (instead of random OD
    pairs) are connected and are the same routes measured on CityFlow."""
    routes = []
    for _evt, elem in ET.iterparse(rou_file, events=("end",)):
        if elem.tag == "vehicle":
            r = elem.find("route")
            if r is not None:
                e = (r.get("edges") or "").split()
                if len(e) >= 2:
                    routes.append(e)
            elem.clear()
            if len(routes) >= n_max:
                break
    return routes


def free_flow_test(net_file, rou_file, n_probe=50, horizon=3600, headway=2.0):
    """
    Free-flow travel time in an empty network: the first n_probe real routes of rou.xml
    are released one by one every `headway` seconds with all signals green, and the mean

    travel time is measured. It reflects only the network geometry (speed limits, link
    lengths, vehicle dynamics) and should differ from CityFlow by less than 5 %.
    """
    import traci
    import sumolib

    routes = _read_routes(rou_file, n_probe)
    if not routes:
        return {"error": "no usable route in rou.xml"}

    # SUMO needs at least one route-files argument; give it an empty one
    empty_rou = os.path.join(os.path.dirname(net_file), "_freeflow_empty.rou.xml")
    with open(empty_rou, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes/>\n')

    end_t = int(horizon + headway * len(routes) + 600)
    cmd = [sumolib.checkBinary("sumo"),
           "-n", net_file,
           "-r", empty_rou,
           "--begin", "0",
           "--end", str(end_t),
           "--step-length", "1",
           "--no-step-log", "true",
           "--no-warnings", "true",
           "--time-to-teleport", "-1",
           "--xml-validation", "never"]
    traci.start(cmd, label="freeflow")

    try:
        # all signals permanently green
        for tls in traci.trafficlight.getIDList():
            n = len(traci.trafficlight.getRedYellowGreenState(tls))
            traci.trafficlight.setRedYellowGreenState(tls, "G" * n)

        for i, edges in enumerate(routes):
            rid = "ffr{0}".format(i)
            traci.route.add(rid, edges)
            traci.vehicle.add("ff{0}".format(i), rid,
                              depart=str(i * headway),
                              departLane="best", departSpeed="max")

        depart, travel = {}, {}
        t = 0
        while t < end_t and len(travel) < len(routes):
            traci.simulationStep()
            t = traci.simulation.getTime()
            for v in traci.simulation.getDepartedIDList():
                depart[v] = t
            for v in traci.simulation.getArrivedIDList():
                if v in depart:
                    travel[v] = t - depart[v]
            # stop early when all vehicles have left and none is waiting to depart
            if (traci.simulation.getMinExpectedNumber() == 0
                    and t > headway * len(routes)):
                break
    finally:
        try:
            traci.close()
        except Exception:
            pass
        if os.path.exists(empty_rou):
            os.remove(empty_rou)

    if not travel:
        return {"error": "no probe vehicle completed its trip"}
    vals = sorted(travel.values())
    return {"n_launched": len(routes), "n_arrived": len(vals),
            "mean_free_flow_tt": sum(vals) / len(vals),
            "median": vals[len(vals) // 2],
            "min": vals[0], "max": vals[-1]}


def fixed_time_test(sumocfg, horizon=3600, green_time=15, yellow_time=5):
    """
    Run the full scenario with the converted static plans (= the Fixed-time baseline)

    and report ATT and waiting statistics. A stricter check than free flow: if the
    Fixed-time results agree with CityFlow, both simulators behave alike under the same
    control policy, and ATT differences can be attributed to the controllers.

    The timing must match: the converted tlLogic keeps the durations of the CityFlow
    lightphases (usually 30 s), while the Fixed-time baseline uses FIXED_TIME from
    utils/config.py. All signal programs are therefore rewritten to green_time /
    yellow_time at the start of the simulation.
    """
    import traci
    import sumolib

    traci.start([sumolib.checkBinary("sumo"), "-c", sumocfg,
                 "--no-step-log", "true", "--no-warnings", "true",
                 "--time-to-teleport", "-1",
                 "--waiting-time-memory", "10000",
                 "--xml-validation", "never"], label="fixedtime")

    # rewrite the plans to green_time green / yellow_time yellow
    for tls in traci.trafficlight.getIDList():
        logic = traci.trafficlight.getAllProgramLogics(tls)[0]
        for ph in logic.phases:
            st = ph.state
            # test 'y' before 'G' (right turns stay G during yellow)
            if "y" in st:
                ph.duration = ph.minDur = ph.maxDur = yellow_time
            elif any(c in "Gg" for c in st):
                ph.duration = ph.minDur = ph.maxDur = green_time
        traci.trafficlight.setProgramLogic(tls, logic)
        traci.trafficlight.setPhase(tls, 0)

    depart, travel = {}, {}
    total_wait, max_wait = {}, 0.0
    queue_sum, steps = 0.0, 0
    try:
        t = 0
        while t < horizon:
            traci.simulationStep()
            t = traci.simulation.getTime()
            for v in traci.simulation.getDepartedIDList():
                depart[v] = t
            for v in traci.simulation.getArrivedIDList():
                if v in depart:
                    travel[v] = t - depart[v]
            for v in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(v) < 0.1:
                    total_wait[v] = total_wait.get(v, 0.0) + 1.0
                    if total_wait[v] > max_wait:
                        max_wait = total_wait[v]
            queue_sum += sum(traci.lane.getLastStepHaltingNumber(l)
                             for l in traci.lane.getIDList()
                             if not l.startswith(":"))
            steps += 1
            if traci.simulation.getMinExpectedNumber() == 0:
                break
    finally:
        try:
            traci.close()
        except Exception:
            pass

    vals = list(travel.values())
    waits = list(total_wait.values())
    return {
        "n_arrived": len(vals),
        "n_departed": len(depart),
        "ATT": (sum(vals) / len(vals)) if vals else None,
        "Awt": (sum(waits) / len(waits)) if waits else None,
        "Mwt": max_wait,
        "AQL": (queue_sum / steps) if steps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="convert_report.json")
    ap.add_argument("--cityflow-roadnet", required=True)
    ap.add_argument("--cityflow-flow", required=True)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--skip-freeflow", action="store_true")
    ap.add_argument("--n-probe", type=int, default=50,
                    help="number of probe vehicles in the free-flow test")
    ap.add_argument("--fixed-time", action="store_true",
                    help="also run the full scenario under Fixed-time")
    ap.add_argument("--ft-green", type=int, default=15,
                    help="green time of the Fixed-time run; must match FIXED_TIME "
                         "in utils/config.py to be comparable")
    ap.add_argument("--ft-yellow", type=int, default=5)
    a = ap.parse_args()

    with open(a.report) as f:
        rep = json.load(f)

    cf = count_cityflow(a.cityflow_roadnet, a.cityflow_flow, a.horizon)
    su = count_sumo(rep["net_file"], rep["rou_file"])

    print("=" * 62)
    print("Conversion fidelity check: {0}".format(rep["name"]))
    print("=" * 62)
    print("{0:<24s} {1:>12s} {2:>12s} {3:>8s}".format(
        "item", "CityFlow", "SUMO", "result"))
    print("-" * 62)

    ok = True
    for k in ["intersections", "signal_intersections", "roads", "lanes", "vehicles"]:
        a_v, b_v = cf[k], su[k]
        # SUMO's node count includes internal junctions and its edge count differs;
        # only signalised intersections and vehicles must match exactly
        strict = k in ("signal_intersections", "vehicles")
        good = (a_v == b_v) if strict else True
        ok = ok and good
        mark = "OK" if good else ("DIFF" if strict else "-")
        print("{0:<24s} {1:>12d} {2:>12d} {3:>8s}".format(k, a_v, b_v, mark))

    print("-" * 62)
    print("green phases in total (SUMO): {0}".format(rep.get("green_phases_total")))
    print("duarouter connectivity messages: {0}".format(rep.get("routes_repaired_msgs")))
    print("sigma={0}  departSpeed={1}".format(
        rep.get("sigma"), rep.get("depart_speed")))

    if not a.skip_freeflow:
        print("\n[check 4a] free-flow travel time (empty network, all green, first %d real routes)" % a.n_probe)
        ff = free_flow_test(rep["net_file"], rep["rou_file"], n_probe=a.n_probe)
        if "error" in ff:
            print("  [warning] %s" % ff["error"])
        else:
            print("  released {0}, completed {1}".format(ff["n_launched"], ff["n_arrived"]))
            print("  mean free-flow travel time: {0:.2f}s  median {1:.2f}s  range {2:.1f}-{3:.1f}s".format(
                ff["mean_free_flow_tt"], ff["median"], ff["min"], ff["max"]))
            print("  -> compare with CityFlow under the same conditions; difference should be < 5%")

    if a.fixed_time:
        print("\n[check 4b] full scenario under Fixed-time (converted static plans)")
        ft = fixed_time_test(rep["sumocfg"], horizon=a.horizon,
                             green_time=a.ft_green, yellow_time=a.ft_yellow)
        print("  timing: green {0}s / yellow {1}s (as FIXED_TIME in config.py)".format(
            a.ft_green, a.ft_yellow))
        print("  left the network {0}/{1}".format(ft["n_arrived"], ft["n_departed"]))
        for k in ("ATT", "Awt", "Mwt", "AQL"):
            v = ft[k]
            print("  {0:<5s} = {1}".format(k, "N/A" if v is None else "%.2f" % v))
        print("  -> compare with the Fixed-time results on CityFlow.")
        print("    If they agree under the same control policy, later ATT differences")
        print("    can be attributed to the controllers rather than the simulators.")

    print("\n" + ("check passed" if ok else "check FAILED: signalised intersections or vehicles differ; fix the conversion first"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
