"""
run_coordlight.py
=================
Evaluates a trained CoordLight model on SUMO with the metric pipeline of this code base.

Why a separate entry point
--------------------
CoordLight is a separate PyTorch + PPO + Ray project and does not fit into the
TF + DQN pipeline of run_baseline.py. What must be shared for a fair comparison
is not the training framework but two things:

    1. simulator   -> coordlight_sumo/cityflow_api_over_sumo.py
                      (TraCI back end with the cityflow.Engine interface)
    2. metrics     -> this file rearranges the evaluation output of CoordLight into
                      records/<memo>/<traffic>.json_<ts>/test_round/round_0/
                      directory layout, with the same field names as D2-MARL,
                      so that d2marl_sumo.collect_results computes
                      ATT / Awt / AQL / Mwt / Cd / Gini / finish_rate with the same code.

CoordLight's observation (QDSE), reward, phase logic and network are unchanged;
only the engine and the metric export are replaced.

Usage
----
    python -m coordlight_sumo.run_coordlight \
        -traffic_file anon_4_4_hangzhou_real.json \
        -model_path coordlight_sumo/CoordLight/Train_CityFlow_CoordLight/<run>_seed10 \
        -seed 10 -out_memo sumo-base-CoordLight-hz1-seed10
"""

import argparse
import csv
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CL_DIR = os.path.join(HERE, "CoordLight")


def scenario_stem(sumo_dir, traffic_file):
    """anon_4_4_hangzhou_real.json -> sumo_scenarios/anon_4_4_hangzhou_real/<same stem>"""
    stem_name = traffic_file[:-5] if traffic_file.endswith(".json") else traffic_file
    d = os.path.join(sumo_dir, stem_name)
    if not os.path.isdir(d):
        raise SystemExit("SUMO scenario directory not found: %s" % d)
    return os.path.join(d, stem_name)


def export_to_records(model_path, episode, out_dir, adv):
    """
    Move the output of Tester into the test_round/round_0 layout of this code base.

    Tester writes to two places:
        <model_path>/veh_info_<ep>/vehicle_inter_*.csv   -- header ",enter_time,leave_time",
                                                            same as here, copied as is
        <model_path>/delay_events_over_90s.csv           -- header needs renaming
    """
    os.makedirs(out_dir, exist_ok=True)

    veh_dir = os.path.join(model_path, "veh_info_%d" % episode)
    n = 0
    if os.path.isdir(veh_dir):
        for f in sorted(os.listdir(veh_dir)):
            if f.startswith("vehicle_inter_") and f.endswith(".csv"):
                shutil.copy2(os.path.join(veh_dir, f), os.path.join(out_dir, f))
                n += 1
    if n == 0:
        raise SystemExit("no vehicle_inter_*.csv found; this evaluation is invalid")

    # delay_events: Tester names the columns cumulative_wait_sec / max_wait_sec,
    # collect_results reads wait_time; map them explicitly instead of relying on
    # matching names.
    src = os.path.join(model_path, "delay_events_over_90s.csv")
    dst = os.path.join(out_dir, "delay_events_over_90s.csv")
    rows = []
    if os.path.exists(src):
        with open(src) as fh:
            for r in csv.DictReader(fh):
                w = r.get("wait_time") or r.get("max_wait_sec")
                if w in (None, ""):
                    continue
                try:
                    w = float(w)
                except ValueError:
                    continue
                rows.append({"vehicle_id": r.get("vehicle_id", ""),
                             "link": r.get("link", ""),
                             "wait_time": w,
                             "exceed_time": r.get("exceed_time", w - 90.0),
                             "end_time": r.get("end_time", "")})
    # Write the file (header only) even if no wait exceeded 90 s.
    # collect_results treats an empty file like a missing one (Mwt "<90", censored),
    # but the empty file shows that the round was run.
    with open(dst, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["vehicle_id", "link", "wait_time",
                                             "exceed_time", "end_time"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    # all_vehicles_wait_time.csv: Tester writes it under model_path with the column
    # cumulative_wait_sec; collect_results reads total_wait_time
    # (= env.vehicle_total_wait_time, cumulative waiting seconds of every vehicle).
    # Same meaning, different column name; without this step Mwt_cum would be
    # missing for CoordLight.
    src_all = os.path.join(model_path, "all_vehicles_wait_time.csv")
    if os.path.exists(src_all):
        with open(src_all) as fh, \
             open(os.path.join(out_dir, "all_vehicles_wait_time.csv"),
                  "w", newline="") as out:
            wtr = csv.DictWriter(out, fieldnames=["vehicle_id", "total_wait_time"])
            wtr.writeheader()
            for r in csv.DictReader(fh):
                w = r.get("total_wait_time") or r.get("cumulative_wait_sec")
                if w in (None, ""):
                    continue
                try:
                    w = float(w)
                except ValueError:
                    continue
                wtr.writerow({"vehicle_id": r.get("vehicle_id", ""),
                              "total_wait_time": w})
    else:
        print("!! %s not found; Mwt_cum will be missing for this round" % src_all)

    # reward_log.txt: the labels must match those written by utils/model_test.py,
    # otherwise the regexes of collect_results.parse_reward_log do not match and
    # Awt/AQL/Gini/Cd silently become missing.
    awt, mwt, aql, cd, gini = adv
    with open(os.path.join(out_dir, "reward_log.txt"), "w") as f:
        f.write("Episode average reward: 0.0\n")
        f.write("Total steps: 0\n")
        f.write("Total reward: 0.0\n")
        f.write("AWT (average waiting time): %s\n" % awt)
        f.write("AQL (average queue length): %s\n" % aql)
        f.write("Gini coefficient: %s\n" % gini)
        f.write("Severe delay events (>90s): %d\n" % int(cd))
        f.write("\n# exported by coordlight_sumo/run_coordlight.py; "
                "Mwt is computed from delay_events_over_90s.csv, not stored here\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_path", required=True,
                    help="CoordLight training output directory (must contain model_MATSC/checkpoint.pkl)")
    ap.add_argument("-traffic_file", required=True)
    ap.add_argument("-seed", type=int, default=10)
    ap.add_argument("-out_memo", required=True)
    ap.add_argument("-sumo_dir", default=os.path.join(ROOT, "sumo_scenarios"))
    ap.add_argument("-records", default=os.path.join(ROOT, "records"))
    ap.add_argument("-horizon", type=int, default=3600)
    ap.add_argument("-use_gpu", action="store_true", default=False)
    ap.add_argument("-n_eval", type=int, default=5,
                    help="number of evaluation episodes, exported as round_0..round_{n-1}. "
                         "Default 5, matching the last-5-test-rounds window of the other "
                         "baselines; with 1 episode CoordLight would have one sample per "
                         "seed, which is not comparable for high-variance metrics such as Mwt")
    a = ap.parse_args()

    stem = scenario_stem(a.sumo_dir, a.traffic_file)
    net, rou = stem + ".net.xml", stem + ".repaired.rou.xml"
    for p in (net, rou):
        if not os.path.exists(p):
            raise SystemExit("missing file: %s" % p)

    model_path = os.path.abspath(a.model_path)
    # either checkpoint.pkl or state_dict.pth is enough: Tester.set_weights prefers
    # the former and falls back to the latter (inference needs only the weights)
    _md = os.path.join(model_path, "model_MATSC")
    if not any(os.path.exists(os.path.join(_md, f))
               for f in ("checkpoint.pkl", "state_dict.pth")):
        raise SystemExit(
            "%s contains neither checkpoint.pkl nor state_dict.pth; "
            "training of this scenario did not finish" % _md)

    # switches for the compatibility layer; they must be set before Parameters is
    # imported, because Parameters.py reads the environment variables at import time.
    os.environ["CL_SIMULATOR"] = "sumo"
    os.environ["CL_SUMO_NET"] = os.path.abspath(net)
    os.environ["CL_SUMO_ROU"] = os.path.abspath(rou)
    os.environ["CL_HORIZON"] = str(a.horizon)
    os.environ["CL_ENABLE_SSM"] = "0"

    print("=" * 66)
    print("CoordLight on SUMO")
    print("  net      : %s" % net)
    print("  rou      : %s" % rou)
    print("  model    : %s" % model_path)
    print("  seed     : %d" % a.seed)
    print("  out_memo : %s" % a.out_memo)
    print("=" * 66)

    # Tester.py uses flat imports (`from Utils import ...`), so CoordLight/ is put
    # on sys.path and becomes the working directory before the import
    cwd = os.getcwd()
    sys.path.insert(0, CL_DIR)
    os.chdir(CL_DIR)
    try:
        from Tester import Tester
        tester = Tester(meta_agent_id=0, args_dict={
            "seed": a.seed,
            "num_test": 1,
            "use_gpu": a.use_gpu,
            "env_type": "CITYFLOW",     # the engine is chosen by CL_SIMULATOR; this only selects the env class
            "test_path": os.path.dirname(model_path) + os.sep,
            "experiment_name": os.path.basename(model_path),
        })
        # remove append-mode csv files left from an earlier run, otherwise mode='a' merges both runs
        for f in ("all_vehicles_wait_time.csv", "delay_events_over_90s.csv"):
            p = os.path.join(tester.model_path, f)
            if os.path.exists(p):
                os.remove(p)
        traffic = (a.traffic_file if a.traffic_file.endswith(".json")
                   else a.traffic_file + ".json")
        run_dir = os.path.join(a.records, a.out_memo,
                               traffic + "_" + time.strftime("%m_%d_%H_%M_%S"))
        results = []
        for ep in range(a.n_eval):
            # a different evaluation seed per episode; otherwise all n_eval episodes
            # would be identical and the variance would look like zero. The other
            # baselines also have independent simulations in their 5 test rounds.
            eval_seed = a.seed + ep
            print("\n---- evaluation episode %d/%d (eval_seed=%d) ----"
                  % (ep + 1, a.n_eval, eval_seed))
            # clear the append-mode csv files every episode
            for f in ("all_vehicles_wait_time.csv", "delay_events_over_90s.csv"):
                pth = os.path.join(tester.model_path, f)
                if os.path.exists(pth):
                    os.remove(pth)
            _, adv = tester.run_episode_single_threaded(
                curr_episode=ep, eval_seed=eval_seed)
            out_dir = os.path.join(run_dir, "test_round", "round_%d" % ep)
            n = export_to_records(tester.model_path, ep, out_dir, adv)
            awt, mwt, aql, cd, gini = adv
            print("  exported %d vehicle_inter_*.csv -> %s" % (n, out_dir))
            print("  reported by Tester: AWT=%.2f MaxWait=%.2f AQL=%.2f Cd=%d Gini=%.4f"
                  % (awt, mwt, aql, int(cd), gini))
            results.append(adv)
    finally:
        os.chdir(cwd)

    print("\nSummary of %d evaluation episodes (collect_results takes their mean/median):" % len(results))
    print("  %-6s %10s %10s %8s %8s %8s" % ("ep", "AWT", "MaxWait", "AQL", "Cd", "Gini"))
    for i, (awt, mwt, aql, cd, gini) in enumerate(results):
        print("  %-6d %10.2f %10.2f %8.2f %8d %8.4f"
              % (i, awt, mwt, aql, int(cd), gini))
    print("\nCheck (required; a mismatch means the export is wrong):")
    print("  python -u -m d2marl_sumo.collect_results --records records "
          "--include %s" % a.out_memo)
    print("  Awt/AQL/Cd/Gini in the table must equal the values reported by Tester above;")
    print("  Mwt should equal MaxWait (or <90, censored, if no wait exceeded 90 s).")


if __name__ == "__main__":
    main()
