# -*- coding: utf-8 -*-
"""Add the missing lane_length_dict to the roadnet_*_net_config.json files of CoordLight.

Why:
    The three net_config files shipped with CoordLight differ: Hangzhou 4x4 and
    Jinan 3x4 have lane_length_dict, New York 16x3 lacks it for all 48 intersections,
    so CoordLight fails on New York at start-up.
        Tls_Cityflow.__init__:18
        self.lane_length_dict = self.config_data['lane_length_dict']
        KeyError: 'lane_length_dict'

Definition:
    lane_length_dict maps "<road_id>_<lane index>" to the length of the road (m),
    i.e. the length of the road.points polyline in roadnet_*.json; all lanes of a
    road share it. Each intersection lists the union of its incoming_lane_list and
    outgoing_lane_list.

    The definition is verified: recomputing the existing values of Hangzhou (384
    lanes) and Jinan (288 lanes) reproduces them exactly (--verify does this), so
    the values generated for New York follow the same definition.

Usage:
    python -m coordlight_sumo.fix_net_config --verify          # check only, no writes
    python -m coordlight_sumo.fix_net_config --write           # add the missing entries
"""
import argparse
import json
import math
import os
import shutil

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "CoordLight", "Network", "Cityflow_map")

SCENARIOS = [
    ("Hangzhou 4x4", "Hangzhou/4_4/roadnet_4_4.json",   "Hangzhou/4_4/roadnet_4_4_net_config.json"),
    ("Jinan 3x4",    "Jinan/3_4/roadnet_3_4.json",      "Jinan/3_4/roadnet_3_4_net_config.json"),
    ("NewYork 16x3", "NewYork/16_3/roadnet_16_3.json",  "NewYork/16_3/roadnet_16_3_net_config.json"),
]


def road_lane_lengths(roadnet_path):
    """{lane_id: length (m)}, lane_id = <road_id>_<lane index>."""
    d = json.load(open(roadnet_path))
    out = {}
    for r in d["roads"]:
        pts = r["points"]
        length = sum(
            math.hypot(pts[i + 1]["x"] - pts[i]["x"], pts[i + 1]["y"] - pts[i]["y"])
            for i in range(len(pts) - 1))
        for li in range(len(r["lanes"])):
            out["%s_%d" % (r["id"], li)] = length
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write lane_length_dict into the net_config files that lack it (backup .bak first)")
    a = ap.parse_args()

    all_ok = True
    for tag, rn, cfg in SCENARIOS:
        rn_p, cfg_p = os.path.join(BASE, rn), os.path.join(BASE, cfg)
        if not (os.path.exists(rn_p) and os.path.exists(cfg_p)):
            print("[skip] %s: files missing" % tag)
            continue

        lanes = road_lane_lengths(rn_p)
        conf = json.load(open(cfg_p))
        missing = [k for k, v in conf.items() if "lane_length_dict" not in v]

        if not missing:
            # existing entries: use them to verify the definition
            tot = bad = 0
            for inter, v in conf.items():
                for lane, val in v["lane_length_dict"].items():
                    tot += 1
                    if lane not in lanes or abs(lanes[lane] - val) > 1e-6:
                        bad += 1
                        if bad <= 3:
                            print("    mismatch %s: file %s / recomputed %s"
                                  % (lane, val, lanes.get(lane)))
            print("[verify] %-12s %d intersections, %d lane lengths compared, %d mismatches"
                  % (tag, len(conf), tot, bad))
            all_ok = all_ok and bad == 0
            continue

        print("[missing] %-12s %d/%d intersections without lane_length_dict"
              % (tag, len(missing), len(conf)))
        if not a.write:
            print("        (use --write to add them, only after both verify lines above "
                  "show 0 mismatches)")
            continue
        if not all_ok:
            raise SystemExit("verification of the existing scenarios failed; refusing to write")

        n_lane = 0
        for inter, v in conf.items():
            need = set(v["incoming_lane_list"]) | set(v["outgoing_lane_list"])
            unknown = [l for l in need if l not in lanes]
            if unknown:
                raise SystemExit(
                    "%s: %s refers to lanes that are not in the roadnet: %s"
                    % (tag, inter, unknown[:5]))
            v["lane_length_dict"] = {l: lanes[l] for l in sorted(need)}
            n_lane += len(need)

        shutil.copy2(cfg_p, cfg_p + ".bak")
        with open(cfg_p, "w") as f:
            json.dump(conf, f)
        print("        added %d intersections, %d lanes; original saved as %s.bak"
              % (len(conf), n_lane, os.path.basename(cfg_p)))


if __name__ == "__main__":
    main()
