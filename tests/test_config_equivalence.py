"""Check that the command lines in the README reproduce the configuration of the paper.

For each of the 18 CityFlow runs of the full model (configs/paper_cityflow_full_model.json)
run_d2marl.py is called with the per-scenario flags of the README, and the env/agent
configs it would pass to the training pipeline are compared key by key. TensorFlow is
not needed: the pipeline is replaced by a stub that only records the configs.

Only the run label (SUMO_LABEL) and START_ROUND (absent = 0) may differ.
"""
import json
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLAGS = {
    "hz": ["-hangzhou", "-weight_max_wait", "-0.1"],
    "jn": ["-jinan"],
    "ny": ["-newyork163"],
}
IGNORED = {"SUMO_LABEL", "START_ROUND"}


class ConfigEquivalence(unittest.TestCase):
    def setUp(self):
        if "tensorflow" not in sys.modules:
            tf = types.ModuleType("tensorflow")
            tf.random = types.SimpleNamespace(set_seed=lambda s: None)
            sys.modules["tensorflow"] = tf
        import run_d2marl
        self.mod = run_d2marl
        self.captured = []
        run_d2marl.pipeline_wrapper = (
            lambda dic_agent_conf, dic_traffic_env_conf, dic_path:
            self.captured.append((dic_agent_conf, dic_traffic_env_conf)))
        self.runs = json.loads((ROOT / "configs/paper_cityflow_full_model.json").read_text())["runs"]

    def test_cli_reproduces_paper_configs(self):
        cwd = os.getcwd()
        os.chdir(str(ROOT))
        try:
            for r in self.runs:
                argv = (["run_d2marl.py"] + FLAGS[r["flow"][:2]]
                        + ["-traffic_file", r["env"]["TRAFFIC_FILE"], "-seed", str(r["seed"]),
                           "-single_process", "-memo", "test"])
                os.environ["PYTHONHASHSEED"] = str(r["seed"])   # avoid the self-restart
                sys.argv = argv
                self.captured.clear()
                self.mod.main(self.mod.parse_args())
                agent, env = json.loads(json.dumps(self.captured[0]))
                self.assertEqual(agent, r["agent"], (r["flow"], r["seed"]))
                exp = r["env"]
                keys = (set(exp) | set(env)) - IGNORED
                diff = {k: (exp.get(k), env.get(k)) for k in keys if exp.get(k) != env.get(k)}
                self.maxDiff = None
                self.assertEqual(diff, {}, (r["flow"], r["seed"]))
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
