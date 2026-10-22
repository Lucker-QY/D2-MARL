"""Regression test (no TensorFlow needed) that executes the actual epsilon-decay statements of D2MARLAgent."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("base_agent", str(ROOT / "models/agent.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tree = ast.parse((ROOT / "models/d2marl_agent.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "D2MARLAgent")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "decayed_epsilon")
# Reuse the AST container of the original module (compatible with ast.Module in Python 3.6).
tree.body = init.body[start:start + 2]
DECAY = compile(tree, "d2marl_epsilon", "exec")


def agent(conf, rnd):
    obj = module.Agent(conf, {}, {}, "0")
    exec(DECAY, {"self": obj, "cnt_round": rnd})
    return obj


class ConfigIsolationTests(unittest.TestCase):
    def setUp(self):
        self.base = {"EPSILON": 0.8, "EPSILON_DECAY": 0.95,
                     "MIN_EPSILON": 0.2, "CNN_layers": [[32, 32]]}

    def test_serial_generator_and_updater_100_rounds(self):
        for rnd in range(100):
            generator = agent(copy.deepcopy(self.base), rnd)
            updater = agent(self.base, rnd)
            expected = max(0.8 * 0.95 ** rnd, 0.2)
            self.assertAlmostEqual(generator.dic_agent_conf["EPSILON"], expected)
            self.assertAlmostEqual(updater.dic_agent_conf["EPSILON"], expected)
            self.assertEqual(self.base["EPSILON"], 0.8)

    def test_agents_at_same_round_do_not_decay_each_other(self):
        eps = [agent(self.base, 7).dic_agent_conf["EPSILON"] for _ in range(16)]
        self.assertTrue(all(abs(v - 0.8 * 0.95 ** 7) < 1e-12 for v in eps))

    def test_nested_config_isolation(self):
        obj = agent(self.base, 1)
        obj.dic_agent_conf["CNN_layers"][0][0] = 99
        self.assertEqual(self.base["CNN_layers"], [[32, 32]])

    def test_baseline_decay_uses_unchanged_base_config(self):
        for filename, classname in (("colight_agent.py", "CoLightAgent"),
                                    ("network_agent.py", "NetworkAgent")):
            parsed = ast.parse((ROOT / "models" / filename).read_text())
            cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == classname)
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "decayed_epsilon")
            parsed.body = init.body[start:start + 2]
            code = compile(parsed, filename, "exec")
            for rnd in range(100):
                obj = module.Agent(self.base, {}, {}, "0")
                obj.cnt_round = rnd
                exec(code, {"self": obj, "cnt_round": rnd, "epsilon_decay": 0.95})
                self.assertAlmostEqual(obj.dic_agent_conf["EPSILON"], max(0.8 * 0.95 ** rnd, 0.2))
                self.assertEqual(self.base["EPSILON"], 0.8)

    def test_evaluation_remains_greedy(self):
        self.base.update(EPSILON=0, MIN_EPSILON=0)
        self.assertEqual(agent(self.base, 99).dic_agent_conf["EPSILON"], 0)


if __name__ == "__main__":
    unittest.main()
