from .generator import Generator
import glob

from .construct_sample import ConstructSample
from .updater import Updater
from . import model_test
import json
import shutil
import os
import time
from multiprocessing import Process
import random
import numpy as np
# TensorFlow is only used for the process-level random seed. LLMLight runs without
# TF and uses oneline rather than pipeline, but utils/utils.py imports both, so a
# hard import would break LLMLight runs.
try:
    import tensorflow as tf
except ImportError:
    tf = None


def path_check(dic_path, resume=False):
    # with resume=True the directory is expected to exist
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default" and not resume:
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default" and not resume:
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]

    # SUMO runs copy net.xml / rou.xml instead of the CityFlow json files
    if str(dic_traffic_env_conf.get("SIMULATOR", "cityflow")).lower() == "sumo":
        for key in ("SUMO_NET_FILE", "SUMO_ROU_FILE"):
            fname = dic_traffic_env_conf[key]
            shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], fname),
                        os.path.join(path, fname))
        return

    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):
    # ================= independent random seed per worker process =================
    base_seed = dic_traffic_env_conf.get("SEED", 0)
    process_seed = base_seed + cnt_round * 100 + cnt_gen
    random.seed(process_seed)
    np.random.seed(process_seed)
    if tf is not None:
        tf.random.set_seed(process_seed)
    # ===================================================================

    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          )
    print("make generator")
    generator.generate()
    print("generator_wrapper end")
    return


def updater_wrapper(cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):
    # ================= random seed of the training process =================
    base_seed = dic_traffic_env_conf.get("SEED", 0)
    process_seed = base_seed + cnt_round * 100 + 9999
    random.seed(process_seed)
    np.random.seed(process_seed)
    if tf is not None:
        tf.random.set_seed(process_seed)
    # ==============================================================

    updater = Updater(
        cnt_round=cnt_round,
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    updater.load_sample_for_agents()
    updater.update_network_for_agents()
    print("updater_wrapper end")
    return


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.initialize()

    def initialize(self):
        path_check(self.dic_path,
                   resume=int(self.dic_traffic_env_conf.get("START_ROUND", 0)) > 0)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self, multi_process=False):
        # Resume: with START_ROUND > 0, running_time.csv is not overwritten and the loop
        # starts at the resume point. Round N loads round_{N-1}_inter_{i}
        # (network_agent.py), so the run continues seamlessly.
        start_round = int(self.dic_traffic_env_conf.get("START_ROUND", 0))
        tpath = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv")
        if start_round == 0 or not os.path.exists(tpath):
            f_time = open(tpath, "w")
            f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
            f_time.close()
        else:
            print("Resuming from round %d (of %d)"
                  % (start_round, self.dic_traffic_env_conf["NUM_ROUNDS"]))
            # The resume round was being written when the job stopped and may hold
            # partial data; it is regenerated from scratch so that no partial
            # samples enter the replay buffer.
            import shutil
            for sub in ("train_round", "test_round"):
                stale = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                     sub, "round_%d" % start_round)
                if os.path.isdir(stale):
                    shutil.rmtree(stale, ignore_errors=True)
                    print("  removed partial data of the resume round: %s" % stale)
        for cnt_round in range(start_round, self.dic_traffic_env_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()
            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                print("-------------- use multi-process for generator -------------")
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    p = Process(target=generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path,
                                      self.dic_agent_conf, self.dic_traffic_env_conf)
                                )
                    print("before")
                    p.start()
                    print("end")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    generator_wrapper(cnt_round=cnt_round,
                                      cnt_gen=cnt_gen,
                                      dic_path=self.dic_path,
                                      dic_agent_conf=self.dic_agent_conf,
                                      dic_traffic_env_conf=self.dic_traffic_env_conf)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time

            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)
            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()

            # The raw inter_*.pkl samples of this round have been aggregated into
            # total_samples_inter_*.pkl and are not read again (updater reads only
            # total_samples_*, ConstructSample only the current round). Over 100 rounds
            # they can reach ~1 TB, so they are deleted after use.
            # Set KEEP_RAW_SAMPLES=True to keep them (e.g. for debugging construct_sample).
            if not self.dic_traffic_env_conf.get("KEEP_RAW_SAMPLES", False):
                _stale = glob.glob(os.path.join(
                    train_round, "round_{0}".format(cnt_round),
                    "generator_*", "inter_*.pkl"))
                _freed = 0
                for _f in _stale:
                    try:
                        _freed += os.path.getsize(_f)
                        os.remove(_f)
                    except OSError:
                        pass
                if _stale:
                    print("[cleanup] round %d: removed %d intermediate pkl files, freed %.1f MB"
                          % (cnt_round, len(_stale), _freed / 1048576.0))

            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_network_start_time = time.time()
            if self.dic_traffic_env_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    updater_wrapper(cnt_round=cnt_round,
                                    dic_agent_conf=self.dic_agent_conf,
                                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                                    dic_path=self.dic_path)

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            # save a simulation replay only if requested
            save_replay_round = self.dic_traffic_env_conf.get("SAVE_REPLAY_ROUND")
            save_replay = (save_replay_round is not None and cnt_round == save_replay_round)
            if save_replay:
                print(f"Round {cnt_round}: saving simulation replay")
            model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round,
                            self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf,
                            save_replay=save_replay)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            print("Generator time: ", generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time, making_samples_total_time,
                                                            update_network_total_time, test_evaluation_total_time,
                                                            time.time()-round_start_time))
            f_time.close()
