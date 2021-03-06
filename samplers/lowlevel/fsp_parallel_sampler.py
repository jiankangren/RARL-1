from samplers.lowlevel.rarl_rollout import rarl_rollout
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import tensor_utils
import pickle
import numpy as np


def _worker_init(G, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(_worker_init, [(id,) for id in range(singleton_pool.n_parallel)])


def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]


def _worker_populate_task(G, env, policy1, policy2, regressor1, regressor2,scope=None):
    G = _get_scoped_G(G, scope)
    G.env = pickle.loads(env)
    G.policy1 = pickle.loads(policy1)
    G.policy2 = pickle.loads(policy2)
    G.regressor1 = pickle.loads(regressor1)
    G.regressor2 = pickle.loads(regressor2)

def _worker_terminate_task(G, scope=None):
    G = _get_scoped_G(G, scope)
    if getattr(G, "env", None):
        G.env.terminate()
        G.env = None
    if getattr(G, "policy1", None):
        G.policy1.terminate()
        G.policy1 = None
    if getattr(G, "policy2", None):
        G.policy2.terminate()
        G.policy2 = None
    if getattr(G, "regressor1", None):
        G.regressor1.terminate()
        G.regressor1 = None
    if getattr(G, "regressor2", None):
        G.regressor2.terminate()
        G.regressor2 = None

def populate_task(env, policy1, policy2, regressor1, regressor2, scope=None):
    logger.log("Populating workers...")
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_populate_task,
            [(pickle.dumps(env), pickle.dumps(policy1), pickle.dumps(policy2), pickle.dumps(regressor1),pickle.dumps(regressor2),scope)] * singleton_pool.n_parallel
        )
    else:
        # avoid unnecessary copying
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = env
        G.policy1 = policy1
        G.policy2 = policy2
        G.regressor1 = regressor1
        G.regressor2 = regressor2
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(
        _worker_terminate_task,
        [(scope,)] * singleton_pool.n_parallel
    )


def _worker_set_seed(_, seed):
    logger.log("Setting seed to %d" % seed)
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in range(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params1, params2, params3, params4, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy1.set_param_values(params1)
    G.policy2.set_param_values(params2)
    G.regressor1.set_param_values(params3)
    G.regressor2.set_param_values(params4)

def _worker_set_env_params(G,params,scope=None):
    G = _get_scoped_G(G, scope)
    G.env.set_param_values(params)

def _worker_collect_one_path(G, player1_avg, player2_avg,policy_num, max_path_length, scope=None):
    G = _get_scoped_G(G, scope)
    agent1 = G.policy1
    agent2 = G.policy2
    if player1_avg:
        agent1 = G.regressor1
    if player2_avg:
        agent2 = G.regressor2
    path = rarl_rollout(G.env, agent1, agent2, policy_num, max_path_length)
    return path, len(path["rewards"])


def sample_paths(
        policy1_params,
        policy2_params,
        regressor1_params,
        regressor2_params,
        player1_avg,
        player2_avg,
        policy_num,
        max_samples,
        max_path_length=np.inf,
        env_params=None,
        scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy1_params, policy2_params, regressor1_params, regressor2_params, scope)] * singleton_pool.n_parallel
    )
    if env_params is not None:
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel
        )
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(player1_avg, player2_avg, policy_num, max_path_length, scope),
        show_prog_bar=True
    )


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
