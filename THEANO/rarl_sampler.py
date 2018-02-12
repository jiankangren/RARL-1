from rllab.algos.base import RLAlgorithm
import THEANO.parallel_sampler as parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy


class RARLSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, self.algo.policy2, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr,policy_num):
        cur_params1 = self.algo.policy.get_param_values()
        cur_params2 = self.algo.policy2.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy1_params=cur_params1,
            policy2_params=cur_params2,
            policy_num=policy_num,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated