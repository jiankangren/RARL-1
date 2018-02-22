import time
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import ext
import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from TF.MultiAd_sampler import RARLSampler
from rllab.misc.overrides import overrides
import numpy as np
import random

class RARL(TRPO):

    def __init__(
            self,
            obs1_dim,
            obs2_dim,
            action1_dim,
            action2_dim,
            n_1,
            n_2, 
            N2,
            policy2_class=None,
            policy2_args=None,
            policies2=None,
            baselines2=None,
            optimizer_args=None,
            transfer=True,
            record_rewards=True,
            rewards=None,
            **kwargs):
        self.transfer = transfer
        sampler_cls = RARLSampler
        sampler_args = dict()
        if policies2 is None:
            self.policies2 = []
            for i in range(N2):
                self.policies2.append(policy2_class(name="RARLTFPolicy2_"+str(i),**policy2_args))
        else:
            self.policies2 = policies2

        self.baselines2 = baselines2
        self.optimizers2 = []
        for i in range(N2):
            optimizer_args = dict()
            self.optimizers2.append(ConjugateGradientOptimizer(**optimizer_args))
        # self.policy2 = policy2
        # optimizer_args = dict()
        # self.optimizer2 = ConjugateGradientOptimizer(**optimizer_args)

        self.obs1_dim = obs1_dim
        self.obs2_dim = obs2_dim
        self.action1_dim = action1_dim
        self.action2_dim = action2_dim

        self.record_rewards = record_rewards
        if self.record_rewards:
            if rewards is None: #create empty dict
                self.rewards = {}
                self.rewards['average_discounted_return1'] = []
                self.rewards['AverageReturn1'] = []
                self.rewards['StdReturn1'] = []
                self.rewards['MaxReturn1'] = []
                self.rewards['MinReturn1'] = []
                for i in range(N2):
                    self.rewards['average_discounted_return2'+'_'+str(i)] = []
                    self.rewards['AverageReturn2'+'_'+str(i)] = []
                    self.rewards['StdReturn2'+'_'+str(i)] = []
                    self.rewards['MaxReturn2'+'_'+str(i)] = []
                    self.rewards['MinReturn2'+'_'+str(i)] = []
            else:
                self.rewards = rewards

        self.n_1 = n_1
        self.n_2 = n_2
        self.N2 = N2
        super(RARL, self).__init__(sampler_cls=sampler_cls,sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        #policy1
        is_recurrent = int(self.policy.recurrent)

        extra_dims=1 + is_recurrent
        name = 'obs'
        obs_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.obs1_dim], name=name)

        name = 'action'
        action_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.action1_dim], name=name)

        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        #policy2s
        for i in range(self.N2):
            is_recurrent = int(self.policies2[i].recurrent)

            extra_dims=1 + is_recurrent
            name = 'obs'
            obs_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.obs2_dim], name=name)

            name = 'action'
            action_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.action2_dim], name=name)

            advantage_var = tensor_utils.new_tensor(
                'advantage',
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            )
            dist = self.policies2[i].distribution

            old_dist_info_vars = {
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
                for k, shape in dist.dist_info_specs
                }
            old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

            state_info_vars = {
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
                for k, shape in self.policies2[i].state_info_specs
                }
            state_info_vars_list = [state_info_vars[k] for k in self.policies2[i].state_info_keys]

            if is_recurrent:
                valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
            else:
                valid_var = None

            dist_info_vars = self.policies2[i].dist_info_sym(obs_var, state_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            if is_recurrent:
                mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
                surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            else:
                mean_kl = tf.reduce_mean(kl)
                surr_loss = - tf.reduce_mean(lr * advantage_var)

            input_list = [
                             obs_var,
                             action_var,
                             advantage_var,
                         ] + state_info_vars_list + old_dist_info_vars_list
            if is_recurrent:
                input_list.append(valid_var)

            self.optimizers2[i].update_opt(
                loss=surr_loss,
                target=self.policies2[i],
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )

    @overrides
    def optimize_policy(self, itr, samples_data, policy_num, policy2_num=-1):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        if policy_num == 1:
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            all_input_values += tuple(state_info_list) + tuple(dist_info_list)
            if self.policy.recurrent:
                all_input_values += (samples_data["valids"],)
        else:
            state_info_list = [agent_infos[k] for k in self.policies2[policy2_num].state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policies2[policy2_num].distribution.dist_info_keys]
            all_input_values += tuple(state_info_list) + tuple(dist_info_list)
            if self.policies2[policy2_num].recurrent:
                all_input_values += (samples_data["valids"],)

        if policy_num == 1:
            logger.log("Computing loss before")
            loss_before = self.optimizer.loss(all_input_values)
            logger.log("Computing KL before")
            mean_kl_before = self.optimizer.constraint_val(all_input_values)
            logger.log("Optimizing")
            self.optimizer.optimize(all_input_values)
            logger.log("Computing KL after")
            mean_kl = self.optimizer.constraint_val(all_input_values)
            logger.log("Computing loss after")
            loss_after = self.optimizer.loss(all_input_values)
        else:
            logger.log("Computing loss before")
            loss_before = self.optimizers2[policy2_num].loss(all_input_values)
            logger.log("Computing KL before")
            mean_kl_before = self.optimizers2[policy2_num].constraint_val(all_input_values)
            logger.log("Optimizing")
            self.optimizers2[policy2_num].optimize(all_input_values)
            logger.log("Computing KL after")
            mean_kl = self.optimizers2[policy2_num].constraint_val(all_input_values)
            logger.log("Computing loss after")
            loss_after = self.optimizers2[policy2_num].loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def obtain_samples(self, itr, policy_num, policy2_num=-1):
        return self.sampler.obtain_samples(itr, policy_num, policy2_num)
    @overrides
    def process_samples(self, itr, paths, policy_num, policy2_num=-1):
        return self.sampler.process_samples(itr, paths, policy_num, policy2_num)

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            
        if not self.transfer:
            sess.run(tf.global_variables_initializer())
        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            for policy2_num in range(self.N2):
                for n1 in range(self.n_1):
                    with logger.prefix('itr #%d ' % itr + 'policy2_num #%d |' % policy2_num + 'n1 #%d |' % n1):                  
                        logger.log("training policy 1...")
                        logger.log("Obtaining samples...")
                        paths = self.obtain_samples(itr, 1, policy2_num=np.random.randint(0,self.N2))
                        logger.log("Processing samples...")
                        samples_data = self.process_samples(itr, paths, 1)

                        if self.record_rewards:
                            undiscounted_returns = [sum(path["rewards"]) for path in paths]
                            average_discounted_return = np.mean([path["returns"][0] for path in paths])
                            AverageReturn = np.mean(undiscounted_returns)
                            StdReturn = np.std(undiscounted_returns)
                            MaxReturn = np.max(undiscounted_returns)
                            MinReturn = np.min(undiscounted_returns)
                            self.rewards['average_discounted_return1'].append(average_discounted_return)
                            self.rewards['AverageReturn1'].append(AverageReturn)
                            self.rewards['StdReturn1'].append(StdReturn)
                            self.rewards['MaxReturn1'].append(MaxReturn)
                            self.rewards['MinReturn1'].append(MinReturn)


                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(paths, 1)
                        logger.log("Optimizing policy...")
                        self.optimize_policy(itr, samples_data, 1)

                        logger.record_tabular('Time', time.time() - start_time)
                        logger.record_tabular('ItrTime', time.time() - itr_start_time)
                        logger.dump_tabular(with_prefix=False)
                for n2 in range(self.n_2):
                    if itr != self.n_itr-1: #don't train adversary at last time
                        with logger.prefix('itr #%d ' % itr + 'policy2_num #%d |' % policy2_num + 'n2 #%d |' % n2):
                            logger.log("training policy 2...")
                            logger.log("Obtaining samples...")
                            paths = self.obtain_samples(itr, 2, policy2_num=policy2_num)
                            logger.log("Processing samples...")
                            samples_data = self.process_samples(itr, paths, 2, policy2_num)

                            if self.record_rewards:
                                undiscounted_returns = [sum(path["rewards"]) for path in paths]
                                average_discounted_return = np.mean([path["returns"][0] for path in paths])
                                AverageReturn = np.mean(undiscounted_returns)
                                StdReturn = np.std(undiscounted_returns)
                                MaxReturn = np.max(undiscounted_returns)
                                MinReturn = np.min(undiscounted_returns)
                                self.rewards['average_discounted_return2'+'_'+str(policy2_num)].append(average_discounted_return)
                                self.rewards['AverageReturn2'+'_'+str(policy2_num)].append(AverageReturn)
                                self.rewards['StdReturn2'+'_'+str(policy2_num)].append(StdReturn)
                                self.rewards['MaxReturn2'+'_'+str(policy2_num)].append(MaxReturn)
                                self.rewards['MinReturn2'+'_'+str(policy2_num)].append(MinReturn)

                            logger.log("Logging diagnostics...")
                            self.log_diagnostics(paths, 2, policy2_num=policy2_num)
                            logger.log("Optimizing policy...")
                            self.optimize_policy(itr, samples_data, 2, policy2_num=policy2_num)

                            logger.record_tabular('Time', time.time() - start_time)
                            logger.record_tabular('ItrTime', time.time() - itr_start_time)
                            logger.dump_tabular(with_prefix=False)


            logger.log("Saving snapshot...")
            params = self.get_itr_snapshot(itr)  # , **kwargs)
            logger.save_itr_params(itr, params)
            logger.log("Saved")
            # logger.record_tabular('Time', time.time() - start_time)
            # logger.record_tabular('ItrTime', time.time() - itr_start_time)
            # logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        if created_session:
            sess.close()

    @overrides
    def get_itr_snapshot(self, itr):
        if self.record_rewards:
            return dict(
                itr=itr,
                policy=self.policy,
                policies2=self.policies2,
                baseline=self.baseline,
                baselines2=self.baselines2,
                env=self.env,
                rewards=self.rewards,
            )
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                policies2=self.policies2,
                baseline=self.baseline,
                baselines2=self.baselines2,
                env=self.env,
            )

    @overrides
    def log_diagnostics(self, paths, policy_num, policy2_num=-1):
        self.env.log_diagnostics(paths)
        if policy_num == 1:
            self.policy.log_diagnostics(paths)
            self.baseline.log_diagnostics(paths)
        else:
            self.policies2[policy2_num].log_diagnostics(paths)
            self.baselines2[policy2_num].log_diagnostics(paths)
        

