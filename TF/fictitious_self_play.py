import time
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import ext
import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from TF.FSP_sampler import RARLSampler
from rllab.misc.overrides import overrides
from TF.reservoir_buffer import ReservoirBuffer
import numpy as np
import copy

class RARL(TRPO):

    def __init__(
            self,
            policy2,
            baseline2,
            regressor1,
            regressor2,
            obs1_dim,
            obs2_dim,
            action1_dim,
            action2_dim,
            optimizer_args=None,
            transfer=True,
            record_rewards=True,
            rewards=None,
            N1=1,
            N2=1,
            Nr1=1,
            Nr2=1,
            reset_regressor=False,
            use_regressor1=True,
            clip_path=True,
            buffer_size=2e4,
            buffer_keep_prob=0.25,
            buffer1=None,
            buffer2=None,
            **kwargs):
        self.transfer = transfer
        sampler_cls = RARLSampler
        sampler_args = dict()
        self.policy2 = policy2
        self.baseline2 = baseline2
        if optimizer_args is None:
            optimizer_args = dict()
        self.optimizer2 = ConjugateGradientOptimizer(**optimizer_args)

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
                self.rewards['Loss1'] = []

                self.rewards['average_discounted_return2'] = []
                self.rewards['AverageReturn2'] = []
                self.rewards['StdReturn2'] = []
                self.rewards['MaxReturn2'] = []
                self.rewards['MinReturn2'] = []
                self.rewards['Loss2'] = []
            else:
                self.rewards = rewards

        self.N1 = N1
        self.N2 = N2
        self.Nr1 = Nr1
        self.Nr2 = Nr2
        if buffer1 is None:
            self.buffer1 = ReservoirBuffer(size=buffer_size,keep_prob=buffer_keep_prob)
        else:
            self.buffer1 = buffer1
        if buffer2 is None:
            self.buffer2 = ReservoirBuffer(size=buffer_size,keep_prob=buffer_keep_prob)
        else:
            self.buffer2 = buffer2
        self.regressor1 = regressor1
        self.regressor2 = regressor2
        self.reset_regressor = reset_regressor
        self.use_regressor1 = use_regressor1
        self.clip_path = clip_path
        super(RARL, self).__init__(sampler_cls=sampler_cls,sampler_args=sampler_args,optimizer_args=optimizer_args,**kwargs)

    @overrides
    def init_opt(self):
        #first policy
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

        #second policy
        is_recurrent = int(self.policy2.recurrent)

        extra_dims=1 + is_recurrent
        name = 'obs2'
        obs_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.obs2_dim], name=name)

        name = 'action2'
        action_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.action2_dim], name=name)

        advantage_var = tensor_utils.new_tensor(
            'advantage2',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy2.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old2_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='new2_%s' %k)
            for k, shape in self.policy2.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy2.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid2")
        else:
            valid_var = None

        dist_info_vars = self.policy2.dist_info_sym(obs_var, state_info_vars)
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

        self.optimizer2.update_opt(
            loss=surr_loss,
            target=self.policy2,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )


        # is_recurrent = int(self.policy2.recurrent)

        # extra_dims=1 + is_recurrent
        # name = 'obs'
        # obs_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.obs2_dim], name=name)

        # name = 'action'
        # action_var = tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.action2_dim], name=name)

        # advantage_var = tensor_utils.new_tensor(
        #     'advantage',
        #     ndim=1 + is_recurrent,
        #     dtype=tf.float32,
        # )
        # dist = self.policy2.distribution

        # old_dist_info_vars = {
        #     k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
        #     for k, shape in dist.dist_info_specs
        #     }
        # old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        # state_info_vars = {
        #     k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
        #     for k, shape in self.policy2.state_info_specs
        #     }
        # state_info_vars_list = [state_info_vars[k] for k in self.policy2.state_info_keys]

        # if is_recurrent:
        #     valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        # else:
        #     valid_var = None

        # dist_info_vars = self.policy2.dist_info_sym(obs_var, state_info_vars)
        # kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        # lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        # if is_recurrent:
        #     mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
        #     surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        # else:
        #     mean_kl = tf.reduce_mean(kl)
        #     surr_loss = - tf.reduce_mean(lr * advantage_var)

        # input_list = [
        #                  obs_var,
        #                  action_var,
        #                  advantage_var,
        #              ] + state_info_vars_list + old_dist_info_vars_list
        # if is_recurrent:
        #     input_list.append(valid_var)

        # self.optimizer2.update_opt(
        #     loss=surr_loss,
        #     target=self.policy2,
        #     leq_constraint=(mean_kl, self.step_size),
        #     inputs=input_list,
        #     constraint_name="mean_kl"
        # )

    @overrides
    def optimize_policy(self, itr, samples_data, policy_num):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        if policy_num == 1:
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        else:
            state_info_list = [agent_infos[k] for k in self.policy2.state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policy2.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        if policy_num == 1:
            if self.policy.recurrent:
                all_input_values += (samples_data["valids"],)
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
            if self.policy2.recurrent:
                all_input_values += (samples_data["valids"],)
            logger.log("Computing loss before")
            loss_before = self.optimizer2.loss(all_input_values)
            logger.log("Computing KL before")
            mean_kl_before = self.optimizer2.constraint_val(all_input_values)
            logger.log("Optimizing")
            self.optimizer2.optimize(all_input_values)
            logger.log("Computing KL after")
            mean_kl = self.optimizer2.constraint_val(all_input_values)
            logger.log("Computing loss after")
            loss_after = self.optimizer2.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def obtain_samples(self, itr, player1_avg, player2_avg, policy_num):
        return self.sampler.obtain_samples(itr, player1_avg, player2_avg, policy_num)
    @overrides
    def process_samples(self, itr, paths, policy_num):
        return self.sampler.process_samples(itr, paths, policy_num)

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
            if self.reset_regressor:
                for var in self.regressor1.get_params(trainable=True):
                    sess.run(var.initializer)
                for var in self.regressor2.get_params(trainable=True):
                    sess.run(var.initializer)

            for n1 in range(self.N1):
                with logger.prefix('itr #%d ' % itr + 'n1 #%d |' % n1):
                    logger.log("training policy 1...")
                    logger.log("Obtaining samples...")

                    paths = self.obtain_samples(itr=itr,player1_avg=False,player2_avg=True,policy_num=1)

                    if self.clip_path:
                        clipped_paths = copy.deepcopy(paths)    
                        for clipped_path in clipped_paths:
                            clipped_path["actions"] = np.clip(clipped_path["actions"],-1.0,1.0)
                        self.buffer1.populate(clipped_paths)
                    else:
                        self.buffer1.populate(paths)

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
            if self.use_regressor1:
                for nr1 in range(self.Nr1):
                    with logger.prefix('itr #%d ' % itr + 'nr1 #%d |' % nr1):
                        xs, ys = self.buffer1.get_data()
                        loss1 = self.regressor1.fit(xs,ys)
                        if self.record_rewards:
                            self.rewards['Loss1'].append(loss1)

            if itr != self.n_itr-1: #don't train adversary at last time
                for n2 in range(self.N2):
                    if itr != self.n_itr-1: #don't train adversary at last time
                        with logger.prefix('itr #%d ' % itr + 'n2 #%d |' % n2):
                            logger.log("training policy 2...")
                            logger.log("Obtaining samples...")

                            if self.use_regressor1:
                                paths = self.obtain_samples(itr=itr,player1_avg=True,player2_avg=False,policy_num=2)
                            else:
                                paths = self.obtain_samples(itr=itr,player1_avg=False,player2_avg=False,policy_num=2)

                            if self.clip_path:
                                clipped_paths = copy.deepcopy(paths)    
                                for clipped_path in clipped_paths:
                                    clipped_path["actions"] = np.clip(clipped_path["actions"],-1.0,1.0)
                                self.buffer2.populate(clipped_paths)
                            else:
                                self.buffer2.populate(paths)

                            logger.log("Processing samples...")
                            samples_data = self.process_samples(itr, paths, 2)

                            if self.record_rewards:
                                undiscounted_returns = [sum(path["rewards"]) for path in paths]
                                average_discounted_return = np.mean([path["returns"][0] for path in paths])
                                AverageReturn = np.mean(undiscounted_returns)
                                StdReturn = np.std(undiscounted_returns)
                                MaxReturn = np.max(undiscounted_returns)
                                MinReturn = np.min(undiscounted_returns)
                                self.rewards['average_discounted_return2'].append(average_discounted_return)
                                self.rewards['AverageReturn2'].append(AverageReturn)
                                self.rewards['StdReturn2'].append(StdReturn)
                                self.rewards['MaxReturn2'].append(MaxReturn)
                                self.rewards['MinReturn2'].append(MinReturn)

                            logger.log("Logging diagnostics...")
                            self.log_diagnostics(paths, 2)
                            logger.log("Optimizing policy...")
                            self.optimize_policy(itr, samples_data, 2)

                            logger.record_tabular('Time', time.time() - start_time)
                            logger.record_tabular('ItrTime', time.time() - itr_start_time)
                            logger.dump_tabular(with_prefix=False)
                for nr2 in range(self.Nr2):
                    with logger.prefix('itr #%d ' % itr + 'nr2 #%d |' % nr2):
                        xs, ys = self.buffer2.get_data()
                        loss2 = self.regressor2.fit(xs,ys)
                        if self.record_rewards:
                            self.rewards['Loss2'].append(loss2)

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
                policy2=self.policy2,
                regressor1=self.regressor1,
                regressor2=self.regressor2,
                buffer1=self.buffer1,
                buffer2=self.buffer2,
                baseline=self.baseline,
                baseline2=self.baseline2,
                env=self.env,
                rewards=self.rewards,
            )
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                policy2=self.policy2,
                regressor1=self.regressor1,
                regressor2=self.regressor2,
                buffer1=self.buffer1,
                buffer2=self.buffer2,
                baseline=self.baseline,
                baseline2=self.baseline2,
                env=self.env,
            )

    @overrides
    def log_diagnostics(self, paths, policy_num):
        self.env.log_diagnostics(paths)
        if policy_num == 1:
            self.policy.log_diagnostics(paths)
            self.baseline.log_diagnostics(paths)
        else:
            self.policy2.log_diagnostics(paths)
            self.baseline2.log_diagnostics(paths)