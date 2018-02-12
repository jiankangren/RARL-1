import numpy as np
from rllab.misc import tensor_utils
import time
from rllab.misc import logger

def rarl_rollout(env, agent1, agent2, policy_num, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    #logger.log("rollout~~~~~~~~~~~~~~~~~~~")
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent1.reset()
    agent2.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a1, agent1_info = agent1.get_action(o)
        a2, agent2_info = agent2.get_action(o)
        action_true = np.append(a1,a2)

        Action = {}
        Action['action'] = np.append(a1,a2)
        Action['dist1'] = agent1_info
        Action['dist2'] = agent2_info
        Action['policy_num'] = policy_num
        next_o, r, d, env_info = env.step(Action)
        observations.append(env.observation_space.flatten(o))

        if policy_num==1:
            rewards.append(r)
            actions.append(env.action_space.flatten(a1))
            agent_infos.append(agent1_info)
        else:
            rewards.append(r)
            actions.append(env.action_space.flatten(a2))  
            agent_infos.append(agent2_info)
            
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
