from rllab.envs.base import Env
from rllab.spaces import Box, Discrete
from rllab.envs.base import Step

class PrisonerDilemma(Env):
	def __init__(self):
		self.num_action = 2

	def render(self):
		pass

	@property
	def observation_space(self):
	    return Discrete(1)

	@property
	def action_space(self):
		return Discrete(self.num_action)

	def close(self):
		pass

	def reset(self):
		return 0

	def step(self, Action):
		obs = 0
		done = True
		action = Action['action']
		a1 = action[0]
		a2 = action[1]
		policy_num = Action['policy_num']
		r = -100
		if a1 == 0:
			if a2 == 0:
				r = -1
			else:
				if policy_num == 1:
					r = -3
				else:
					r = 0
		else:
			if a2 == 0:
				if policy_num == 1:
					r = 0
				else:
					r = -3
			else:
				r = -2
		return Step(observation=obs, reward=r, done=done)