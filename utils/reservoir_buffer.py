import numpy as np
import random

class ReservoirBuffer():
	def __init__(self,size,keep_prob=None):
		self.size = size
		self.current_size = 0
		self.item_count = 0
		self.observations = []
		self.actions = []
		self.keep_prob = keep_prob

	def populate(self,paths): # array of dictionary
		#[sum(path["rewards"]) for path in paths]
		for path in paths:
			for i in range(len(path["observations"])):
				self.item_count += 1
				if self.current_size<self.size:
					self.observations.append(path["observations"][i])
					self.actions.append(path["actions"][i])
					self.current_size += 1
				elif self.current_size>=self.size:
					if self.keep_prob is None:
						if random.random()<self.size/self.item_count:
							replace = random.randint(0,self.size-1)
							self.observations[replace] = path["observations"][i]
							self.actions[replace] = path["actions"][i]
					else:
						if random.random()<self.keep_prob:
							replace = random.randint(0,self.size-1)
							self.observations[replace] = path["observations"][i]
							self.actions[replace] = path["actions"][i]

	def get_data(self):
		return self.observations, self.actions

	def reset(self):
		self.current_size = 0
		self.item_count = 0
		self.observations = []
		self.actions = []


