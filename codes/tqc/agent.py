import numpy as np

from sb3_contrib import TQC
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import time

import gym
import numpy as np
from os.path import exists
from collections import OrderedDict
import time
import gym

class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(
		self, 
		env_specs,
	 	logpath=None,
		):
		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space']
		self.act_space = env_specs['action_space']


		env_id = 'Hopper-v2'
		num_cpu= 4
		seed = 0

		set_random_seed(seed)
		self.vec_env = make_vec_env(env_id, n_envs=num_cpu,seed=seed)

		self.args_Model = OrderedDict([('policy', 'MlpPolicy'), 
                            ('env', self.vec_env),
                            ('verbose', 0),
                            ('top_quantiles_to_drop_per_net', 5),
                            ('learning_starts', 10000)])

		self.model = TQC(**self.args_Model)

	
	def act(self, curr_obs, mode='eval'):
		return self.model.predict(curr_obs)[0]
			
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep):

		self.model.learn(1000, reset_num_timesteps=False, 
			)
		self.model._last_obs = None

		self.model.set_env(self.vec_env)


	def save(self, root_path):		
		self.model.save(root_path+'save')

	def load_weights(self, root_path):
		if exists(root_path + "save"):
			print('There is not saved weights for this run')
		else:
			print('weights loaded')
			self.model = TQC.load(root_path+'save.zip', env=self.vec_env)
