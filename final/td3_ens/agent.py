import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")


class Actor(nn.Module):
	'''
	Actor agent,
	input observations, output action
	3 dense layes
	'''
	def __init__(self, env_specs, neurons=int(2**8)):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], neurons)
		self.l2 = nn.Linear(neurons, neurons)
		self.l3 = nn.Linear(neurons, env_specs['action_space'].shape[0] )
		self.max_action = float(env_specs['action_space'].high[0])

	def forward(self, state):	
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	'''
	Critic agent,
	inputs observations and actions, output w Q values for each critic agent.
	Each network has 3 dense layers.
	'''
	def __init__(self, env_specs, neurons=int(2**8)):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(env_specs['observation_space'].shape[0] + \
            env_specs['action_space'].shape[0], neurons)
		self.l2 = nn.Linear(neurons , neurons)
		self.l3 = nn.Linear(neurons, 1)

		self.l4 = nn.Linear(env_specs['observation_space'].shape[0] + \
            env_specs['action_space'].shape[0], neurons)
		self.l5 = nn.Linear(neurons , neurons)
		self.l6 = nn.Linear(neurons, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(
		self, 
		env_specs, 
		discount=0.99, 
		tau=0.005, 
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
		):

		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space']
		self.act_space = env_specs['action_space']

		self.actors = [Actor(env_specs).to(device) for _ in range(4)]

		self.critics = [Critic(env_specs).to(device) for _ in range(4)]
    
		self.discount = discount
		self.tau = tau
		self.start_timesteps = 0
		self.outputs = []
		self.timestep = 0
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.max_action = float(env_specs['action_space'].high[0])
		self.start_timesteps = int(5e3)
		
	
	def act(self, curr_obs, mode='eval'):
		curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
		actions = np.array([actor(curr_obs).cpu().data.numpy().flatten() for actor in self.actors])
		# print(actions+1)

		# print(np.prod(actions+1,axis=0)**(1/len(actions))-1)
		# return np.prod(actions+1,axis=0)**(1/len(actions))-1
		d=[]
		size=4
		for i in range(size):
			for j in range(i+1,size):
				d.append((i,j,sum((actions[i]-actions[j])**2)))

		args = min(d,key = lambda x: x[2])
		action = (actions[args[0]] + actions[args[1]])/2
		return action
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=int(2**8)):
		pass
	


	def save(self, root_path):		
		pass


	def load_weights(self, root_path):
		try:
			for i in range(4):
				self.critics[i].load_state_dict(torch.load(root_path + f'seed_{i}/W' + "_critic"))
				
				# self.actor.load_state_dict(torch.load(root_path + "_actor"))
				self.actors[i].load_state_dict(torch.load(root_path + f'seed_{i}/W' + "_actor"))
		except:
			pass
