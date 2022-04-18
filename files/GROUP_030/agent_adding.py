import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Process, Lock, Pool
from parfor import pmap
from joblib import Parallel, delayed
def f(self, a, tau=0.005): 
	a[1].data.copy_(tau*a[0] +(1-tau) *a[1])
	# return 0
device = torch.device("cuda:0")
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.curr_obs = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_obs = np.zeros((max_size, state_dim))
		self.done = np.zeros((max_size, 1))

		self.device = device

	def add(self, curr_obs, action, reward, next_obs, done):
		self.curr_obs[self.ptr] = curr_obs
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_obs[self.ptr] = next_obs
		self.done[self.ptr] = float(done)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.curr_obs[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_obs[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)


class Actor(nn.Module):
	def __init__(self, env_specs, neurons=100):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], neurons)	
		self.linear_relu_stack = nn.Sequential(
            nn.ReLU()
        )		
		self.l9 = nn.Linear(neurons, int(neurons/5))
		self.l10= nn.Linear(int(neurons/5), env_specs['action_space'].shape[0])
		self.max_action = float(env_specs['action_space'].high[0])

	def forward(self, state):
		q = F.relu(self.l1(state))
		q = self.linear_relu_stack(q)
		q = F.relu(self.l9(q))
		return self.max_action * torch.tanh(self.l10(q))


class Critic(nn.Module):
	def __init__(self, env_specs, neurons=100):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], neurons)
		self.linear_relu_stack = nn.Sequential(
            nn.Linear(neurons +env_specs['action_space'].shape[0] , neurons),
            nn.ReLU()
        )
		self.l9 = nn.Linear(neurons, int(neurons/5))
		self.l10= nn.Linear(int(neurons/5), 1)

	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = self.linear_relu_stack(torch.cat([q, action],1))
		q = F.relu(self.l9(q))
		return self.l10(q)


class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(self, env_specs, discount=0.98, tau=0.005, expl_noise=0.1, neurons=300):
		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space']
		self.act_space = env_specs['action_space']
		self.expl_noise = expl_noise
		self.replay_buffer = ReplayBuffer(self.obs_space.shape[0], self.act_space.shape[0])
		self.neurons = neurons

		self.actor = Actor(env_specs, self.neurons).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(env_specs, self.neurons).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)
    
		self.discount = discount
		self.tau = tau
		self.start_timesteps = 5000
		self.outputs = []
		self.timestep = 0
		print(self.actor)
		print(self.critic)
		self.addedL = 0
		self.addLayerFreq = 15000
		self.maxLayers = 5
		self.L1, self.L2 = [], []

		
	
	def act(self, curr_obs, mode='eval'):
		
		if mode=='train':
			if  self.timestep<self.start_timesteps:
				# print(self.timestep,1)
				return self.act_space.sample()
			else:
				# print(self.timestep,2)
				curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
				a = self.actor(curr_obs).cpu().data.numpy().flatten()
				b = self.act_space.sample()
				index = np.random.binomial(1, 1/3, self.act_space.shape[0])
				a[index] = .9*a[index] + .1*b[index]
				return a

		elif mode=='eval':
			curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
			# print(self.timestep,0)
			return (self.actor(curr_obs).cpu().data.numpy().flatten()
				).clip(-1, 1)
			
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=64):
		self.timestep = timestep


		self.replay_buffer.add(curr_obs, action, reward, next_obs, done)
		_curr_obs, _action, _reward, _next_obs, _done = self.replay_buffer.sample(batch_size)

		target_Q = self.critic_target(_next_obs, self.actor_target(_next_obs))
		target_Q = _reward + ((1-_done) * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(_curr_obs, _action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(_curr_obs, self.actor(_curr_obs)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# print(z for z in list(zip(self.critic.parameters(), self.critic_target.parameters())))
		# pmap(f, zip(self.critic.parameters(), self.critic_target.parameters()))
		# pmap(f, zip(self.actor.parameters(), self.actor_target.parameters()))
		# Parallel(n_jobs=5)(delayed(f)(a,b) for a,b in zip(self.critic.parameters(), self.critic_target.parameters()))
		# Parallel(n_jobs=5)(delayed(self.f)(z) for z in zip(self.actor.parameters(), self.actor_target.parameters()))
		# with Pool(4) as p:
		# 	p.map(self.f, zip(self.critic.parameters(), self.critic_target.parameters()))

		# with Pool(4) as p:
		# 	p.map(self.f, zip(self.actor.parameters(), self.actor_target.parameters()))
		# Update the frozen target models
		# for z in zip(self.critic.parameters(), self.critic_target.parameters()):
		# 	p = Process(target=f, args=z)
		# 	p.start
		# 	# self.process_list.append(p)
		# for z in zip(self.actor.parameters(), self.actor_target.parameters()):
		# 	p = Process(target=f, args=z)
		# 	p.start
		# 	# self.process_list.append(p)

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
		if int(timestep+5) % self.addLayerFreq == 0 and self.addedL<2*self.maxLayers:
			self.addLayer()
		



	def addLayer(self):
		if self.addedL % 2:
			self.L1.append(nn.Linear(in_features=self.neurons, out_features=self.neurons, bias=True).to(device))
			self.actor.linear_relu_stack.add_module('la'+str(self.addedL+1),self.L1[-1])

			self.L1.append(nn.ReLU().to(device))
			self.actor.linear_relu_stack.add_module('ra'+str(self.addedL+1),self.L1[-1])

			self.actor_target = copy.deepcopy(self.actor)
			self.actor_optimizer.add_param_group({'params':self.L1[-1].parameters()})
			self.actor_optimizer.add_param_group({'params':self.L1[-2].parameters()})
			print('One layer added to the actor')

		else:
			self.L2.append(nn.Linear(in_features=self.neurons, out_features=self.neurons, bias=True).to(device))
			self.critic.linear_relu_stack.add_module('la'+str(self.addedL+1),self.L2[-1])

			self.L2.append(nn.ReLU().to(device))
			self.critic.linear_relu_stack.add_module('ra'+str(self.addedL+1),self.L2[-1])

			# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)
			self.critic_target = copy.deepcopy(self.critic)
			self.critic_optimizer.add_param_group({'params':self.L2[-1].parameters()})
			self.critic_optimizer.add_param_group({'params':self.L2[-2].parameters()})
			# self.critic_optimizer.add_param_group({'param':self.critic.linear_relu_stack..parameters() })
			print('One layer added to the critic')
		self.addedL += 1


	def save(self, root_path):		
		# self.outputs.append([[_curr_obs.cpu(), _action.cpu(), _reward.cpu(), _next_obs.cpu(), _done.cpu()]])
		# print(self.outputs)
		# np.save(file_name + 'outputs', self.outputs)    
		torch.save(self.critic.state_dict(), root_path + "_critic")
		torch.save(self.critic_optimizer.state_dict(), root_path + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), root_path + "_actor")
		torch.save(self.actor_optimizer.state_dict(), root_path + "_actor_optimizer")

	def load_weights(self, root_path):
		try:
			self.critic.load_state_dict(torch.load(root_path + "_critic"))
			self.critic_optimizer.load_state_dict(torch.load(root_path + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)

			self.actor.load_state_dict(torch.load(root_path + "_actor"))
			self.actor_optimizer.load_state_dict(torch.load(root_path + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)
		except:
			pass