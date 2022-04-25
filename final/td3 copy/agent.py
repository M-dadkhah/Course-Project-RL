import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")
class ReplayBuffer(object):
	
	'''
	History of the environment and the policy
	'''
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr  	  = 0
		self.size     = 0

		self.curr_obs = np.zeros((max_size, state_dim ))
		self.action	  = np.zeros((max_size, action_dim))
		self.reward   = np.zeros((max_size, 1         ))
		self.next_obs = np.zeros((max_size, state_dim ))
		self.done     = np.zeros((max_size, 1         ))

		self.device   = torch.device("cuda:0")

	def add(self, curr_obs, action, reward, next_obs, done):
		self.curr_obs[self.ptr] = curr_obs
		self.action[self.ptr]   = action
		self.reward[self.ptr]   = reward
		self.next_obs[self.ptr] = next_obs
		self.done[self.ptr]     = float(done)

		# if self.size >= self.max_size:
		# self.ptr  = np.random.randint(self.max_size)
		# else:
		self.ptr  = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.curr_obs[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]  ).to(self.device),
			torch.FloatTensor(self.reward[ind]  ).to(self.device),
			torch.FloatTensor(self.next_obs[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]    ).to(self.device)
		)


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
		self.replay_buffer = ReplayBuffer(self.obs_space.shape[0], self.act_space.shape[0])

		self.actor = Actor(env_specs).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(env_specs).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)
    
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
		# if mode=='train' and self.timestep<self.start_timesteps:
		# 	return self.act_space.sample()
			
		curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
		action = self.actor(curr_obs).cpu().data.numpy().flatten()

		if mode=='train':
			self.expl_noise = 0.3*(1-np.tanh(self.timestep/3.5e5))
			action = (action \
				+ np.random.normal(0, 1 * self.expl_noise, size=self.act_space.shape[0])
				).clip(-1, 1)
		return action
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=int(2**8)):
		self.timestep = timestep
		self.replay_buffer.add(curr_obs, action, reward, next_obs, done) # adding the observation to the buffer
		_curr_obs, _action, _reward, _next_obs, _done = self.replay_buffer.sample(batch_size) # sampling the buffer

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(_action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(_next_obs) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(_next_obs, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = _reward + (1-_done) * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(_curr_obs, _action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.timestep % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(_curr_obs, self.actor(_curr_obs)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
	


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
			# self.critic_optimizer.load_state_dict(torch.load(root_path + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)
			# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)
			self.critic_optimizer.load_state_dict(torch.load(root_path + "_critic_optimizer"))
			# self.critic_target = copy.deepcopy(self.critic)
			
			# self.actor.load_state_dict(torch.load(root_path + "_actor"))
			self.actor.load_state_dict(torch.load(root_path + "_actor"))
			# self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
			self.actor_optimizer.load_state_dict(torch.load(root_path + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)
		except:
			pass