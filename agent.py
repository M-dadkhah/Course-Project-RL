import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.curr_obs = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_obs = np.zeros((max_size, state_dim))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda:0")

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
		self.l2 = nn.Linear(neurons, neurons)
		self.l3 = nn.Linear(neurons, env_specs['action_space'].shape[0] )
		self.max_action = float(env_specs['action_space'].high[0])

	def forward(self, state):	
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, env_specs, neurons=100):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], neurons)
		self.l2 = nn.Linear(neurons + env_specs['action_space'].shape[0], neurons)
		self.l3 = nn.Linear(neurons, 1)

	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(self, env_specs, discount=0.99, tau=0.005, expl_noise=0.1):
		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space']
		self.act_space = env_specs['action_space']
		self.expl_noise = expl_noise
		self.replay_buffer = ReplayBuffer(self.obs_space.shape[0], self.act_space.shape[0])

		self.actor = Actor(env_specs).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(env_specs).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)
    
		self.discount = discount
		self.tau = tau
		self.start_timesteps = 25e3
		self.outputs = []
		self.timestep = 0
    
	
	def load_weights(self, root_path):
		pass
	
	def act(self, curr_obs, mode='eval'):
		if self.timestep<self.start_timesteps:
			return self.act_space.sample()
		else:
			curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
			if mode=='train':
				return (self.actor(curr_obs).cpu().data.numpy().flatten() \
					+ np.random.normal(0, 1 * self.expl_noise, size=self.act_space.shape[0])
					).clip(-1, 1)
			elif mode=='eval':
				return (self.actor(curr_obs).cpu().data.numpy().flatten()
					).clip(-1, 1)
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=256):
		self.timestep = timestep
		self.replay_buffer.add(curr_obs, action, reward, next_obs, done)
		if timestep>= self.start_timesteps:
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

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 			if (timestep + 1) % 1000 == 0:		
# 				file_name = f"./GROUP_030/"
# 				# self.outputs.append([[_curr_obs.cpu(), _action.cpu(), _reward.cpu(), _next_obs.cpu(), _done.cpu()]])
# 				# print(self.outputs)
# 				# np.save(file_name + 'outputs', self.outputs)    
# 				torch.save(self.critic.state_dict(), file_name + "_critic")
# 				torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")
				
# 				torch.save(self.actor.state_dict(), file_name + "_actor")
# 				torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")
