import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import exists
device = torch.device("cuda:0")
class ReplayBuffer(object):

	'''
	History of the environment and the policy
	'''
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
	Actor agent,
	input observations and actions, output the Q value
	3 dense layes
	'''
	def __init__(self, env_specs, neurons=int(2**8)):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], neurons)
		self.l2 = nn.Linear(neurons + env_specs['action_space'].shape[0], neurons)
		self.l3 = nn.Linear(neurons , neurons)
		self.l4 = nn.Linear(neurons, 1)

	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		q = F.relu(self.l3(q))
		return self.l4(q)


class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(self,
	 env_specs,
	 logpath=None,
	 discount=0.99, 
	 tau=0.005, 
	 ):
		super(Agent, self).__init__()
		self.logpath=logpath
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
		self.start_timesteps = int(5e3)
		self.outputs = []
		self.timestep = 0
		self.policy_freq = 2
		self.log_freq = int(3e3)
    

		
	
	def act(self, curr_obs, mode='eval'):
		if mode=='train' and self.timestep<self.start_timesteps:
			return self.act_space.sample()
			
		curr_obs = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
		action = self.actor(curr_obs).cpu().data.numpy().flatten()

		if mode=='train':
			self.expl_noise = 0.3*(1-np.tanh(self.timestep/1.4e6))
			action = (action \
				+ np.random.normal(0, 1 * self.expl_noise, size=self.act_space.shape[0])
				).clip(-1, 1)
		return action
			
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=int(2**7)):
		self.timestep = timestep
		self.replay_buffer.add(curr_obs, action, reward, next_obs, done) # adding the observation to the buffer
		if timestep> self.start_timesteps:
			_curr_obs, _action, _reward, _next_obs, _done = self.replay_buffer.sample(batch_size) # sampling the buffer

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



			if self.timestep % self.policy_freq == 0:
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
				
				if self.logpath is not None:
					if timestep % self.log_freq == 0:
						log = [net.cpu().data.numpy().flatten()[0] for net in [actor_loss, critic_loss]]
						with open( self.logpath + 'results_loss.log', 'a+') as f:
							f.write(
								f'timestep: {timestep}, actor_loss: {log[0]}, ' + \
								f'critic_loss: {log[1]}	\n')


	def save(self, root_path):		
		# self.outputs.append([[_curr_obs.cpu(), _action.cpu(), _reward.cpu(), _next_obs.cpu(), _done.cpu()]])
		# print(self.outputs)
		# np.save(file_name + 'outputs', self.outputs)    
		torch.save(self.critic.state_dict(), root_path + "W_critic")
		torch.save(self.critic_optimizer.state_dict(), root_path + "W_critic_optimizer")
		
		torch.save(self.actor.state_dict(), root_path + "W_actor")
		torch.save(self.actor_optimizer.state_dict(), root_path + "W_actor_optimizer")

	def load_weights(self, root_path):
		if not exists(root_path + "W_critic"):
			print(f'There is not saved weights for this run {root_path + "W_critic"}')
		else:			
			print('weights loaded')
			self.critic.load_state_dict(torch.load(root_path + "W_critic"))
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)
			# self.critic_optimizer.load_state_dict(torch.load(root_path + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)
			
			self.actor.load_state_dict(torch.load(root_path + "W_actor"))
			self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
			# self.actor_optimizer.load_state_dict(torch.load(root_path + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)
			

