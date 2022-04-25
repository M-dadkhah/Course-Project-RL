'''
Soft Actor-Critic version 1
using state value function: 1 V net, 1 target V net, 2 Q net, 1 policy net
paper: https://arxiv.org/pdf/1801.01290.pdf
'''


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
		
	def sample(self, batch_size):
		
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
		''' 
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		return state, action, reward, next_state, done
	
	def __len__(self):
		return len(self.buffer)

class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim, activation=F.relu, init_w=3e-3):
		super(ValueNetwork, self).__init__()
		
		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)
		# weights initialization
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

		self.activation = activation
		
	def forward(self, state):
		x = self.activation(self.linear1(state))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)
		return x
		
		
class SoftQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3):
		super(SoftQNetwork, self).__init__()
		
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)
		
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

		self.activation = activation
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1) # the dim 0 is number of samples
		x = self.activation(self.linear1(x))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)
		return x
		
		
class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3, log_std_min=-20, log_std_max=2):
		super(PolicyNetwork, self).__init__()
		
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, hidden_size)

		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)
		
		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

		self.num_actions = num_actions
		self.activation = activation
		self.action_range = 10.
		
	def forward(self, state):
		x = self.activation(self.linear1(state))
		x = self.activation(self.linear2(x))
		x = self.activation(self.linear3(x))
		x = self.activation(self.linear4(x))

		mean	= (self.mean_linear(x))
		# mean	= F.leaky_relu(self.mean_linear(x))
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		
		return mean, log_std


class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(
		self, 
		env_specs, 
		discount=0.99, 
		tau=0.01, 
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_dim = 512):
		
		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space'].shape[0]
		self.act_space = env_specs['action_space'].shape[0]

		self.value_net		= ValueNetwork(self.obs_space, hidden_dim, activation=F.relu).to(device)
		self.target_value_net = ValueNetwork(self.obs_space, hidden_dim, activation=F.relu).to(device)

		self.soft_q_net1 = SoftQNetwork(self.obs_space, self.act_space, hidden_dim, activation=F.relu).to(device)
		self.soft_q_net2 = SoftQNetwork(self.obs_space, self.act_space, hidden_dim, activation=F.relu).to(device)
		self.policy_net = PolicyNetwork(self.obs_space, self.act_space, hidden_dim, activation=F.relu).to(device)

		print('(Target) Value Network: ', self.value_net)
		print('Soft Q Network (1,2): ', self.soft_q_net1)
		print('Policy Network: ', self.policy_net)

		for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
			target_param.data.copy_(param.data)


		self.value_criterion  = nn.MSELoss()
		self.soft_q_criterion1 = nn.MSELoss()
		self.soft_q_criterion2 = nn.MSELoss()
		self.discount = discount 


		self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=3e-4)
		self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
		self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)

		self.replay_buffer = ReplayBuffer(int(1e6))
		self.tau = tau
		self.start_timesteps = int(5e3)
		
	
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=int(2**8)):
		self.replay_buffer.push(curr_obs, action, reward, next_obs, done) # adding the observation to the buffer
		if timestep> 400:
			alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)
			state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
			# print('sample:', state, action,  reward, done)

			state	  = torch.FloatTensor(state).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			action	 = torch.FloatTensor(action).to(device)
			reward	 = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
			done	   = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

			predicted_q_value1 = self.soft_q_net1(state, action)
			predicted_q_value2 = self.soft_q_net2(state, action)
			predicted_value	= self.value_net(state)
			new_action, log_prob, z, mean, log_std = self.evaluate(state)

			reward = (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

		# Training Q Function
			target_value = self.target_value_net(next_state)
			target_q_value = reward + (1 - done) * self.discount * target_value # if done==1, only reward
			q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
			q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


			self.soft_q_optimizer1.zero_grad()
			q_value_loss1.backward()
			self.soft_q_optimizer1.step()
			self.soft_q_optimizer2.zero_grad()
			q_value_loss2.backward()
			self.soft_q_optimizer2.step()  

		# Training Value Function
			predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
			target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
			value_loss = self.value_criterion(predicted_value, target_value_func.detach())

			
			self.value_optimizer.zero_grad()
			value_loss.backward()
			self.value_optimizer.step()

		# Training Policy Function
			''' implementation 1 '''
			policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
			''' implementation 2 '''
			# policy_loss = (alpha * log_prob - self.soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
			''' implementation 3 '''
			# policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

			''' implementation 4 '''  # version of github/higgsfield
			# log_prob_target=predicted_new_q_value - predicted_value
			# policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
			# mean_lambda=1e-3
			# std_lambda=1e-3
			# mean_loss = mean_lambda * mean.pow(2).mean()
			# std_loss = std_lambda * log_std.pow(2).mean()
			# policy_loss += mean_loss + std_loss


			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()
			
			# print('value_loss: ', value_loss)
			# print('q loss: ', q_value_loss1, q_value_loss2)
			# print('policy loss: ', policy_loss )


		# Soft update the target value net
			for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
				target_param.data.copy_(  # copy data value into target parameters
					target_param.data * (1.0 - self.tau) + param.data * self.tau
				)
			return predicted_new_q_value.mean()


	def evaluate(self, state, mode = 'train', epsilon=1e-6):
		'''
		generate sampled action with state as input wrt the policy network;
		deterministic evaluation provides better performance according to the original paper;
		'''
		mean, log_std = self.policy_net.forward(state)
		std = log_std.exp() # no clip in evaluation, clip affects gradients flow
		
		normal = Normal(0, 1)
		z	  = normal.sample(mean.shape) 
		action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
		action = self.policy_net.action_range*action_0
		''' stochastic evaluation '''
		log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.policy_net.action_range)
		''' deterministic evaluation '''
		# log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
		'''
		 both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
		 the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
		 needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
		 '''
		log_prob = log_prob.sum(dim=-1, keepdim=True)
		return action, log_prob, z, mean, log_std
		
	def act(self, curr_obs, mode='eval'):
		curr_obs = torch.FloatTensor(curr_obs).unsqueeze(0).to(device)
		mean, log_std = self.policy_net.forward(curr_obs)
		std = log_std.exp()
		
		normal = Normal(0, 1)
		z	  = normal.sample(mean.shape).to(device)
		action = self.policy_net.action_range* torch.tanh(mean + std*z)		
		action = action.detach().cpu().numpy()[0]
		
		return action

	def save(self, root_path):		
		torch.save(self.policy_net.state_dict(), root_path + "_policy")

	def load_weights(self, root_path):
		try:
			self.policy_net.load_state_dict(torch.load(root_path + "_policy"))
		except:
			pass


