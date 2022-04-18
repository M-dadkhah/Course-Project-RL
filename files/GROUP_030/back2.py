import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5), steps=3):
		self.max_size = max_size
		self.steps=steps
		self.ptr = steps-1
		self.size = steps-1

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

	def getLast(self):
		return (
			self.action[self.size-self.steps+1:self.ptr].flatten(),
			self.curr_obs[self.size-self.steps+1:self.ptr].flatten()
		)	

	def sample(self, batch_size):
		try:
			ind = np.random.randint(0, self.size-self.steps, size=batch_size)
		except:
			print(1)
			ind = np.random.randint(0, self.size-self.steps+1, size=batch_size)

		cur_ = np.array([self.curr_obs[i:i+self.steps].flatten() for i in ind])
		# act_ = np.array([self.action[i:i+self.steps].flatten() for i in ind])
		nexts_ = np.array([self.next_obs[i:i+self.steps].flatten() for i in ind])
		return (
			torch.FloatTensor(self.curr_obs[ind+self.steps]).to(self.device),
			torch.FloatTensor(cur_).to(self.device),
			torch.FloatTensor(self.action[ind+self.steps]).to(self.device),
			torch.FloatTensor(self.reward[ind+self.steps]).to(self.device),
			torch.FloatTensor(nexts_).to(self.device),
			torch.FloatTensor(self.next_obs[ind+self.steps]).to(self.device),
			torch.FloatTensor(self.done[ind+self.steps]).to(self.device)
		)


class Actor(nn.Module):
	def __init__(self, env_specs, neurons=100, steps=3):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(env_specs['observation_space'].shape[0]*steps, 400)

		self.l2 = nn.Linear(400 , 300)
		# self.n1 = nn.LayerNorm(neurons)

		# self.dropout = nn.Dropout(0.5)
		self.l3 = nn.Linear(300, 100)
		self.l4 = nn.Linear(100, env_specs['action_space'].shape[0] )
		self.max_action = float(env_specs['action_space'].high[0])

	def forward(self, state):			
		q = self.l1(state)
		q = F.relu(q)


		q = self.l2(q)
		q = F.relu(q)
		# q = self.n1(q)

		q = self.l3(q)
		# q = self.n3(q)
		q = F.relu(q)
		# a = self.dropout(a)
		return self.max_action * torch.tanh(self.l4(q))



class Critic(nn.Module):
	def __init__(self, env_specs, neurons=100):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(env_specs['observation_space'].shape[0], 400)


		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300 + env_specs['action_space'].shape[0], 200)
		# self.n1 = nn.LayerNorm(int(neurons/2))
		# self.n1 = nn.LayerNorm(int(neurons/2))

		self.l4 = nn.Linear(200, 50)
		# self.dropout = nn.Dropout(0.5)
		self.l5 = nn.Linear(50, 1)

	def forward(self, state, action):
		q = self.l1(state)
		q = F.relu(q)
		# q = self.n1(q)



		q = self.l2(q)
		q = F.relu(q)
		# q = self.n1(q)
		q = self.l3(torch.cat((q, action), 1))
		q = F.relu(q)

		q = self.l4(q)
		# q = self.n3(q)
		q = F.relu(q)

		# q = self.dropout(q)
		# print(q.size())
		# print(action.size())
		# print(torch.cat((q, action), 1).size())
		return self.l5(q)


class Agent(object):
	'''The agent class that is to be filled.
		You are allowed to add any method you
		want to this class.
	'''

	def __init__(self, env_specs, discount=0.99, tau=0.005, expl_noise=0.1, steps=2):
		super(Agent, self).__init__()
		self.env_specs = env_specs 
		self.obs_space = env_specs['observation_space']
		self.act_space = env_specs['action_space']
		self.expl_noise = expl_noise
		self.steps=steps
		self.replay_buffer = ReplayBuffer(self.obs_space.shape[0], self.act_space.shape[0], steps=self.steps)

		self.actor = Actor(env_specs, steps=self.steps).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(env_specs).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)
    
		self.discount = discount
		self.tau = tau
		self.start_timesteps = 5e3
		self.outputs = []
		self.timestep = 0
    

		
	
	def act(self, curr_obs, mode='eval'):
		
		if mode=='train':
			if self.timestep<self.start_timesteps:
				# print(self.timestep,1)
				return self.act_space.sample()
			else:
				# print(self.timestep,2)
				actions, curr_obss = self.replay_buffer.getLast()
				# print(curr_obss.shape)
				curr_obs = torch.FloatTensor(np.append(curr_obss,curr_obs).flatten()).to(device)
				# print(curr_obs.size())
				return (self.actor(curr_obs).cpu().data.numpy().flatten() \
					+ np.random.normal(0, 1 * self.expl_noise, size=self.act_space.shape[0])
					).clip(-1, 1)
		elif mode=='eval':
			actions, curr_obss = self.replay_buffer.getLast()
			# curr_obss.size()
			curr_obs = torch.FloatTensor(np.append(curr_obss,curr_obs).reshape(1, -1)).to(device)
			# curr_obs.size()
			# print(self.timestep,0)
			return (self.actor(curr_obs).cpu().data.numpy().flatten()
				).clip(-1, 1)
			
		
	def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=64):
		self.timestep = timestep
		self.replay_buffer.add(curr_obs, action, reward, next_obs, done)
		# if timestep>= self.start_timesteps:
		_curr_obs, _curr_obss, _action, _reward, _next_obss,_next_obs, _done = self.replay_buffer.sample(batch_size)

		target_Q = self.critic_target(_next_obs, self.actor_target(_next_obss))
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
		actor_loss = -self.critic(_curr_obs, self.actor(_curr_obss)).mean()
		
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
			self.critic_optimizer.load_state_dict(torch.load(root_path + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)

			self.actor.load_state_dict(torch.load(root_path + "_actor"))
			self.actor_optimizer.load_state_dict(torch.load(root_path + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)
		except:
			pass