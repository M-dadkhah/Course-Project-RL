# powershell -file 'E:\mostafa\PhD\COMP579\project\codes\COMP579-Project-Template\run.ps1'
import gym
import argparse
import importlib
import time
import random
import numpy as np

import tensorflow as tf
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from os import listdir, makedirs, system
from os.path import exists, isfile, join

from environments import JellyBeanEnv, MujocoEnv



import subprocess



def evaluate_agent(agent, env, n_episodes_to_evaluate):
  '''Evaluates the agent for a provided number of episodes.'''
  array_of_acc_rewards = []
  for _ in range(n_episodes_to_evaluate):
    acc_reward = 0
    done = False
    curr_obs = env.reset()
    while not done:
      action = agent.act(curr_obs, mode='eval')
      # print(action)
      next_obs, reward, done, _ = env.step(action)
      env.render()
      acc_reward += reward
      curr_obs = next_obs
    array_of_acc_rewards.append(acc_reward)
  return np.mean(np.array(array_of_acc_rewards))


def get_environment(env_type):
  '''Generates an environment specific to the agent type.'''
  if 'jellybean' in env_type:
    env = JellyBeanEnv(gym.make('JBW-COMP579-obj-v1'))
  elif 'mujoco' in env_type:
    env = MujocoEnv(gym.make('Hopper-v2'))
  else:
    raise Exception("ERROR: Please define your env_type to be either 'jellybean' or 'mujoco'!")
  return env


def train_agent(agent,
                env,
                env_eval,
                total_timesteps,
                evaluation_freq,
                n_episodes_to_evaluate,
                algo,
                seed):
  
  path = f'./{algo}/runs/seed_{seed}/'

  if not exists(path):
    
    # Create a new directory because it does not exist 
    makedirs(path)


  f = open(path+f'log.log', 'a')
  f.write(f'\n\n {algo} _ seed: {seed} \n\n')
  agent.load_weights(path+'W')
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  tf.compat.v1.random.set_random_seed(seed)
  env.seed(seed)
  env_eval.seed(seed)
  
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 
  # system("cls") # Windows

  timestep = 0
  array_of_mean_acc_rewards = []
  max_acc_rewards = 0
  while timestep < total_timesteps:

    done = False
    curr_obs = env.reset()
    while not done:    
      action = agent.act(curr_obs, mode='train')
      next_obs, reward, done, _ = env.step(action)
      agent.update(curr_obs, action, reward, next_obs, done, timestep)
      curr_obs = next_obs
        
      timestep += 1
      if timestep % evaluation_freq == 0:
        # if timestep>=25e3:
          mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
          print(f'timestep: {timestep}, acc_reward: {mean_acc_rewards:.2f}')
          f.write(f'timestep: {timestep}, acc_reward: {mean_acc_rewards:.2f}\n')
          array_of_mean_acc_rewards.append(mean_acc_rewards)
          # print(get_gpu_memory_map())
          if max_acc_rewards<mean_acc_rewards:
            agent.save(path+'W')
            max_acc_rewards = mean_acc_rewards
        # else:
        #   print(timestep)
  torch.cuda.empty_cache()
  return array_of_mean_acc_rewards


if __name__ == '__main__':
    
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--algo', type=str, default='td3', help='group directory')
  parser.add_argument('--seed', type=int, default=0, help='agent file')
  args = parser.parse_args()

  path = './'+args.algo+'/'
  files = [f for f in listdir(path) if isfile(join(path, f))]
  if ('agent.py' not in files) or ('env_info.txt' not in files):
    print("Your GROUP folder does not contain agent.py or env_info.txt!")
    exit()

  with open(path+'env_info.txt') as f:
    lines = f.readlines()
  env_type = lines[0].lower()
  total_timesteps = int(2e3)
  # total_timesteps=1
  evaluation_freq = 1000
  n_episodes_to_evaluate = 1


  
  # Note these can be environment specific and you are free to experiment with what works best for you

  agent_module = importlib.import_module(args.algo+'.agent')
  try:
    learning_curve=np.load(path+f'runs/seed_{args.seed}/result'+ str(args.seed) +'.npy')
  except:
    learning_curve=np.array([])
  env = get_environment(env_type) 
  env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
  env_eval = get_environment(env_type)
  agent = agent_module.Agent(env_specs)
  learning_curve = np.append(learning_curve,train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, args.algo, args.seed))
  np.save(path+f'runs/seed_{args.seed}/result'+ str(args.seed) +'.npy', learning_curve)

