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



def evaluate_agent(agent, env, n_episodes_to_evaluate, render=False):
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
      acc_reward += reward
      curr_obs = next_obs
      if render:
        env.render()
    array_of_acc_rewards.append(acc_reward)
  mm = np.mean(np.array(array_of_acc_rewards))
  std = np.std(np.array(array_of_acc_rewards))
  return mm, std


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

  # with open(path+f'log.log', 'a+') as f:
  #   f.write(f'\n\n {algo} _ seed: {seed} \n\n')
  # agent.load_weights(path+'W')
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
  array_of_std_acc_rewards = []
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
          mean_acc_rewards, std_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
          string=f'timestep: {timestep}, acc_reward: {mean_acc_rewards:.2f}, std: {std_acc_rewards:.2f}'
          with open(path+f'log.log', 'a') as f:
            f.write(string)
          print(string)
          array_of_mean_acc_rewards.append(mean_acc_rewards)
          array_of_std_acc_rewards.append(std_acc_rewards)
          # print(get_gpu_memory_map())
          if max_acc_rewards<mean_acc_rewards:
            # agent.save(path)
            print('saved')
            max_acc_rewards = mean_acc_rewards
        # else:
        #   print(timestep)
  torch.cuda.empty_cache()
  return array_of_mean_acc_rewards, array_of_std_acc_rewards


if __name__ == '__main__':
    
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--algo', type=str, default='td3', help='group directory')
  parser.add_argument('--seed', type=int, default=0, help='agent file')
  parser.add_argument('--mode', type=str, default='e', help='[e] evaluate, [t] train (default: evaluate)')
  parser.add_argument('--render', type=str, default='r', help='[on]/[off] (default: off)')
  args = parser.parse_args()

  if args.mode is not 'e':
    args.render = 'off'
  path = './'+args.algo+'/'
  files = [f for f in listdir(path) if isfile(join(path, f))]
  if ('agent.py' not in files) or ('env_info.txt' not in files):
    print("Your GROUP folder does not contain agent.py or env_info.txt!")
    exit()

  with open(path+'env_info.txt') as f:
    lines = f.readlines()
  env_type = lines[0].lower()
  total_timesteps = int(2e6)
  # total_timesteps=1
  evaluation_freq = 1000
  if args.algo in ['sac', 'tqc']:
    evaluation_freq = 1 
  n_episodes_to_evaluate = 10
  
  # Note these can be environment specific and you are free to experiment with what works best for you

  agent_module = importlib.import_module(args.algo+'.agent')

  env = get_environment(env_type) 
  env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
  env_eval = get_environment(env_type)
  agent = agent_module.Agent(env_specs, path+f'runs/seed_{args.seed}/')

  print(args.render=='on')
  if args.mode=='e':
    print(f'start, [ mode: elvauate ]')
    agent.load_weights(path+f'runs/seed_{args.seed}/')
    for _ in range(1):
      mean, std = evaluate_agent(agent, env, n_episodes_to_evaluate, render=args.render=='on')
      print(f'{_}:= mean: {mean:.2f}, std: {std:.2f}')
  else :
    print(f'start, [ mode= training ]')
    train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, args.algo, args.seed)

