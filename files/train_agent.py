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
from os.path import isfile, join
import shutup; 
shutup.please()

from environments import JellyBeanEnv, MujocoEnv



import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map





def evaluate_agent(agent, env, n_episodes_to_evaluate, video_recorder, show_graphics):
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
      if show_graphics:
        env.render()
      # video_recorder.capture_frame()
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
                agentFile,
                show_graphics):

  f = open(agentFile+'.log', 'a')
  f.write(f'\n\n {agentFile} \n\n')
  f.write(f'timestep, acc_reward\n')
  agent.load_weights(f"./GROUP_030/"+agentFile)
  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  tf.random.set_random_seed(seed)
  env.seed(seed)
  env_eval.seed(seed)
  
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 
  # system("cls") # Windows
  
  enable_recorder = False
  video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, f"./GROUP_030/{agentFile}.mp4", enabled=show_graphics and enable_recorder)

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

      if show_graphics:
        env.render()
      video_recorder.capture_frame()
      timestep += 1
      if timestep % evaluation_freq == 0:
        # if timestep>=25e3:
          mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate, video_recorder, show_graphics)
          print(f'timestep: {timestep}, acc_reward: {mean_acc_rewards:.2f}')
          f.write(f'{timestep}, {mean_acc_rewards:.2f}\n')
          array_of_mean_acc_rewards.append(mean_acc_rewards)
          # print(get_gpu_memory_map())
          if max_acc_rewards<mean_acc_rewards:
            agent.save(f"./GROUP_030/"+agentFile)
            max_acc_rewards = mean_acc_rewards
        # else:
        #   print(timestep)
  
  video_recorder.close()
  torch.cuda.empty_cache()
  return array_of_mean_acc_rewards


if __name__ == '__main__':
    
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--group', type=str, default='GROUP_030', help='group directory')
  parser.add_argument('--agentFile', type=str, default='.agent', help='agent file')
  parser.add_argument('--showGraphics', action='store_true', help='should the simulation be rendered?')
  args = parser.parse_args()

  path = './'+args.group+'/'
  files = [f for f in listdir(path) if isfile(join(path, f))]
  if ('agent.py' not in files) or ('env_info.txt' not in files):
    print("Your GROUP folder does not contain agent.py or env_info.txt!")
    exit()

  with open(path+'env_info.txt') as f:
    lines = f.readlines()
  env_type = lines[0].lower()

  env = get_environment(env_type) 
  env_eval = get_environment(env_type)
  if 'jellybean' in env_type:
    env_specs = {'scent_space': env.scent_space, 'vision_space': env.vision_space, 'feature_space': env.feature_space, 'action_space': env.action_space}
  if 'mujoco' in env_type:
    env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
  agent_module = importlib.import_module(args.group+args.agentFile)
  agent = agent_module.Agent(env_specs)
  
  # Note these can be environment specific and you are free to experiment with what works best for you
  total_timesteps = 1000000
  evaluation_freq = 1000
  n_episodes_to_evaluate = 20

  learning_curve = train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, args.agentFile, args.showGraphics)

