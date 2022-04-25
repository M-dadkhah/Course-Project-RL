import matplotlib.pyplot as plt
import csv
import numpy as np

label = [
'agent',
'agent_adding',
'agent_td3',
'agent_twoact',
'back2'
]

filename = []
for l in label:
    filename.append('.{}.log'.format(l))

for it in range(len(label)):

    timesteps = []
    acc_reward = []

    with open(filename[it], 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        next(reader)
        for row in reader:
            timesteps.append(float(row[0]))
            acc_reward.append(float(row[1]))
            
    plt.plot(timesteps, acc_reward, label=label[it])
    plt.xlabel('timestep')
    plt.ylabel('acc_reward')

plt.legend(prop={'size': 6})
plt.title('accumulated rewards')
plt.savefig('accumulated rewards.pdf')
plt.show()

for it in range(len(label)):

    timesteps = []
    acc_reward = []

    with open(filename[it], 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        next(reader)
        for row in reader:
            timesteps.append(float(row[0]))
            acc_reward.append(float(row[1]))
            
    plt.plot(timesteps, acc_reward, label=label[it])
    plt.xlabel('timestep')
    plt.ylabel('acc_reward')

    plt.title('{}'.format(label[it]))
    plt.savefig('{}.pdf'.format(label[it]))
    plt.show()