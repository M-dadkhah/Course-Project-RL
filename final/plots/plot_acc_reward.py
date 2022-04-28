import matplotlib.pyplot as plt
import csv
import numpy as np
import glob

label = [
'ddpg',
'sac',
'sac2',
'td3',
'td3_ens'
]

#####################################################################
####################### one plot for all agents ##########################
#####################################################################

filename = []
for l in label:
    log_files = glob.glob("../" + l + "/runs/**/log.log", recursive = True)
    
    timesteps = np.array([])
    acc_reward = np.array([[]])
    first = True

    # loop over all log files for an agent
    for j, log in enumerate(log_files):
        print(log)
        
        # determine the number of rows in the file
        file_length = 0
        with open(log, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            file_length = sum(1 for row in reader) - 4 #
            
        # read the log file, skip first 4 rows
        with open(log, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            
            # resize the numpy array in the first run according to the number of rows in the file
            if first:
                timesteps = np.resize(timesteps, (file_length))
                acc_reward = np.resize(acc_reward, (file_length, len(log_files)))
                first = False
            
            # read the timestep and reward values
            for i, row in enumerate(reader):
                timesteps[i] = float(row[0][9:])
                acc_reward[i][j] = float(row[1][12:])
                
    # compute mean and standard deviation for an agent
    mean = np.mean(acc_reward, axis=1)
    std = np.std(acc_reward, axis=1)
    alpha = 0.5 # scale the standard deviation
    plt.plot(timesteps, mean, linewidth=1 , label=l)
    plt.fill_between(timesteps, mean- alpha*std, mean+ alpha*std, alpha = 0.2)
    plt.xlabel('timestep')
    plt.ylabel('accumulated rewards')

plt.legend(prop={'size': 6})
plt.title('accumulated rewards')
plt.savefig('accumulated rewards.pdf')
plt.show()


#####################################################################
####################### one plot per agent ##########################
#####################################################################

filename = []
for l in label:
    log_files = glob.glob("../" + l + "/runs/**/log.log", recursive = True)
    
    timesteps = np.array([])
    acc_reward = np.array([[]])
    first = True

    for j, log in enumerate(log_files):
        print(log)
        
        # determine the number of rows in the file
        file_length = 0
        with open(log, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            file_length = sum(1 for row in reader) - 4 #
            
        with open(log, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            
            # resize the numpy array in the first run according to the number of rows in the file
            if first:
                timesteps = np.resize(timesteps, (file_length))
                acc_reward = np.resize(acc_reward, (file_length, len(log_files)))
                first = False
            for i, row in enumerate(reader):
                timesteps[i] = float(row[0][9:])
                acc_reward[i][j] = float(row[1][12:])
                
    mean = np.mean(acc_reward, axis=1)
    std = np.std(acc_reward, axis=1)
    alpha = 0.5 # scale the standard deviation
    plt.plot(timesteps, mean, linewidth=1 , label=l)
    plt.fill_between(timesteps, mean- alpha*std, mean+ alpha*std, alpha = 0.2)
    plt.xlabel('timestep')
    plt.ylabel('accumulated rewards')

    plt.title('{}'.format(l))
    plt.savefig('{}.pdf'.format(l))
    plt.show()