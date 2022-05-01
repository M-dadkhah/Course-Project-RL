cd 'E:\mostafa\PhD\COMP579\project\codes\final'
for ($seed = 0 ; $seed -le 1 ; $seed++) {
    python '.\train_agent.py' --algo ddpg --seed $seed --mode e
    # python '.\train_agent.py' --algo sac --seed $seed --mode e
    python '.\train_agent.py' --algo td3 --seed $seed --mode e
}




