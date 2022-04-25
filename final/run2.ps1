cd 'E:\mostafa\PhD\COMP579\project\codes\final'
# for ($i = 1 ; $i -le 5 ; $i++) {
    for ($seed = 1 ; $seed -le 3 ; $seed++) {
        python '.\train_agent.py' --algo sac --seed $seed
        # python '.\train_agent.py' --algo ddpg --seed $seed
    }
# }

# python '.\anim.py' --algo td3Anim --seed 0