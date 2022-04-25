cd 'E:\mostafa\PhD\COMP579\project\codes\final'
# for ($i = 1 ; $i -le 5 ; $i++) {
    for ($seed = 0 ; $seed -le 3 ; $seed++) {
        python '.\train_agent.py' --algo td3 --seed $seed
        # python '.\train_agent.py' --algo ddpg --seed $seed
    }
# }

# python '.\anim.py' --algo td3Anim --seed 0