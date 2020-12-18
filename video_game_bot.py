import gym
import universe # register universe environment
import random

env = gym.make('flashgames.NeonRace-v0')
#env.configure(remotes=1) #automatically creates a local docker container
observation_n = env.reset()

# Move left
left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True),
('KeyEvent', 'ArrowRight', False)]
#Move right
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False),
('KeyEvent', 'ArrowRight', True)]
# Move forward
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowRight', False),
('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'n', True)]

# We use turn variable for deciding whether to turn or not
turn = 0
# We store all the rewards in rewards list
rewards = []
#we will use buffer as some threshold
buffer_size = 100
#we will initially set action as forward, which just move the car forward #without any turn
action = forward

while True:
    turn -= 1
# Let us say initially we take no turn and move forward.
# We will check value of turn, if it is less than 0
# then there is no necessity for turning and we just move forward.
    if turn <= 0:
        action = forward
        turn = 0
    action_n = [action for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    rewards += [reward_n[0]]

    if len(rewards) >= buffer_size:
        mean = sum(rewards)/len(rewards)
        if(mean == 0):
            turn = 20
            if random.random() <0.5:
                action = right
            else:
                action = left
        
        rewards = []
    env.render()

