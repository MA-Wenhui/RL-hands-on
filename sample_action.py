import gym
env = gym.make('BipedalWalker-v2')
for episode in range(100):
    observation = env.reset()
    for i in range(10000):
        env.render()
        action = env.action_space.sample()
        observation,reward,done,info = env.step(action)
        if done:
            print("{} timesteps taken for the episode".format(i+1))
            break
