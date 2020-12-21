import gym
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial

plt.style.use('ggplot')

env = gym.make('Blackjack-v0')


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    # stand if score>=20, otherwise hit
    return 0 if score >= 20 else 1


# one round
def generate_episode(env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        states.append(observation)
        action = sample_policy(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    return states, actions, rewards


player_sum = np.arange(12, 21 + 1)
dealer_show = np.arange(1, 10 + 1)
usable_ace = np.array([False, True])

X, Y = np.meshgrid(player_sum, dealer_show)
state_values = np.zeros((len(player_sum),
                         len(dealer_show),
                         len(usable_ace)))

fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})


def plot_blackjack(V, ax1, ax2):
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]

    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])


value_table = defaultdict(float)


def first_visit_mc_prediction(env, n_episodes):
    N = defaultdict(int)
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(env)
        returns = 0
        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            returns += R
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]



first_visit_mc_prediction(env, 1000000000)

for ax in axes:
    ax.set_zlim(-1, 1)
    ax.set_ylabel('player sum')
    ax.set_xlabel('dealer showing')
    ax.set_zlabel('state-value')

plot_blackjack(value_table, axes[0], axes[1])
plt.show()
