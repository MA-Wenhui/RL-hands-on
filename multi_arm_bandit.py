import gym
import gym_bandits
import numpy as np
import math
import random
import matplotlib.pyplot as plt

env = gym.make("BanditTenArmedGaussian-v0")

print(env.action_space)

# number of rounds (iterations)
num_rounds = 20000


def epsilon_greedy(epsilon, Q):
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action


# Boltzmann function
# P(x) = e^(Qx/τ) / ∑e^(Qi/τ)
def softmax(tau, Q):
    total = sum([math.exp(val / tau) for val in Q])
    probs = [math.exp(val / tau) / total for val in Q]
    threshold = random.random()  # explore actions randomly
    cumulative_prob = 0.0
    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if cumulative_prob > threshold:
            return i
    return np.argmax(probs)


# upper confidence bound
def get_best_arm_ucb():
    # Count of number of times an arm was pulled
    count = np.zeros(10)
    # Sum of rewards of each arm
    sum_rewards = np.zeros(10)
    # Q value which is the average reward
    Q = np.zeros(10)
    for i in range(num_rounds):
        ucb = np.zeros(10)
        arm_select = int()
        if i < 10:
            arm_select = i
        else:
            for arm in range(10):
                upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])
                ucb[arm] = Q[arm] + upper_bound
            arm_select = (np.argmax(ucb))
        observation, reward, done, info = env.step(arm_select)
        count[arm_select] += 1
        sum_rewards[arm_select] += reward
        Q[arm_select] = sum_rewards[arm_select] / count[arm_select]
    return Q


def get_best_arm_epsilon():
    # Count of number of times an arm was pulled
    count = np.zeros(10)
    # Sum of rewards of each arm
    sum_rewards = np.zeros(10)
    # Q value which is the average reward
    Q = np.zeros(10)

    for i in range(num_rounds):
        # select an arm using epsilon greedy
        arm = epsilon_greedy(0.4, Q)
        # Get the reward
        observation, reward, done, info = env.step(arm)
        # update the count of this arm
        count[arm] += 1
        # sum the rewards obtained from the arm
        sum_rewards[arm] += reward
        # print(reward)
        # calculate Q value which is the average rewards of the arm
        Q[arm] = sum_rewards[arm] / count[arm]

    print("The optimal arm is {}".format(np.argmax(Q)))
    return Q


def get_best_arm_softmax():
    # Count of number of times an arm was pulled
    count = np.zeros(10)
    # Sum of rewards of each arm
    sum_rewards = np.zeros(10)
    # Q value which is the average reward
    Q = np.zeros(10)

    for i in range(num_rounds):
        arm = softmax(0.5, Q)
        observation, reward, done, info = env.step(arm)
        count[arm] += 1
        sum_rewards[arm] += reward
        Q[arm] = sum_rewards[arm] / count[arm]

    return Q


def thompson_sampling(alpha, beta):
    samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(10)]
    return np.argmax(samples)


def get_best_arm_thompson_sampling():
    # Count of number of times an arm was pulled
    count = np.zeros(10)
    # Sum of rewards of each arm
    sum_rewards = np.zeros(10)
    # Q value which is the average reward
    Q = np.zeros(10)
    # initialize alpha and beta values
    alpha = np.ones(10)
    beta = np.ones(10)

    for i in range(num_rounds):
        # select an arm using thompson sampling
        arm = thompson_sampling(alpha, beta)
        observation, reward, done, info = env.step(arm)
        count[arm] += 1
        sum_rewards[arm] += reward
        Q[arm] = sum_rewards[arm] / count[arm]
        if reward > 0:
            alpha[arm] += 1
        else:
            beta[arm] += 1

    print('The optimal arm is {}'.format(np.argmax(Q)))
    return Q


a1 = get_best_arm_epsilon()
a2 = get_best_arm_softmax()
a3 = get_best_arm_ucb()
a4 = get_best_arm_thompson_sampling()

plt.subplot(221, title="epsilon")
plt.plot(range(len(a1)), a1, 'r-')
plt.subplot(222, title="softmax")
plt.plot(range(len(a2)), a2, 'b-*')
plt.subplot(223, title="ucb")
plt.plot(range(len(a3)), a3, 'y-.')
plt.subplot(224, title="thompson sampling")
plt.plot(range(len(a4)), a4, 'y-*')
plt.show()
