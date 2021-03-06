import gym
import random
import matplotlib.pyplot as plt

class QLearningTaxi(object):
    def __init__(self, en, a, g, e):
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.n_episodes = 5000
        self.env = en
        self.env.reset()
        self.q = {}
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                self.q[(s, a)] = 0.0

    # Q(s,a) = Q(s,a) + α( r + γ*maxQ(s',a') - Q(s,a) )
    def update_q_table(self, prev_state, action, reward, next_state, alpha, gamma):
        qa = max(self.q[(next_state, a)] for a in range(self.env.action_space.n))
        self.q[(prev_state, action)] += alpha * (reward + gamma * qa - self.q[(prev_state, action)])

    def epsilon_greedy_policy(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            # do random action
            return self.env.action_space.sample()
        else:
            # select action with biggest q-value in this state
            return max(list(range(self.env.action_space.n)), key=lambda x: self.q[(state, x)])

    def solve(self):
        rewards = []
        # initialize Q table

        for i in range(self.n_episodes):
            r = 0
            prev_state = self.env.reset()
            while True:
                action = self.epsilon_greedy_policy(prev_state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(prev_state, action, reward, next_state, self.alpha, self.gamma)

                prev_state = next_state
                r += reward
                if done:
                    break
            if i % 20 == 0:
                rewards.append(r / 20)
        return rewards
        # print("total reward: ", r)

#
# a1 = QLearningTaxi(env, 0.4, 0.999, 0.17).solve()
# a2 = QLearningTaxi(env, 0.8, 0.999, 0.17).solve()
# a3 = QLearningTaxi(env, 0.4, 0.999, 0.02).solve()
#
# plt.plot(range(len(a1)), a1, 'b-', linewidth=0.5)
# plt.plot(range(len(a2)), a2, 'r-', linewidth=0.5)
# plt.plot(range(len(a3)), a3, 'y-', linewidth=0.5)
#
# plt.show()
