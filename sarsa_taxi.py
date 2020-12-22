import gym
import random
import matplotlib.pyplot as plt
import qlearning_taxi

class SarsaTaxi(object):
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
            state = env.reset()
            # pick up an action from ε-greedy
            action = self.epsilon_greedy_policy(state, self.epsilon)
            while True:
                # move a step
                next_state, reward, done, _ = env.step(action)
                # choose next action, still by ε-greedy
                next_action = self.epsilon_greedy_policy(next_state, 0.001)
                # update Q-table
                # Q(s,a) = Q(s,a) + α( r + γ*Q(s',a') - Q(s,a) )
                self.q[(state, action)] += self.alpha * (reward + self.gamma * self.q[(next_state, next_action)] -
                                                         self.q[(state, action)])
                action = next_action
                state = next_state
                r += reward
                if done:
                    break
            if i % 20 == 0:
                rewards.append(r / 20)
        return rewards
        # print("total reward: ", r)


env = gym.make("Taxi-v2")

a1 = SarsaTaxi(env, 0.4, 0.999, 0.05).solve()
a2 = qlearning_taxi.QLearningTaxi(env, 0.4, 0.999, 0.05).solve()

plt.plot(range(len(a1)), a1, 'b-', linewidth=0.5)
plt.plot(range(len(a2)), a2, 'r-', linewidth=0.5)

plt.show()
