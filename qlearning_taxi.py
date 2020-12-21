import gym
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v2")
env.render()


# Q(s,a) = Q(s,a) + α( r + γ*maxQ(s',a') - Q(s,a) )
def update_q_table(prev_state, action, reward, next_state, alpha, gamma, q_table):
    qa = max(q_table[(next_state, a)] for a in range(env.action_space.n))
    q_table[(prev_state, action)] += alpha * (reward + gamma * qa - q_table[(prev_state, action)])


def epsilon_greedy_policy(state, epsilon, q_table):
    if random.uniform(0, 1) < epsilon:
        # do random action
        return env.action_space.sample()
    else:
        # select action with biggest q-value in this state
        return max(list(range(env.action_space.n)), key=lambda x: q_table[(state, x)])


n_episodes = 5000


def taxi_solve(alpha, gamma, epsilon):
    rewards = []
    # initialize Q table
    q = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            q[(s, a)] = 0.0

    for i in range(n_episodes):
        r = 0
        prev_state = env.reset()
        while True:
            action = epsilon_greedy_policy(prev_state, epsilon, q)
            next_state, reward, done, _ = env.step(action)
            update_q_table(prev_state, action, reward, next_state, alpha, gamma, q)

            prev_state = next_state
            r += reward
            if done:
                break
        if i % 20 == 0:
            rewards.append(r/20)
    return rewards
    # print("total reward: ", r)


a1 = taxi_solve(0.4, 0.999, 0.17)
a2 = taxi_solve(0.8, 0.999, 0.17)
a3 = taxi_solve(0.4, 0.999, 0.02)

plt.plot(range(len(a1)), a1, 'b-', linewidth=0.5)
plt.plot(range(len(a2)), a2, 'r-', linewidth=0.5)
plt.plot(range(len(a3)), a3, 'y-', linewidth=0.5)

plt.show()
env.close()
