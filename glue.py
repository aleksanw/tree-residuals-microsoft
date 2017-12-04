import sklearn.tree
import numpy as np
import random
import itertools

from approximator import Approximator_Table, Approximator_ResidualBoosting
from policy import Policy_ConstantActionLoop, Policy_EpsilonGreedy, Policy_Greedy, Policy_Deterministic
from td import TD0_targets
from utils import rollout,test_rollout, test_policy, decay, assert_shapetype, v
from replay_buffer import Replay_buffer

# Used under development
import sys
from time import sleep
from pprint import pprint

import matplotlib.pyplot as plt

def avg(it):
    length = 0
    sum_ = 0
    for x in it:
        length += 1
        sum_ += x
    return sum_/length

class PolicyPlotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca()

    def update(self, env, policy):
        X,Y,U,V = self.policy_to_xyuv(env, policy)
        self.ax.clear()
        self.ax.invert_yaxis()
        self.ax.quiver(X,Y,U,V, angles='xy', pivot='mid')
        self.fig.tight_layout()
        self.fig.canvas.draw()

    @staticmethod
    def policy_to_xyuv(env, policy):
        right = (1, 0)
        left = (-1, 0)
        up = (0, -1)
        down = (0, 1)
        action_to_uv = {0: left, 1: down, 2: right, 3: up}

        X = []
        Y = []
        U = []
        V = []

        world = env.unwrapped.desc
        world_height, world_width = world.shape

        for y in range(0, world_height):
            for x in range(0, world_width):
                if world[y,x] == b'F':
                    observation = [y*world_width + x]
                    action = policy(observation)
                    u, v = action_to_uv.get(action, (0,0))
                    X.append(x)
                    Y.append(y)
                    U.append(u)
                    V.append(v)

        return X,Y,U,V

def run(env):
    action_space = list(range(env.action_space.n))
    replay_buffer = Replay_buffer()

    q = Approximator_ResidualBoosting(action_space)
    initial_learning_rate = 0.50
    learning_rate = initial_learning_rate
    initial_epsilon = 0.80
    epsilon = initial_epsilon
    batch_size = 50

    interaction_count = 0
    interactions = []
    rewards = []

    policy_plot = PolicyPlotter()

    for learning_iteration in range(300):
        policy = Policy_EpsilonGreedy(q, epsilon=epsilon)
        episodes = [list(rollout(policy, env)) for _ in range(batch_size)]
        interaction_count += sum(map(len, episodes))
        replay_buffer += episodes
        sampled_episodes = replay_buffer.sample(50)
        targets = TD0_targets(sampled_episodes, q)
        X, Y_target = zip(*targets)
        Y_target = np.reshape(Y_target, (-1, 1))

        learning_rate = decay(initial_learning_rate, learning_iteration*batch_size)
        epsilon = decay(initial_epsilon, learning_iteration*batch_size)
        q.learn(learning_rate, X, Y_target)

        if learning_iteration % 1 == 0:
            greedy_policy = Policy_Greedy(q)
            policy_plot.update(env, greedy_policy)
            reward_sum = avg(test_policy(greedy_policy, env) for _ in range(1))
            interactions.append(interaction_count)
            rewards.append(reward_sum)
            print(f"Episode {learning_iteration*batch_size} Reward {reward_sum} lr {learning_rate} epsilon {epsilon}")

    return interactions, rewards
