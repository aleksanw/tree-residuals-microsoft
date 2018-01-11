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

def images_from_episodes(episodes):
    return np.array([sars[0] for e in episodes for sars in e])

def run(env):
    initial_learning_rate = 0.20
    initial_epsilon = 0.50
    rollout_batch_size = 1
    replay_batch_size = 1
    learning_iterations = 40
    num_episodes = learning_iterations*rollout_batch_size

    params = locals()

    def gen():
        action_space = list(range(env.action_space.n))
        replay_buffer = Replay_buffer()

        q = Approximator_ResidualBoosting(action_space)
        learning_rate = initial_learning_rate
        epsilon = initial_epsilon
        interaction_count = 0

        plot_policy = hasattr(env.unwrapped, 'desc')
        if plot_policy:
            policy_plot = PolicyPlotter()

        for learning_iteration in range(learning_iterations):
            if learning_iteration % 1 == 0:
                greedy_policy = Policy_Greedy(q)
                if plot_policy:
                    policy_plot.update(env, greedy_policy)
                reward_sum = avg(test_policy(greedy_policy, env) for _
                        in range(1))
                print(f"Episode {learning_iteration*rollout_batch_size} Reward {reward_sum} lr {learning_rate} epsilon {epsilon}")
                yield interaction_count, reward_sum

            policy = Policy_EpsilonGreedy(q, epsilon=epsilon)
            episodes = [list(rollout(policy, env)) for _ in range(rollout_batch_size)]
            interaction_count += sum(map(len, episodes))
            replay_buffer += episodes
            sampled_episodes = replay_buffer.sample(replay_batch_size)
            sampled_episodes = episodes

            targets = TD0_targets(sampled_episodes, q)
            X, Y_target = zip(*targets)
            Y_target = np.reshape(Y_target, (-1, 1))

            learning_rate = decay(initial_learning_rate, learning_iteration*rollout_batch_size)
            epsilon = decay(initial_epsilon, learning_iteration*rollout_batch_size)
            q.learn(learning_rate, X, Y_target)

    return (params, gen())
