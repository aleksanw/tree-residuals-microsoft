import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

from environments import corridor

from approximator import Approximator_Table, Approximator_ResidualBoosting
from policy import Policy_ConstantActionLoop, Policy_EpsilonGreedy, Policy_Greedy, Policy_Deterministic
from td import TD0_targets
from utils import rollout,test_rollout, test_policy, decay, assert_shapetype, v


# Used under development
import sys
from time import sleep
from pprint import pprint


def avg(it):
    length = 0
    sum_ = 0
    for x in it:
        length += 1
        sum_ += x
    return sum_/length


def run():
    env = gym.make('CorridorSmall-v10')
    action_space = list(range(env.action_space.n))

    q = Approximator_ResidualBoosting(action_space)
    initial_learning_rate = 0.15
    learning_rate = initial_learning_rate
    initial_epsilon = 0.15
    epsilon = initial_epsilon
    batch_size = 10

    for learning_iteration in range(1000):
        policy = Policy_EpsilonGreedy(q, epsilon)
        episodes = [rollout(policy, env) for _ in range(batch_size)]
        targets = TD0_targets(episodes, q)
        X, Y_target = zip(*targets)
        Y_target = np.reshape(Y_target, (-1, 1))

        learning_rate = decay(initial_learning_rate, learning_iteration)
        epsilon = decay(initial_epsilon, learning_iteration)
        q.learn(learning_rate, X, Y_target)

        if learning_iteration % 1 == 0:
            greedy_policy = Policy_Greedy(q)
            reward_sum = avg(test_policy(greedy_policy, env) for _ in range(10))
            print(f"Episode {learning_iteration*batch_size} Reward {reward_sum} lr {learning_rate} epsilon {epsilon}")


if __name__ == '__main__':
    run()
