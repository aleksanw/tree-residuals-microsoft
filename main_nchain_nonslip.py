import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

import corridor

from approximator import Approximator_Table, Approximator_ResidualBoosting
from policy import Policy_ConstantActionLoop, Policy_EpsilonGreedy, Policy_Greedy, Policy_Deterministic
from td import TD0_targets
from utils import rollout,test_rollout, test_policy, decay, assert_shapetype, v

# Used under development
import sys
from time import sleep
from pprint import pprint


def run():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    action_space = list(range(env.action_space.n))

    q = Approximator_ResidualBoosting(action_space)
    initial_learning_rate = 1.0
    learning_rate = initial_learning_rate
    for learning_iteration in range(100):
        policy = Policy_EpsilonGreedy(q, epsilon=0.5)
        episodes = [rollout(policy, env) for _ in range(10)]
        targets = TD0_targets(episodes, q)
        X, Y_target = zip(*targets)
        Y_target = np.reshape(Y_target, (-1, 1))

        learning_rate = decay(initial_learning_rate, learning_iteration)
        q.learn(learning_rate, X, Y_target)

        greedy_policy = Policy_Greedy(q)
        reward_sum = test_policy(greedy_policy, env)
        print(f"Episode {learning_iteration} RewardSum {reward_sum} lr {learning_rate}")


if __name__ == '__main__':
    run()