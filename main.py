import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

import corridor

from utils import rollout,test_rollout, test_policy, decay, assert_shapetype, v
from policy import Policy_EpsilonGreedy, Policy_Greedy, Policy_Deterministic
from approximator import Approximator_Table, Approximator_ResidualBoosting

# Used under development
import sys
from time import sleep
from pprint import pprint


def TDinf_targets(episodes, q):
    """Generate td_targets (TDinf).
    episodes = (episode, ..)
    episode = (state, action, reward, newstate)
    Events in episode must come in order. That is event[0].newstate == event[1].state.
    """
    discount = 0.95
    for episode in episodes:
        episode = list(episode)
        # Work backwards and calculate td_targets
        td_target = 0.0  # assuming episode is a full rollout
        for state, action, reward, _ in episode[::-1]:
            td_target = reward + discount*td_target
            yield ((com*state, action), td_target)


def TD0_targets(episodes, q):
    discount = 0.95
    for episode in episodes:
        for state, action, reward, newstate in episode:
            td_target = reward + discount*v(q, newstate)
            yield ((*state, action), td_target)


def runner():
    env = gym.make('NChain-v0')
    action_space = list(range(env.action_space.n))

    initial_learning_rate = learning_rate = 0.15
    initial_epsilon = epsilon = 0.4
    episodes_per_update = 1

    q = Approximator_ResidualBoosting(action_space)
    for learn_iteration in itertools.count():
        learning_rate = decay(initial_learning_rate, learn_iteration)
        epsilon = decay(initial_epsilon, learn_iteration)
        policy = Policy_EpsilonGreedy(q, epsilon)
        episodes = [list(rollout(policy, env)) for _ in range(episodes_per_update)]
        targets = list(TD0_targets(episodes, q))

        # Checking if residuals are OK
        X, Y_target = zip(*targets)

        X_ = np.asarray(X)
        Y_target_ = np.asarray(Y_target)

        assert_shapetype(X, 'int64', (-1,-1))
        assert_shapetype(Y_target, 'float64', (-1,1))
        q.learn(learning_rate, X, Y_target)

        if learn_iteration % 1 == 0:
            Y_error_before = sum(Y_target_ - q(X_))
            Y_error_after = sum(Y_target - q(X))
            greedy_policy = Policy_Greedy(q)
            reward_sum = test_policy(greedy_policy, env)
            n_trees = len(q.approximators)
            print(f"Episode {learn_iteration} RewardSum {reward_sum} lr {learning_rate:.5f} epsilon {epsilon:.5f} n_trees {n_trees} errorBefore {Y_error_before} errorAfter {Y_error_after}")

if __name__ == '__main__':
    runner()
