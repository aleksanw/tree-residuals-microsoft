import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

import corridor

from utils import rollout,test_rollout, test_policy, decay, assert_shapetype, v
from policy import Policy_ConstantActionLoop, Policy_EpsilonGreedy, Policy_Greedy, Policy_Deterministic
from approximator import Approximator_Table, Approximator_ResidualBoosting

# Used under development
import sys
from time import sleep
from pprint import pprint


def TDinf_targets(episodes):
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
            yield ((*state, action), td_target)


def TD0_targets(episodes, q):
    discount = 0.95
    for episode in episodes:
        for state, action, reward, newstate in episode:
            td_target = reward + discount*v(q, newstate)
            yield ((*state, action), td_target)


def run():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    action_space = list(range(env.action_space.n))

    # Generate episodes by just repeatedly applying the same action
    loops = [
            [0],
            [1],
            [0,1],
            [0,0,1],
            [0,0,0,1],
            [0,0,0,0,1],
            ]

    episodes = []
    for loop in loops:
        episodes.append(list(rollout(Policy_ConstantActionLoop(loop), env)))


    # Attempt to overfit targets
    # Hopefully the approximator will still generalize to states it has not seen
    q = Approximator_ResidualBoosting(action_space)
    initial_learning_rate = 1.0
    learning_rate = initial_learning_rate
    for learning_iteration in range(100):
        targets = list(TD0_targets(episodes, q))
        targets = random.choices(targets, k=1000)
        X, Y_target = zip(*targets)
        Y_target = np.reshape(Y_target, (-1, 1))

        learning_rate = decay(initial_learning_rate, learning_iteration)
        q.learn(learning_rate, X, Y_target)

        greedy_policy = Policy_Greedy(q)
        reward_sum = test_policy(greedy_policy, env)
        print(f"Episode {learning_iteration} RewardSum {reward_sum} lr {learning_rate}")


if __name__ == '__main__':
    run()
