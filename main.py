import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

import corridor

# Used under development
import sys
from time import sleep
from pprint import pprint


class Approximator_Table:
    def __init__(self):
        self.table = {}

    def update(self, updates):
        """
        updates: ((arg, val), ...)
        arg must be hashable
        """
        self.table.update(updates)

    def __call__(self, arg):
        return self.table.get(arg, 0.0)


class Approximator_ResidualBoosting:

    def __init__(self):
        self.trees = []
        self.tree_regressor = sklearn.tree.DecisionTreeRegressor(max_depth=2)

    def update(self, updates):
        """Update approximator.
        """
        residuals = [(x, (y - self(x))) for (x, y) in updates]
        xs, ys = zip(*residuals)

        # No value in keeping all-zero trees
        if not any(ys):
            return

        h = self.tree_regressor.fit(xs, ys)
        self.trees.append(h.predict(x))

    # TODO? memorize
    def __call__(self, arg):
        return sum(tree(arg) for tree in self.trees)


class Policy_EpsilonGreedy:
    def __init__(self, q, actions, epsilon):
        self.q = q
        self.actions = actions
        self.epsilon = epsilon

        self.greedy = Greedy(actions, q)

    def __call__(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.greedy(state)


class Policy_Greedy:
    def __init__(self, q, actions):
        self.q = q
        self.actions = actions

    def __call__(self, state):
        """Get most promising action.
        """
        random.shuffle(self.actions)
        return max(self.actions, key=lambda x: self.q((*state,x)))


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
            yield ((*state, action), td_target)


def v(q, action_space, state):
    return max(q((*state, x)) for x in action_space)


def TD0_targets(epsiodes, q, action_space):
    for episode in episodes:
        for state, action, reward, newstate in episode:
            td_target = reward + discount*v(q, action_space, newstate)
            yield ((*state, action), td_target)


def rollout(policy, env):
    state = env.reset()
    done = False
    while not done:
        action = policy((state,))
        newstate, reward, done, _ = env.step(action)
        yield state, action, reward, newstate
        state = newstate


def make_policy(self):
    return EpsilonGreedy(self.epsilon, self.actions, self.q)


def make_policy_greedy(self):
    return Greedy(self.actions, self.q)



def test_rollout(policy, env):
    state = env.reset()
    done = False
    reward_sum = 0
    step_n = 0
    while not done:
        action = policy((state, step_n))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        step_n += 1
    return reward_sum


def test_policy(policy, env):
    return np.average([test_rollout(policy, env) for _ in range(10)])

def runner():
    env = gym.make('CorridorSmall-v5')
    #env.unwrapped.P = {s:{a:([(1.0, y[1][1], y[1][2], y[1][3])] if len(y) == 3 else y) for (a,y) in x.items()} for (s,x) in env.unwrapped.P.items()}
    action_space = list(range(env.action_space.n))

    epsilon = 0.5
    episode_number = 0
    learn_number = 0
    lambda_ = 0.9

    qlearner = Table()
    learner = TDlambda(epsilon, action_space, qlearner, lambda_=lambda_)

    print(f"epsilon: {epsilon}")
    for i in itertools.count():
        epsilon *= 0.9999

        episode_batch_size = 1
        episode_number += episode_batch_size
        policy = learner.make_policy()
        episodes = (rollout(policy, env) for _ in range(episode_batch_size))
        learner.learn(episodes)

        if i % 1000 == 0:
            greedy_policy = learner.make_policy_greedy()
            print(f"Episode {episode_number}")
            reward = test_policy(greedy_policy, env)
            print(reward)
            if reward > 100:
                print("Sucess!!!! o<:)~")
                return

if __name__ == '__main__':
    runner()
