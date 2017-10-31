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
    def __init__(self, action_space):
        self.table = {}
        self.action_space = action_space

    def apply_learning_rate(self, learning_rate, targets):
        for x,y in targets:
            yield x, learning_rate*y + (1-learning_rate)*self(x)

    def learn(self, learning_rate, targets):
        """
        updates: ((arg, val), ...)
        arg must be hashable
        """
        self.table.update(self.apply_learning_rate(learning_rate, targets))

    def __call__(self, arg):
        return self.table.get(arg, 0.0)


class Approximator_ResidualBoosting:
    """Gradient boosted trees approximator.
    Features may be vectors or scalars.  Value is scalar.
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.fit_tree = sklearn.tree.DecisionTreeRegressor(max_depth=2).fit
        self.approximators = []
        self.learning_rates = []

    def learn(self, learning_rate, X, Y_target):
        """Update approximator, correcting `learning_rate` of the error.

        Parameters
        ----------
        learning_rate : weight of 
        X : sequence of scalars or vectors -- features
        Y_target : sequence of scalars -- target values corresponding to features in X
        """
        X = np.asarray(X)
        Y_target = np.asarray(Y_target)

        # Allow scalar features.
        if len(X.shape) == 1:
            X.shape = (-1, 1)

        Y_error = Y_target - self(X)
        h = self.fit_tree(X, Y_error).predict

        # As in Microsoft's paper, apply learning_rate after fitting.
        #     h = lambda X: learning_rate*h(X)
        # Avoid expensive lambdas by instead applying learning rate on
        # evaluation.  Save learning_rate for this purpose.
        self.learning_rates.append(learning_rate)

        # Sidenote: It should be more or less equivalent to apply the learning
        # rate to the residuals before fitting, saving on storage and computation.

        self.approximators.append(h)

    def __call__(self, X):
        """Evaluate approximator at each x in X.

        Parameters
        ----------
        X : sequence of scalars or vectors -- features

        Returns
        -------
        numpy array of approximated values
        """
        # Approximators do not yet have learning rates applied.  Do that during
        # summation.
        sum_ = np.zeros(len(X))
        for lr, h in zip(self.learning_rates, self.approximators):
            sum_ += lr * h(X)
        return sum_


class Policy_EpsilonGreedy:
    def __init__(self, q, epsilon):
        self.q = q
        self.epsilon = epsilon
        self.greedy = Policy_Greedy(q)

    def __call__(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.q.action_space)
        else:
            return self.greedy(state)


class Policy_Greedy:
    def __init__(self, q):
        self.q = q

    def __call__(self, state):
        """Get most promising action.
        """
        random.shuffle(self.q.action_space)
        return max(self.q.action_space, key=lambda x: self.q((*state,x)))


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


def v(q, state):
    return max(q((*state, x)) for x in q.action_space)


def TD0_targets(episodes, q):
    discount = 0.95
    for episode in episodes:
        for state, action, reward, newstate in episode:
            td_target = reward + discount*v(q, newstate)
            yield ((*state, action), td_target)


def rollout(policy, env):
    state = np.array(env.reset()).flatten()
    done = False
    while not done:
        action = policy(state)
        newstate, reward, done, _ = env.step(action)
        newstate = np.array(newstate).flatten()
        yield state, action, reward, newstate
        state = newstate


def test_rollout(policy, env):
    reward_sum = 0
    for (_, _, reward, _) in rollout(policy, env):
        reward_sum += reward
    return reward_sum


def test_policy(policy, env):
    return np.average([test_rollout(policy, env) for _ in range(10)])


def runner():
    env = gym.make('NChain-v0')
    action_space = list(range(env.action_space.n))

    epsilon = 0.3
    learning_rate = 0.6
    episodes_per_update = 1

    q = Approximator_ResidualBoosting(action_space)
    for learn_iteration in itertools.count():
        policy = Policy_EpsilonGreedy(q, epsilon)
        episodes = (rollout(policy, env) for _ in range(episodes_per_update))
        targets = TD0_targets(episodes, q)
        q.learn(learning_rate, targets)

        if learn_iteration % 10 == 0:
            greedy_policy = Policy_Greedy(q)
            reward_sum = test_policy(greedy_policy, env)
            print(f"Episode {learn_iteration}: {reward_sum}")


if __name__ == '__main__':
    runner()
