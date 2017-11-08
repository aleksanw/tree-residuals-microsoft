import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random
import itertools

#import corridor

# Used under development
import sys
from time import sleep
from pprint import pprint


def assert_shapetype(array, dtype, shape):
    array = np.asarray(array)
    assert (array.dtype == dtype
            and len(array.shape) == len(shape)
            and all(array.shape[i] == shape[i] for i in range(len(shape)) if shape[i] != -1)
            ), f"Shapetype mismatch: Expected {dtype}{shape}, Got {array.dtype}{array.shape}."


class Approximator_Table:
    def __init__(self, action_space):
        self.action_space = action_space
        self.table = {}

    def learn(self, learning_rate, X, Y_target):
        """Update approximator, correcting `learning_rate` of the error.

        Parameters
        ----------
        learning_rate : weight of 
        X : sequence of scalars or vectors -- features
        Y_target : sequence of scalars -- target values corresponding to features in X
        """
        # Coerce scalars to 1-dim vectors
        X = np.reshape(X, (-1,1))

        Y = self(X)
        Y_target = np.asarray(Y_target)
        Y_update = learning_rate*Y_target + (1-learning_rate)*Y

        self.table.update(zip(map(tuple, X), Y_update))

    def __call__(self, X):
        """Evaluate approximator at each x in X.

        Parameters
        ----------
        X : sequence of scalars or vectors -- features

        Returns
        -------
        numpy array of approximated values
        """
        # Coerce scalars to 1-dim vectors
        X = np.reshape(X, (-1,1))
        return np.fromiter((self.table.get(tuple(x), 0.0) for x in X),
                            np.float64, count=len(X))


class Approximator_ResidualBoosting:
    """Gradient boosted trees approximator.
    Features may be vectors or scalars.  Value is scalar.
    TODO: Require features to be vectors.  Makes for easier debugging.
    """
    def __init__(self, action_space):
        self.action_space = action_space
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
        assert_shapetype(X, 'int64', (-1,-1))
        assert_shapetype(Y_target, 'float64', (-1,1))

        # This function is not pure at all.  The returned tree is owned by the
        # Regressor and will be in-place replaced by future calls to fit.
        # Instantiate a new Regressor for every fiting.
        fit_tree = sklearn.tree.DecisionTreeRegressor(max_depth=1).fit

        X = np.asarray(X)
        Y_target = np.asarray(Y_target)

        # Allow scalar features.
        if len(X.shape) == 1:
            X.shape = (-1, 1)

        Y_residual = Y_target - self(X)
        h = fit_tree(X, Y_residual).predict

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
        assert_shapetype(X, 'int64', (-1,-1))
        # Approximators do not yet have learning rates applied.  Do that during
        # summation.
        sum_ = np.zeros((len(X),1))
        for lr, h in zip(self.learning_rates, self.approximators):
            Y = h(X).reshape(-1,1)
            sum_ += lr * Y

        assert_shapetype(sum_, 'float64', (-1,1))
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
        X = [(*state, action) for action in self.q.action_space]
        random.shuffle(X)
        assert_shapetype(X, 'int64', (-1,-1))
        action = X[self.q(X).argmax()][1]
        return action


class Policy_Deterministic:
    def __init__(self, q):
        self.q = q
        self.action_sequence = []
        self.step = 0
        self._file_to_sequence("sequence.txt")

    def __call__(self, state):
        """Get the next action in the sequence
        """
        action = self.action_sequence[self.step]
        self.step = (self.step+1) % len(self.action_sequence)
        return action

    def _file_to_sequence(self, filename):
        with open(filename) as f:
            for line in f:
                self.action_sequence.append(int(line.strip()))


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


def v(q, state):
    return max(q(((*state, x),))[0] for x in q.action_space)


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


def decay(initial, t):
    return initial/(1 + 0.04*t)

class Policy_ConstantActionLoop:
    def __init__(self, actions):
        self.actions = itertools.cycle(actions)

    def __call__(self, state):
        return self.actions.__next__()


def runner():
    env = gym.make('NChain-v0')
    action_space = list(range(env.action_space.n))

    initial_learning_rate = learning_rate = 0.15
    initial_epsilon = epsilon = 0.4
    episodes_per_update = 10

    q = Approximator_ResidualBoosting(action_space)
    for learn_iteration in itertools.count():
        learning_rate = decay(initial_learning_rate, learn_iteration)
        epsilon = decay(initial_epsilon, learn_iteration)
        policy = Policy_EpsilonGreedy(q, epsilon)
        episodes = [list(rollout(policy, env)) for _ in range(episodes_per_update)]
        targets = list(TD0_targets(episodes, q))
        X, Y_target = zip(*targets)
        assert_shapetype(X, 'int64', (-1,-1))
        assert_shapetype(Y_target, 'float64', (-1,1))
        q.learn(learning_rate, X, Y_target)

        if learn_iteration % 1 == 0:
            greedy_policy = Policy_Greedy(q)
            reward_sum = test_policy(greedy_policy, env)
            #n_trees = len(q.trees)
            #print(f"Episode {learn_iteration} RewardSum {reward_sum} Ntrees {n_trees}".format(learn_iteration, reward_sum, n_trees))
            print(f"Episode {learn_iteration} RewardSum {reward_sum} lr {learning_rate} epsilon {epsilon}".format(learn_iteration, reward_sum, learning_rate, epsilon))

if __name__ == '__main__':
    runner()
