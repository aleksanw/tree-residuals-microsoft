import random
import numpy as np

from utils import assert_shapetype

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
        action = X[self.q(X).argmax()][-1]
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
