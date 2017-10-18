import gym  # OpenAI gym
import sklearn.tree
import numpy as np
import random

# Used under development
import sys
from time import sleep
from pprint import pprint

class FunctionSum:
    """Function summation.

    Usage:
    funsum = FunctionSum()
    funsum += fun1
    funsum += fun2
    funsum_at_x = funsum(x)
    """
    def __init__(self):
        self.fs = []

    def __call__(self, *args, **kwargs):
        return sum(f(*args, **kwargs) for f in self.fs)

    def __iadd__(self, f):
        assert callable(f)
        self.fs.append(f)
        return self

class Table:
    # denumpyfied
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

asd = False
class ResidualBoostingApproximator:
    def __init__(self):
        self.funsum = FunctionSum()

    def q(self, arg):
        return self.funsum((arg,))

    def update(self, updates):
        """Update approximator.
        """
        regressor = sklearn.tree.DecisionTreeRegressor(max_depth=5)

        learning_rate = 0.2
        residuals = [(x, learning_rate*(y-self.q(x))) for (x, y) in updates]
        xs, ys = zip(*residuals)
        if not any(ys):
            return
        h = regressor.fit(xs, ys)
        self.funsum += h.predict

    def __call__(self, arg):
        return self.q(arg)

class EpsilonGreedy:
    def __init__(self, epsilon, actions, q):
        self.actions = actions
        self.q = q
        self.epsilon = epsilon
        self.greedy = Greedy(actions, q)

    def __call__(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.greedy(state)


class Greedy:
    def __init__(self, actions, q):
        self.actions = actions
        self.q = q

    def __call__(self, state):
        """Get most promising action.
        """
        return max(self.actions, key=lambda x: self.q((state,x)))


class TD0:
    """Learner type.  Implements greedy TD(0)
    """
    def __init__(self, epsilon, action_space, q):
        self.q = q
        self.actions = action_space
        self.epsilon = epsilon

    def td_target(self, reward, newstate):
        newstate_value = max(self.q((newstate, action)) for action in self.actions)
        discount = 0.90
        return reward + discount * newstate_value

    def learn(self, episodes):
        updates = []
        for episode in episodes:
            for state, action, reward, newstate in episode:
                updates.append(((state, action), self.td_target(reward, newstate)))
        self.q.update(updates)

    def make_policy(self):
        return EpsilonGreedy(self.epsilon, self.actions, self.q)

    def make_policy_greedy(self):
        return Greedy(self.actions, self.q)


# Types:
#  - Learner:
#      Takes batches of episodes of (state, action, reward, next_state) triplets.
#      Produces policies on request.
#  - Policy:

#      May have internal state.  Results in an action when applied to a state.

def rollout(policy, env):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        newstate, reward, done, _ = env.step(action)
        yield state, action, reward, newstate
        state = newstate

def test_rollout(policy, env):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
    return reward


def test_policy(policy, env):
    reward = test_rollout(policy, env)
    if reward != 0:
        print("Sucess!!!! <:O")
        sys.exit()


def policy_display(policy):
    arrows = '←↓→↑'
    out = ""
    for y in range(4):
        for x in range(4):
            if (y,x) in [(1,1), (1,3), (2,3), (3,0)]:
                out += ' '
                continue
            if (y,x) == (3,3):
                out += 'T'
                continue
            observation = y*4+x
            out += arrows[policy(observation)]
        out += "\n"
    print(out)


def runner():
    env = gym.make('FrozenLake-v0')
    env.unwrapped.P = {s:{a:([(1.0, y[1][1], y[1][2], y[1][3])] if len(y) == 3 else y) for (a,y) in x.items()} for (s,x) in env.unwrapped.P.items()}
    action_space = list(range(env.action_space.n))

    epsilon = 0.35
    episode_number = 0

    qlearner = ResidualBoostingApproximator()
    learner = TD0(epsilon, action_space, qlearner)

    print(f"epsilon: {epsilon}")
    while True:
        policy = learner.make_policy()

        episode_batch_size = 10
        episode_number += episode_batch_size
        episodes = (rollout(policy, env) for _ in range(episode_batch_size))
        learner.learn(episodes)

        greedy_policy = learner.make_policy_greedy()
        print(f"Episode {episode_number}")
        policy_display(greedy_policy)
        test_policy(greedy_policy, env)

if __name__ == '__main__':
    runner()
