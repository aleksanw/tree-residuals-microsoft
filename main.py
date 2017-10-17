import gym  # OpenAI gym
import sklearn.tree
import numpy as np
from time import sleep
import random

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

class ResidualBoostingApproximator:
    #FIXME types
    def __init__(self):
        self.q = FunctionSum()

    def update(self, x, updated_y):
        """Update approximator.
        """
        residual = updated_y - self.q(x)
        regressor = sklearn.tree.DecisionTreeRegressor(max_depth=10)
        h = regressor.fit(x, residual)
        self.q += h.predict

    def __call__(self, *args, **kwargs):
        return self.q(*args, **kwargs)

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
    def __init__(self, action_space, q):
        self.q = q
        self.actions = action_space
        self.epsilon = 0.8

    def td_target(self, reward, newstate):
        newstate_value = max(self.q((newstate, action)) for action in self.actions)
        discount = 0.1
        return reward + discount * newstate_value

    def learn(self, episodes):
        for episode in episodes:
            updates = []
            for state, action, reward, newstate in episode:
                updates.append(((state, action), self.td_target(reward, newstate)))
            self.q.update(updates)

    def make_policy(self):
        return EpsilonGreedy(self.epsilon, self.actions, self.q)


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
    reward_sum = 0
    for episode_number in range(1000):
        reward_sum += test_rollout(policy, env)
    print(reward_sum/episode_number)


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
    qlearner = Table()

    greedy_policy = Greedy(action_space, qlearner)
    policy_display(greedy_policy)

    learner = TD0(action_space, qlearner)

    for episode_number in range(100000):
        if(episode_number % 100 == 0):
            print(f"Episode {episode_number}")
            test_policy(greedy_policy, env)
            policy_display(greedy_policy)

        policy = learner.make_policy()
        episode = rollout(policy, env)
        learner.learn([episode])

if __name__ == '__main__':
    runner()
