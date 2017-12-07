import os

import corridor
import gym

import matplotlib.pyplot as plt
import pandas as pd

import glue as tree_agent
import dqn as dqn_agent

import pickle


def nonslip_nchain():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    return env


envs = [
    #('Nonslip nchain', nonslip_nchain),
    #('Blackjack', lambda: gym.make('Blackjack-v0')),
    nonslip_nchain,
    #lambda: gym.make('CorridorBig-v10'),
    ]


def dropafter(predicate, iterable):
    for x in iterable:
        yield x
        if predicate(x):
            return

def main():
    for make_env in envs:
        if os.path.isfile('tree_run.pickle'):
            tree_run_result = pickle.load(open('tree_run.pickle', 'rb'))
        else:
            tree_run_result = tree_agent.run(make_env())
            tree_run_result = list(dropafter(lambda x: x[1] == 9960, tree_run_result))
            pickle.dump(tree_run_result, open('tree_run.pickle', 'wb'))

        if os.path.isfile('dqn_run.pickle'):
            dqn_run_result = pickle.load(open('dqn_run.pickle', 'rb'))
        else:
            dqn_run_result = dqn_agent.run(make_env())
            dqn_run_result = list(dropafter(lambda x: x[1] == 9960, dqn_run_result))
            pickle.dump(dqn_run_result, open('dqn_run.pickle', 'wb'))

        both_results = pd.DataFrame({
             'Tree': pd.Series(dict(tree_run_result)),
             'DQN': pd.Series(dict(dqn_run_result)),
             })

        fig = plt.figure()
        ax = fig.gca(xlabel='Interactions with environment',
                     ylabel='Episode reward')
        both_results.plot(ax=ax)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
