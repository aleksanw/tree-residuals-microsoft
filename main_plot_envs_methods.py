import matplotlib
matplotlib.use('Agg')
import corridor
import gym
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd
import pickle

import dqn as dqn_agent
import glue as tree_agent
import helpers.pickler as pickler
import helpers.plotter as plotter
from helpers.variables import write_variables_to_latex



def nchain_microsoft():
    env = gym.make('NChain-v0')
    # Non-default reward for goal from Microsoft paper.
    env.unwrapped.large = 100
    return env


def run_tree(env_name):
    env = gym.make(env_name)
    params, prefs = tree_agent.run(env)
    return params, dict(prefs)


def run_dqn(env_name):
    env = gym.make(env_name)
    if env_name == 'Blackjack-v0':
        env.observation_space.shape = [3]
    params, prefs = dqn_agent.run(env)
    return params, dict(prefs)



def dropafter(predicate, iterable):
    for x in iterable:
        yield x
        if predicate(x):
            return


envs = [
    'Blackjack-v0',
    'NChain-v0' ,
    ]

agents = [
    ('tree', run_tree),
    ('dqn', run_dqn)
    ]


def main():
    basedir = 'perfs'

    for env_name in envs:
        for agent_name, agent in agents:
            pool = multiprocessing.Pool()
            threads = 30
            if agent_name == 'dqn':
                threads = 1
            results = pool.map(agent, [env_name]*threads)
            perfs = [result[1] for result in results]

            variables = results[0][0]
            wanted_written = ['initial_learning_rate', 'initial_epsilon',
                              'rollout_batch_size', 'num_episodes',
            ]
            write_variables_to_latex(variables, wanted_written,
                    env_name, agent_name)

            pickler.dump(os.path.join('perfs', env_name, 'tree.pickle'), perfs)
            plotter.plot_with_mean(env_name, perfs)



if __name__ == '__main__':
    main()
