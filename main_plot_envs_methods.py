'''
import multiprocessing, logging
mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)
'''


import matplotlib
matplotlib.use('Agg')
import corridor
import gym
import time
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

class Config(dict):
    def __getattr__(self, name):
        if name[0] == '_':
            raise AttributeError(name)
        return self[name]

    def __repr__(self):
        return f"Config({', '.join(k + '=' + repr(v) for k,v in self.items())})"

    def __add__(self, other):
        return Config(**{**self, **other})



default_tree_config = Config(
        initial_epsilon = 0.50,
        initial_learning_rate = 0.20,
        learning_iterations = 20,
        replay_batch_size = 1,
        rollout_batch_size = 1,
        test_rollouts = 4,
        discount = 0.95,
        )

tree_config = {
        'Blackjack-v0' : default_tree_config + Config(
            #learning_iterations = 350,
            ),
        'NChain-v0' : default_tree_config + Config(
            learning_iterations = 40,
            ),
        'Pong-v0' : default_tree_config + Config(
            ),
        }

default_ae_tree_config = Config(
        initial_epsilon = 0.50,
        initial_learning_rate = 0.20,
        learning_iterations = 20,
        replay_batch_size = 1,
        rollout_batch_size = 1,
        test_rollouts = 1,
        discount = 0.95,
        )

ae_tree_config = {
        'Blackjack-v0' : default_ae_tree_config + Config(
            #learning_iterations = 350,
            ),
        'NChain-v0' : default_ae_tree_config + Config(
            learning_iterations = 40,
            ),
        'Pong-v0' : default_ae_tree_config + Config(
            learning_iterations = 40000,
            ),
        }


default_dqn_config = Config(
        batch_size = 3, #How many experiences to use for each training step.
        update_freq = 1, #How often to perform a training step.
        y = .99, #Discount factor on the target Q-values
        startE = 1, #Starting chance of random action
        endE = 0, #0.1 #Final chance of random action
        max_epLength = 100, #The max allowed length of our episode.
        load_model = False, #Whether to load a saved model.
        tau = 0.001, #Rate to update target network toward primary network
        num_episodes = 1000, #How many episodes of game environment to train network with.
        annealing_steps = 5000, #How many steps of training to reduce startE to endE.
        pre_train_steps = 100, #How many steps of random actions before training begins.
        hiddens = [50, 50],
        )

dqn_config = {
        'Blackjack-v0' : default_dqn_config + Config(
            #learning_iterations = 400,
            ),
        'NChain-v0' : default_dqn_config + Config(
            ),
        'Pong-v0' : default_dqn_config + Config(
            ),
        }

def aggregate_config(env_name, config):
    config = config[env_name]
    aggregated_config = config + Config(
            num_episodes = config.learning_iterations*config.rollout_batch_size
            )
    return aggregated_config


def nchain_microsoft():
    env = gym.make('NChain-v0')
    # Non-default reward for goal from Microsoft paper.
    env.unwrapped.large = 100
    return env


def run_tree(env_name):
    env = gym.make(env_name)
    config = aggregate_config(env_name, tree_config)
    perfs = tree_agent.run(env, config)
    return (config, dict(perfs))



def run_dqn(env_name):
    env = gym.make(env_name)
    if env_name == 'Blackjack-v0':
        env.observation_space.shape = [3]
    #config = aggregate_config(env_name, dqn_config)
    config = dqn_config[env_name]
    perfs = dqn_agent.run(env, config)
    '''
    d_perfs = {}
    for i, v in perfs:
        d_perfs[i] = v
    '''
    return (config, dict(perfs))


def run_ae(env_name):
    import ae_agent
    env = gym.make(env_name)
    config = aggregate_config(env_name, ae_tree_config)

    perfs = ae_agent.run(env, config)
    return (config, dict(perfs))


def run_table(env_name):
    env = gym.make(env_name)
    params, prefs = table_agent.run(env)
    d_perfs = {}
    for i, v in perfs:
        d_perfs[i] = v
    return params, d_prefs


def dropafter(predicate, iterable):
    for x in iterable:
        yield x
        if predicate(x):
            return


def get_config(results):
    # Config are yielded first by agent runs
    # The same config are used for all agents of same type
    return results[0][0]


def get_perfs(results):
    # Performances are yielded as index 1 by agent runs
    return [result[1] for result in results]


def get_current_time():
    return time.strftime("%Y_%m_%d_%H_%M")


envs= [
    'Blackjack-v0',
    'NChain-v0' ,
    #'Pong-v0',
    ]

agents = [
    #('tree', run_tree),
    ('dqn', run_dqn),
    #('ae', run_ae),
    ]

thread_config = {
        'tree' : 4,
        'dqn' : 1,
        'ae' : 1,
        }

def main():
    start_time = get_current_time()

    for env_name in envs:
        for agent_name, agent in agents:
            pool = multiprocessing.Pool()
            results = pool.map(agent, [env_name]*thread_config[agent_name])
            config = get_config(results)
            perfs = get_perfs(results)

            #wanted_written = ['initial_learning_rate', 'initial_epsilon',
            #                  'rollout_batch_size', 'num_episodes',
            #]
            agent_run = Config(
                    start_time = start_time,
                    env_name = env_name,
                    agent_name = agent_name,
                    perfs = perfs, 
                    #wanted_written = wanted_written,
                    config = config,
                    )

            #write_variables_to_latex(agent_run)
            plotter.plot_with_mean(agent_run)
            pickler.dump(agent_run)


if __name__ == '__main__':
    main()
