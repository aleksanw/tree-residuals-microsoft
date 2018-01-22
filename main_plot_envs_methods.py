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


default_dqn_config = Config(
        initial_epsilon = 0.50,
        initial_learning_rate = 0.20,
        learning_iterations = 4000,
        replay_batch_size = 1,
        rollout_batch_size = 1,
        test_rollouts = 4,
        update_freq = 4,
        y = 0.99,
        end_epsilon = 0,
        tau = 0.001,
        hiddens = [50, 50],
        annealing_steps = 4000,
        pre_train_steps = 0,
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
    config = aggregate_config(env_name, dqn_config)
    perfs = dqn_agent.run(env, config)
    return (config, dict(perfs))


def run_table(env_name):
    env = gym.make(env_name)
    params, prefs = table_agent.run(env)
    return params, dict(prefs)



def dropafter(predicate, iterable):
    for x in iterable:
        yield x
        if predicate(x):
            return


def get_config(results):
    # Config are yielded first by agent runs
    return results[0][0]


def get_perfs(results):
    # Performances are yielded last(index 1) by agent runs
    return [result[1] for result in results]


def get_current_time():
    # Not really start-time, but called only once at start
    return time.strftime("%Y_%m_%d_%H_%M")


envs= [
    'Blackjack-v0',
    #'NChain-v0' ,
    #'Pong-v0',
    ]

agents = [
    ('tree', run_tree),
    #('dqn', run_dqn)
    ]

thread_config = {
        'tree' : 1,
        'dqn' : 5
        }

def main():
    start_time = get_current_time()

    for env_name in envs:
        for agent_name, agent in agents:
            pool = multiprocessing.Pool()
            results = pool.map(agent, [env_name]*thread_config[agent_name])
            config = get_config(results)
            perfs = get_perfs(results)


            wanted_written = ['initial_learning_rate', 'initial_epsilon',
                              'rollout_batch_size', 'num_episodes',
            ]

            agent_run = Config(
                    start_time = start_time,
                    env_name = env_name,
                    agent_name = agent_name,
                    perfs = perfs, 
                    wanted_written = wanted_written,
                    config = config,
                    )

            write_variables_to_latex(agent_run)
            plotter.plot_with_mean(agent_run)
            pickler.dump(agent_run)


if __name__ == '__main__':
    main()
