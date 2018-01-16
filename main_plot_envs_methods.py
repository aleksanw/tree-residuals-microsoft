import matplotlib
matplotlib.use('Agg')
import os

import corridor
import gym

import matplotlib.pyplot as plt
import pandas as pd

import glue as tree_agent

import pickle
import helpers.pickler as pickler
import helpers.plotter as plotter


def nonslip_nchain():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    return env

def nchain_microsoft():
    env = gym.make('NChain-v0')
    # Non-default reward for goal from Microsoft paper.
    env.unwrapped.large = 100
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
    basedir = 'perfs'
    env_name = 'NChain-v0'
    model = 'tree'
    result = pickler.load(os.path.join(basedir, env_name, model))
    print(result)

    plotter.plot_with_mean(env_name, result)

if __name__ == '__main__':
    main()
