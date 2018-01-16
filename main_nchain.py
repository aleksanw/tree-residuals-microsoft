import matplotlib
matplotlib.use('Agg')
import glue
import gym  # OpenAI gym
import os
import multiprocessing

import helpers.pickler as pickler

from helpers.variables import write_variables_to_latex
import helpers.plotter as plotter

env_name = 'NChain-v0'

def runglue(_):
    env = gym.make(env_name)
    params, prefs = glue.run(env)
    return params, dict(prefs)

def main():
    pool = multiprocessing.Pool()
    threads = 3
    results = pool.map(runglue, range(threads))
    perfs = [result[1] for result in results]

    variables = results[0][0]
    wanted_written = ['initial_learning_rate', 'initial_epsilon',
                      'rollout_batch_size', 'num_episodes',
    ]
    write_variables_to_latex(variables, wanted_written, env_name)

    pickler.dump(os.path.join('perfs', env_name, 'tree.pickle'), perfs)
    plotter.plot_with_mean(env_name, perfs)


if __name__ == '__main__':
    main()
