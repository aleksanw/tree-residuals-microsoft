import matplotlib
matplotlib.use('Agg')
import glue
import gym  # OpenAI gym
import matplotlib.pyplot as plt
import helpers.plotter as plotter
import pandas as pd
import multiprocessing

from helpers.variables import write_variables_to_latex

env_name = 'NChain-v0'

def runglue(_):
    env = gym.make(env_name)
    params, prefs = glue.run(env)
    return params, dict(prefs)

def main():
    print('Hello from main_nchain main()')
    pool = multiprocessing.Pool()
    threads = 40
    results = pool.map(runglue, range(threads))
    perfs = [result[1] for result in results]

    variables = results[0][0]
    wanted_written = ['initial_learning_rate', 'initial_epsilon',
                      'rollout_batch_size', 'num_episodes',
    ]
    write_variables_to_latex(variables, wanted_written, env_name)

    fig = plt.figure()
    ax = fig.gca(xlabel='Interactions with environment',
                 ylabel=f'Episode reward in {env_name}')

    df = pd.DataFrame()
    for i, perf in enumerate(perfs):
        ser =  pd.Series(dict(perf))
        df[i] = ser
        ser.plot(ax=ax, color='gray')

    mean = df.mean(axis=1)
    print(mean)
    mean_df = pd.DataFrame({'Average': mean})
    mean_df.plot(ax=ax, color='orange')

    fig.tight_layout()
    figures = 'figures/'
    plt.savefig(figures + env_name + '.pdf', format='pdf', dpi=1000)


if __name__ == '__main__':
    main()
