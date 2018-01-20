import matplotlib.pyplot as plt
import pandas as pd
import time
import os


directory = 'figures'

def compose_path(*subnames):
    subnames = list(subnames)
    suffix = '.pdf'
    subnames.append(time.strftime("%Y_%m_%d_%H_%M"))
    filename = '_'.join(subnames)
    filename += suffix
    path = os.path.join(directory, filename)

    return path


def plot(name, perfs):
    data = list(zip(*perfs))

    plt.plot(*data)
    plt.savefig(figures + name + '.pdf', format='pdf', dpi=1000)


def plot_with_mean(env_name, agent_name, perfs):
    fig = plt.figure()
    ax = fig.gca(xlabel='Interactions with environment',
                 ylabel=f'Episode reward in {env_name}')

    df = pd.DataFrame()
    for i, perf in enumerate(perfs):
        ser =  pd.Series(dict(perf))
        df[i] = ser
        ser.plot(ax=ax, color='gray')

    mean = df.mean(axis=1)
    mean_df = pd.DataFrame({'Average': mean})
    mean_df.plot(ax=ax, color='orange')
    path = compose_path(env_name, agent_name)
    plt.savefig(path, format='pdf', dpi=1000)

