import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import helpers.pickler as pickler



def compose_path(directory, suffix, *subnames):
    subnames = list(subnames)
    filename = '_'.join(subnames)
    filename += suffix
    path = os.path.join(directory, filename)
    
    return path


def plot(name, perfs):
    data = list(zip(*perfs))

    plt.plot(*data)
    plt.savefig(figures + name + '.pdf', format='pdf', dpi=1000)


def plot_with_mean(agent_run):
    fig = plt.figure()
    ax = fig.gca(xlabel='Interactions with environment',
                 ylabel=f'Episode reward in {agent_run.env_name}')

    df = pd.DataFrame()
    for i, perf in enumerate(agent_run.perfs):
        ser =  pd.Series(dict(perf))
        df[i] = ser
        ser.plot(ax=ax, color='gray')

    mean = df.mean(axis=1)
    mean_df = pd.DataFrame({'Average': mean})
    mean_df.plot(ax=ax, color='orange')
    path = compose_path('figures', '.pdf', agent_run.env_name,
            agent_run.agent_name,
            agent_run.start_time)
    plt.savefig(path, format='pdf', dpi=1000)

def plot_agents(paths):
    perfs = [pickler.load(path) for path in paths]
    print(perfs)
    fig = plt.figure()
    ax = fig.gca(xlabel='Interactions with environment',
                 ylabel=f'Episode reward in env')

    colors = ['red', 'blue']
    df = pd.DataFrame()
    for i in range(len(paths)):
        for j, perf in enumerate(perfs[i]):
            ser =  pd.Series(dict(perf))
            df[j] = ser
            ser.plot(ax=ax, color=colors[i])

        mean = df.mean(axis=1)
        mean_df = pd.DataFrame({f'Average {i}': mean})
        mean_df.plot(ax=ax, color='orange')
    path = compose_path('figures', '.pdf', 'testing')
    plt.savefig(path, format='pdf', dpi=1000)
