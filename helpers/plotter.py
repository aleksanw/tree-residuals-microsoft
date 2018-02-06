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

def plot_agents(pickle_path):
    runs = pickler.load(pickle_path)
    runs = pd.DataFrame(runs)
    mean = pd.DataFrame({
        'Mean': runs.interpolate(axis=1).mean(),
        })
    median = pd.DataFrame({
        'Median': runs.interpolate(axis=1).median(),
        })
    fig = plt.figure()
    ax = fig.gca(title='Learning runs for Blackjack',
                 xlabel='Cumulative step count',
                 ylabel='Mean episode reward')

    for _, run in runs.iterrows():
        run.dropna().plot(ax=ax, alpha=0.3)

    mean.plot(ax=ax)
    median.plot(ax=ax)

    #ax.set_xlim(lims['x'])
    #ax.set_ylim(lims['y'])

    path = compose_path('figures', '.pdf', 'testing')
    plt.savefig(path, format='pdf', dpi=1000)
