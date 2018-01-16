import matplotlib.pyplot as plt
import pandas as pd

figures = 'figures/'

def plot(name, perfs):
    data = list(zip(*perfs))

    plt.plot(*data)
    plt.savefig(figures + name + '.pdf', format='pdf', dpi=1000)

def plot_with_mean(env_name, perfs):
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
    plt.savefig(figures + env_name + '.pdf', format='pdf', dpi=1000)

