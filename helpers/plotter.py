import matplotlib.pyplot as plt
import pandas as pd

figures = 'figures/'

def plot(name, run_yields):
    data = list(zip(*run_yields))

    plt.plot(*data)
    plt.savefig(figures + name + '.pdf', format='pdf', dpi=1000)

def plot_rolling_mean(env_name, run_yields):
    data = pd.DataFrame({
        'Tree': pd.Series(dict(run_yields)),
    })
    data['Moving Average'] = data.rolling(window=20, min_periods=1,
            center=True).mean()

    fig = plt.figure()
    ax = fig.gca(xlabel='Interactions with environment',
                 ylabel=f'Episode reward in {env_name}')
    data.plot(ax=ax)
    fig.tight_layout()
    plt.savefig(figures + env_name + '.pdf', format='pdf', dpi=1000)
