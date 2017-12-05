import corridor
import gym

import matplotlib.pyplot as plt
import pandas as pd

import glue as tree_agent
import dqn as dqn_agent

import pickle


def nonslip_nchain():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    return env


envs = [
    #('Nonslip nchain', nonslip_nchain),
    #('Blackjack', lambda: gym.make('Blackjack-v0')),
    #nonslip_nchain,
    lambda: gym.make('CorridorBig-v10'),
    ]


def main():
    for make_env in envs:
        #tree_run_result = tree_agent.run(make_env())
        #pickle.dump(tree_run_result, open('tree_run.data', 'wb'))
        #tree_run_result = pickle.load(open('tree_run.data', 'rb'))
        dqn_run_result = dqn_agent.run(make_env())
        #pickle.dump(dqn_run_result, open('dqn_run.data', 'wb'))
        #dqn_run_result = pickle.load(open('dqn_run.data'
        both_results = pd.DataFrame({
             'Tree': pd.Series(tree_run_result[1], index=tree_run_result[0]),
             'DQN': pd.Series(dqn_run_result[1], index=dqn_run_result[0]),
             })
        both_results.plot()
        plt.show()


if __name__ == '__main__':
    main()
