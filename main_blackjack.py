import glue
import gym  # OpenAI gym
import matplotlib.pyplot as plt
import helpers.plotter as plotter

from helpers.variables import write_variables_to_latex

def main():
    env_name = 'Blackjack-v0'
    env = gym.make(env_name)
    result = list(glue.run(env))

    data = result[:-1]
    variables = result[-1]
    plotter.plot_rolling_mean(env_name, data)

    wanted_written = ['initial_learning_rate', 'initial_epsilon',
            'rollout_batch_size',
            ]
    write_variables_to_latex(variables, wanted_written, env_name)






if __name__ == '__main__':
    main()
