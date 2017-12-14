import corridor
import gym

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glue as tree_agent
import dqn as dqn_agent


class SimpleWrapper(gym.Wrapper):
    def _reset(self):
        observation = self.env.reset()
        return self._apply(observation)

    def _step(self, action):
        observation, *rest = self.env.step(action)
        return (self._apply(observation), *rest)


class PongSimplify(SimpleWrapper):
    def _apply(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()


class Render(SimpleWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._n = 0
        self.fig = plt.figure()
        self.fig.show()

    def _apply(self, I):
        ax = self.fig.gca()
        ax.clear()
        ax.imshow(np.reshape(I, (80,80)))
        self.fig.canvas.draw()
        return I


def main():
    env = gym.make('Pong-v0')
    env = PongSimplify(env)
    #env = Render(env)
    tree_run_result = tree_agent.run(env)
    list(tree_run_result)


if __name__ == '__main__':
    main()
