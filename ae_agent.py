import corridor
import gym
import tensorflow

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glue as tree_agent
import autoencoder
import replay_buffer

from utils import make_data_dirs


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


class LatentSpace(SimpleWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = 1
        self.replay_buffer = replay_buffer.Replay_buffer()
        self.sample_size = 40
        self.train = True
        self.auto_train = True

    def _apply(self, I):
        I = np.reshape(I, (1, -1))
        print("Applying latent space!")

        if self.train:
            self.replay_buffer.add(I.ravel())
            if len(self.replay_buffer.buffer) == self.replay_buffer.length:
                while self.auto_train:
                    loss = autoencoder.train_on(self.replay_buffer.sample_as_nd(self.sample_size))
                    autoencoder.print_loss_and_lr(I)
                    print(f"AE loss {loss}")
                    if self.n > 5000000:
                        self.train = False
                        self.auto_train = False
                        print(f"Setting train to false, {self.train}, {loss}")
                    if self.n % 1000 == 0:
                        autoencoder.visualize_reconstruction(I, self.n, self.n)
                        autoencoder.save_checkpoint(self.n)
                    self.n += 1
            
        return autoencoder.latent_of(I)


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


def run(env, config):
    # Make directories for training checkpointing and visualization
    make_data_dirs()

    env = PongSimplify(env)
    env = LatentSpace(env)
    
    print(env)
    return tree_agent.run(env, config)
