import numpy as np
import os
import sys


def rollout(policy, env):
    state = np.array(env.reset()).flatten()
    done = False
    while not done:
        action = policy(state)
        newstate, reward, done, _ = env.step(action)
        newstate = np.array(newstate).flatten()
        yield state, action, reward, newstate
        state = newstate


def test_rollout(policy, env):
    reward_sum = 0
    for (_, _, reward, _) in rollout(policy, env):
        reward_sum += reward
    return reward_sum


def test_policy(policy, env):
    return np.average([test_rollout(policy, env) for _ in range(20)])


def decay(initial, t, decay=0.001):
    '''Decay variables according to microsoft paper
    '''
    return initial/(1 + decay*t)


def assert_shapetype(array, dtype, shape):
    array = np.asarray(array)
    assert (array.dtype == dtype
            and len(array.shape) == len(shape)
            and all(array.shape[i] == shape[i] for i in range(len(shape)) if shape[i] != -1)
            ), f"Shapetype mismatch: Expected {dtype}{shape}, Got {array.dtype}{array.shape}."


def v(q, state):
    return max(q(((*state, x),))[0] for x in q.action_space)


def make_data_dirs():
    directories = ['images', 'images/reconstructed',
            'images/original','checkpoints']
    for directory in directories:
        try:
            os.mkdir(directory)
        except Exception as e:
            # Directory is already there
            print(f"{e}, skipping")

def clean_data_dirs():
    rel_path = 'images'
    try:
        for folder in os.listdir(rel_path):
            for image in os.listdir(os.path.join(rel_path,folder)):
                os.remove(os.path.join(rel_path, folder, image))
    except Exception as e:
        print(e)
        sys.exit()

