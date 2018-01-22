import pickle
import os

def load(path):
    path = path + '.pickle'
    with open(path, 'rb') as f:
        return pickle.load(f)

def dump(agent_run):
    path = os.path.join('perfs', agent_run.env_name, f'{agent_run.agent_name}.pickle')
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError as e:
        print('Directory is already made', e)

    with open(path, 'wb') as f:
        pickle.dump(agent_run.perfs, f)
        print('Wrote data to', f.name)
