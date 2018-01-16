import pickle
import os

def load(path):
    path = path + '.pickle'
    with open(path, 'rb') as f:
        return pickle.load(f)

def dump(path, data):
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError as e:
        print('Directory is already made', e)

    with open(path, 'wb') as f:
        pickle.dump(data, f)
        print('Wrote data to', f.name)
