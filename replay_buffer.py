import random
import numpy as np

class Replay_buffer:
    def __init__(self):
        self.buffer = []
        self.length = 30000

    def __iadd__(self, collection):
        self.buffer += collection
        return self

    def add(self, item):
        self.buffer.append(item)

    def sample(self, n):
        if len(self.buffer) >= self.length:
            self.cleanup_buffer()
        return random.choices(self.buffer, k=n)

    def sample_as_nd(self, n):
        return np.array(self.sample(n))

    def cleanup_buffer(self):
        self.buffer = self.buffer[int(self.length/5):]
        #self.buffer = self.buffer
