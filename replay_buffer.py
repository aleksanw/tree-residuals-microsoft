import random

class Replay_buffer:
    def __init__(self):
        self.buffer = []

    def __iadd__(self, collection):
        self.buffer += collection
        return self

    def add(self, item):
        self.buffer.append(item)

    def sample(self, n):
        return random.choices(self.buffer, k=n)
