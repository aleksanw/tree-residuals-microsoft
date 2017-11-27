import random

class Replay_buffer(list):
    def sample(self, n):
        return random.choices(self, k=n)
