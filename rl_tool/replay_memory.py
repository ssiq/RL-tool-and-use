import random


class ReplayMemory(object):
    def sample(self, size):
        pass

    def add_sample(self, action_tuple, done):
        pass


class NormalMemory(ReplayMemory):
    def __init__(self, start_size, max_size):
        self.start_size = start_size
        self.max_size = max_size
        self.memory = []

    def sample(self, size):
        if len(self.memory) > self.start_size:
            return random.sample(self.memory, size)
        else:
            return None

    def add_sample(self, action_tuple, done):
        self.memory.append(action_tuple)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)


class ProritizedMemory(ReplayMemory):
    def __init__(self, start_size, max_size):
        self.start_size = start_size
        self.max_size = max_size
        self.memory = []

    def sample(self, size):
        super(ProritizedMemory, self).sample(size)

    def add_sample(self, action_tuple, done):
        super(ProritizedMemory, self).add_sample(action_tuple, done)
        self.memory.append(action_tuple)