import random
import heapq


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
        self.now_episode = []
        self.episode_id = 0

    def sample(self, size):
        if size > len(self.memory):
            return None

        r = []
        for i in xrange(size):
            t = heapq.heappop(self.memory)
            r.append(t[3])
            heapq.heappush(self.memory, (t[0]+1, t[1], t[2], t[3]))
        return r

    def add_sample(self, action_tuple, done):
        self.now_episode.append(action_tuple)
        if done:
            self.now_episode.reverse()
            self.now_episode = [(0, i, self.episode_id, t) for (i, t) in
                                zip(xrange(len(self.now_episode)), self.now_episode)]
            self.episode_id += 1
            heapq.heapify(self.now_episode)
            self.memory = list(heapq.merge(self.memory, self.now_episode))
            self.now_episode = []