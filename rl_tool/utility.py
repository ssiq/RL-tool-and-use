import pandas as pd
import matplotlib.pyplot as plt


class ApproximatorAndFearureBuilder(object):
    def __init__(self, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space

    def set_observation_space(self, observation_space):
        self.observation_space = observation_space
        return self

    def set_action_space(self, action_space):
        self.action_space = action_space
        return self

    def build(self):
        pass


def flip_coin(p):
    import random
    return random.random() < p


def plot_lines(line_list, label_list, path):
    plt.ioff()
    f = plt.figure()
    line_list = []
    for line, label in zip(line_list, label_list):
        line_list.append(plt.plot(line[0], line[1], label=label))
    plt.legend(line_list)
    if path is None:
        plt.show()
    else:
        f.savefig(path)
    plt.close(f)
