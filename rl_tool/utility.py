import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def plot_lines(line_list, label_list, path=None, colormap_name='gist_ncar', lut=5):
    plt.ioff()
    f = plt.figure()
    lines_list = []
    color_map = cm.get_cmap(colormap_name, lut)
    for line, i in zip(line_list, xrange(len(line_list))):
        t, = plt.plot(line[0], line[1], color=color_map(i))
        lines_list.append(t)
    plt.legend(lines_list, label_list)
    if path is None:
        plt.show()
    else:
        f.savefig(path)
    plt.close(f)
