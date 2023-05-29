# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from PSO import PSO
import numpy as np


def sphere_function(x):
    return np.sum(x ** 2)


import numpy as np



if __name__ == '__main__':

    max_iterations = [10, 15, 20, 25, 30, 35]
    swarm_sizes = [10, 20, 30, 40, 50, 60]


    for it in max_iterations:
        for swarm_size in swarm_sizes:
            pso = PSO(lower_bound=-100, upper_bound=100, swarm_size=swarm_size, no_features=10, no_iterations=it,
                      fitness_func=sphere_function)
            pso.start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
