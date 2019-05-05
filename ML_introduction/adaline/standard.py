from init_obj import create_universe, show_universe
import numpy as np
import matplotlib.pyplot as plt


def show_universe(groups):
    colors=['red', 'green', 'blue', 'lightgreen', 'gray', 'cyan']
    markers=['o', 'x', '^', 'v', 's']    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    for (i, group) in enumerate(groups):
        ax[0].scatter(np.array(group).T[0], np.array(group).T[1], color=colors[i], marker=markers[i], label=chr(ord('A')+i))
    ax[0].set_xlabel("x Attribute")
    ax[0].set_ylabel("y Attribute")
    ax[0].legend(loc="upper right")

    std_groups = []

    population = np.array(groups[0] + groups[1])

    std_group = np.copy(population)
    std_group.T[0] = (std_group.T[0] - std_group.T[0].mean()) / std_group.T[0].std()
    std_group.T[1] = (std_group.T[1] - std_group.T[1].mean()) / std_group.T[1].std()
    std_groups.append(std_group)

    for (i, group) in enumerate(std_groups):
        ax[1].scatter(group.T[0], group.T[1], color=colors[i], marker=markers[i], label=chr(ord('A')+i))
    ax[1].set_xlabel("x Attribute")
    ax[1].set_ylabel("y Attribute")
    ax[1].legend(loc="upper right")

    

    plt.show()



groups = create_universe()

show_universe(groups)
