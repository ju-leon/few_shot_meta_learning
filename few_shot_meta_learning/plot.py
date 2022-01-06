from typing import List
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
import wandb


"""
    each dict should contain N-dimensional vectors under the following keys:
    'x_train', 'y_train',
    'x_test', 'y_test',
    furthermore the 'y_pred' entry should contain an NxR log_prob_heat_map and x and y vectors that index the heat_map 
"""
def plot_predictions(plotting_data: List[dict], config: dict):
    fig, axs = plt.subplots(2, len(plotting_data))
    fig.suptitle(f"noise_sttdev={config['noise_stddev']}, num_models={config['num_models']}")
    for i, data in enumerate(plotting_data):
        plot_dist(data, axs[0, i], fig)
        plot_samples(data, axs[1, i])
    if config['wandb']:
        wandb.log({"Prediction": plt})
    else:
        plt.show()


def plot_dist(data, ax, fig):
    base_plot(data, ax)
    # plot posterior predictive distribution
    max_heat = np.max(data['heat_map'])
    min_heat = np.min(data['heat_map'])
    c = ax.pcolormesh(data['x_test'], data['y_resolution'],
                      data['heat_map'], vmin=min_heat, vmax=max_heat)
    fig.colorbar(c, ax=ax)


def plot_samples(data, ax):
    base_plot(data, ax)
    # plot samples
    if data['y_pred'].shape == data['x_test'].shape:
        ax.plot(data['x_test'], data['y_pred'], linestyle='--')
        return
    for i in range(data['y_pred'].shape[0]):
        ax.plot(data['x_test'], data['y_pred'][i,:], linestyle='--')

    

def base_plot(data, ax):
    # plot ground truth
    ax.plot(data['x_test'], data['y_test'], color='black',
            linewidth=1, linestyle='-')
    # plot samples
    ax.scatter(x=data['x_train'], y=data['y_train'],
               s=40, marker='^', color='C3', zorder=2, alpha=0.75)
    # additional information
    ax.set_xlabel('x')
    ax.set_ylabel('y')