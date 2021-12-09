from typing import List
import matplotlib.pyplot as plt
import torch
import wandb


"""
    each dict should contain N-dimensional vectors under the following keys:
    'x_train', 'y_train',
    'x_test', 'y_test',
    'y_pred_mean', 'y_pred_std'
"""
def plot_predictions(plotting_data: List[dict]):
    for i, data in enumerate(plotting_data):
        plt.subplot(2, (len(plotting_data)+1)//2, i+1)
        # plot ground truth
        plt.plot(data['x_test'], data['y_test'], color='black',
                 linewidth=1, linestyle='-')
        # plot samples
        plt.scatter(x=data['x_train'], y=data['y_train'],
                    s=80, marker='^', color='C0')
        # plot prediction mean
        plt.plot(data['x_test'], data['y_pred_mean'],
                 color='C2', linestyle='--')
        # plot confidence bounds if given
        if not all(data['y_pred_std'] == 0):
            plt.fill_between(x=data['x_test'],
                             y1=data['y_pred_mean'] - data['y_pred_std'],
                             y2=data['y_pred_mean'] + data['y_pred_std'],
                             color='C3', alpha=0.25)
        # additional information
        plt.xlabel('x')
        plt.ylabel('y')
    
    if config["wandb"]:
        wandb.log({"Prediction": plt})
    else:
        plt.show()
