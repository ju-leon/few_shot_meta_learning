import wandb
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections


def plot_visualization_tasks(config, algo, test_dataloader):
    model = model = algo.load_model(
        resume_epoch=config['current_epoch'], hyper_net_class=algo.hyper_net_class, eps_dataloader=test_dataloader)
    plotting_data = [None] * config['num_visualization_tasks']
    for i in range(config['num_visualization_tasks']):
        # true target function of the task
        visualization_task = test_dataloader.dataset[i]
        x_test, sort_indices = torch.sort(visualization_task[0])
        y_test = visualization_task[1][sort_indices]

        # generate training samples and move them to GPU (if there is a GPU)
        split_data = config['train_val_split_function'](
            eps_data=visualization_task, k_shot=config['k_shot'])
        x_train = split_data['x_t'].to(config['device'])
        y_train = split_data['y_t'].to(config['device'])

        y_pred, heat_map, y_resolution = _predict_test_data(
            config, algo, model, x_train, y_train, x_test, y_test)
        plotting_data[i] = {
            'x_train': x_train.squeeze().cpu().detach().numpy(),
            'y_train': y_train.squeeze().cpu().detach().numpy(),
            'x_test': x_test.squeeze().cpu().detach().numpy(),
            'y_test': y_test.squeeze().cpu().detach().numpy(),
            'y_pred': y_pred.squeeze().detach().numpy(),
            'heat_map': heat_map.cpu().detach().numpy(),
            'y_resolution': y_resolution.detach().numpy(),
        }
    # plot the plotting data
    _generate_plots(plotting_data, config)


def _predict_test_data(config, algo, model, x_train, y_train, x_test, y_test):
    S = 1 if config['algorithm'] == 'maml' else config['num_models']
    N = config['points_per_minibatch_test']  # equals x_test.shape[0]
    R = config['y_plotting_resolution']
    noise_var = config['noise_stddev']**2

    # predict x_test
    adapted_hyper_net = algo.adaptation(
        x_train[:, None], y_train[:, None], model)
    y_pred = algo.prediction(x_test[:, None], adapted_hyper_net, model)
    if config['algorithm'] == 'platipus' or config['algorithm'] == 'bmaml':
        # platipus/bmaml return no tensor but a list of S tensors
        y_pred = torch.stack(y_pred)
    y_pred = torch.broadcast_to(y_pred, (S, N, 1))

    # discretize the relevant space of y-values
    y_combined = torch.concat([y_test[None, :, None], y_pred])
    start, end = (torch.min(y_combined).data, torch.max(y_combined).data)
    y_resolution = torch.linspace(start, end, R)
    y_broadcasted = torch.broadcast_to(y_resolution, (1, N, R))

    # generate heat_map with density values at the discretized points
    heat_maps = torch.exp(-(y_broadcasted-y_pred)**2/(
        2*noise_var)) / np.sqrt(2*torch.pi*noise_var)
    heat_map = torch.mean(heat_maps, axis=0)
    heat_map = heat_map[1:, 1:].T
    return y_pred, heat_map, y_resolution


# ==============================================
# =================Plotting=====================
# ==============================================

def _generate_plots(plotting_data, config):
    figure_counter = {
        'num': config['current_epoch'] if config['plot_each_saved_model'] else config['evaluation_epoch']
    }
    fig, axs = plt.subplots(2, len(plotting_data), **figure_counter)
    # plot the data
    for i, data in enumerate(plotting_data):
        _plot_distribution(data, axs[0, i], fig)
        _plot_samples(data, axs[1, i])
    # add cosmetics
    fig.suptitle(
        f"epochs={figure_counter['num']}, noise_sttdev={config['noise_stddev']}, num_models={config['num_models']}, \n minbatch_test={config['minbatch_test']}, points_per_minibatch_test={config['points_per_minibatch_test']}, y_plotting_resolution={config['y_plotting_resolution']}")
    fig.set_figwidth(12)
    plt.tight_layout()
    # save the plot
    save_path = os.path.join(
        config['logdir_plots'], f"Epoch_{config['current_epoch']}")
    plt.savefig(save_path)
    if config['wandb']:
        wandb.log({"Prediction": plt})


def _plot_distribution(data, ax, fig):
    _base_plot(data, ax)
    # plot posterior predictive distribution
    max_heat = np.max(data['heat_map'])
    min_heat = np.min(data['heat_map'])
    c = ax.pcolormesh(data['x_test'], data['y_resolution'],
                      data['heat_map'], vmin=min_heat, vmax=max_heat)
    fig.colorbar(c, ax=ax)


def _plot_samples(data, ax):
    _base_plot(data, ax)
    # plot samples
    if data['y_pred'].shape == data['x_test'].shape:
        ax.plot(data['x_test'], data['y_pred'], linestyle='--')
        return
    for i in range(data['y_pred'].shape[0]):
        ax.plot(data['x_test'], data['y_pred'][i, :], linestyle='--')


def _base_plot(data, ax):
    # plot ground truth
    ax.plot(data['x_test'], data['y_test'], color='black',
            linewidth=1, linestyle='-')
    # plot samples
    ax.scatter(x=data['x_train'], y=data['y_train'],
               s=40, marker='^', color='C3', zorder=2, alpha=0.75)
    # additional information
    ax.set_xlabel('x')
    ax.set_ylabel('y')
