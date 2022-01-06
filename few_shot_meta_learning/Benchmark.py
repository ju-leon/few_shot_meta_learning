import torch
import os
import numpy as np
import random

from few_shot_meta_learning.fsml.algorithms.Maml import Maml
from few_shot_meta_learning.fsml.algorithms.Platipus import Platipus
from few_shot_meta_learning.fsml.algorithms.Bmaml import Bmaml
from few_shot_meta_learning.fsml.HyperNetClasses import IdentityNet, NormalVariationalNet
from few_shot_meta_learning.benchmark_dataloader import create_benchmark_dataloaders
from few_shot_meta_learning.plot import plot_predictions


class Benchmark():
    def __init__(self, config) -> None:
        self.config = config

        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

        # TODO: Add validation dataloader?
        self.train_dataloader, self.test_dataloader = create_benchmark_dataloaders(
            config)

        algorithms = {
            'maml': Maml,
            'bmaml': Bmaml,
            'platipus': Platipus,
        }
        self.algo = algorithms[config['algorithm']](config)

    def run(self) -> None:
        checkpoint_path = os.path.join(
            self.config['logdir'], 'Epoch_{0:d}.pt'.format(self.config['evaluation_epoch']))
        if not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader=self.train_dataloader,
                            val_dataloader=None)
        self.algo.test(
            num_eps=self.config['minbatch_test'], eps_dataloader=self.test_dataloader)

        plotting_data = self.predict_visualization_tasks()
        plot_predictions(plotting_data, self.config)
        # TODO: Calculate/Query all the statistics we want to know about...

    """
        generates plotting data of the first num_visualization_tasks test_tasks
    """

    def predict_visualization_tasks(self):
        # load model
        model = self.algo.load_model(
            resume_epoch=self.config['evaluation_epoch'], hyper_net_class=self.algo.hyper_net_class, eps_dataloader=self.test_dataloader)
        plotting_data = [None] * self.config['num_visualization_tasks']
        for i in range(self.config['num_visualization_tasks']):
            # true target function of the task
            visualization_task = self.test_dataloader.dataset[i]
            x_test, sort_indices = torch.sort(visualization_task[0])
            y_test = visualization_task[1][sort_indices]

            # generate training samples and move them to GPU (if there is a GPU)
            split_data = self.config['train_val_split_function'](
                eps_data=visualization_task, k_shot=self.config['k_shot'])
            x_train = split_data['x_t'].to(self.config['device'])
            y_train = split_data['y_t'].to(self.config['device'])

            # prepare output data structure
            # equals x_test.shape[0]
            N = self.config['points_per_minibatch_test']
            R = self.config['y_plotting_resolution']
            noise_var = self.config['noise_stddev']**2

            # calculcate posterior predictive distribution
            if self.config['algorithm'] == 'maml':
                adapted_hyper_net = self.algo.adaptation(
                    x=x_train[:, None], y=y_train[:, None], model=model)
                y_pred = self.algo.prediction(
                    x=x_test[:, None], adapted_hyper_net=adapted_hyper_net, model=model)
                start = torch.min(torch.concat(
                    (y_test[:, None], y_pred))).data
                end = torch.max(torch.concat(
                    (y_test[:, None], y_pred))).data
                y_resolution = torch.linspace(start, end, R)
                y_broadcasted = torch.broadcast_to(y_resolution, (N, R))
                log_prob_heat_map = - ((y_broadcasted - y_pred)**2 /
                                       noise_var - torch.pi*noise_var)[1:, 1:].T
            elif self.config['algorithm'] == 'platipus':
                phi = self.algo.adaptation(
                    x=x_train[:, None], y=y_train[:, None], model=model)
                y_pred = self.algo.prediction(
                    x=x_test[:, None], phi=phi, model=model)
                # discretize the relevant space of y-values
                y_combined = torch.concat([y_test[:, None]] + y_pred)
                start = torch.min(y_combined).data
                end = torch.max(y_combined).data
                y_resolution = torch.linspace(start, end, R)
                y_broadcasted = torch.broadcast_to(y_resolution, (1, N, R))
                # generate heat_map with density values at the discretized points
                y_pred = torch.stack(y_pred)
                heat_maps = torch.exp(-(y_broadcasted-y_pred)**2/(
                    2*noise_var)) / np.sqrt(2*torch.pi*noise_var)
                heat_map = torch.mean(heat_maps, axis=0)
                heat_map = heat_map[1:, 1:].T
            elif self.config['algorithm'] == 'bmaml':
                pass
            # store plotting data
            plotting_data[i] = {
                'x_test': x_test.squeeze().cpu().detach().numpy(),
                'y_test': y_test.squeeze().cpu().detach().numpy(),
                'x_train': x_train.squeeze().cpu().detach().numpy(),
                'y_train': y_train.squeeze().cpu().detach().numpy(),
                'heat_map': heat_map.cpu().detach().numpy(),
                'y_resolution': y_resolution.detach().numpy(),
                'y_pred': y_pred.squeeze().detach().numpy()
            }
        return plotting_data
