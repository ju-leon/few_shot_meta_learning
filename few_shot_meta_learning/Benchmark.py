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
        evaluation_epoch = self.config['resume_epoch'] + \
            self.config['num_epochs']
        checkpoint_path = os.path.join(
            self.config['logdir'], 'Epoch_{0:d}.pt'.format(evaluation_epoch))
        if not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader=self.train_dataloader,
                            val_dataloader=None)
        if not self.config['algorithm'] == 'platipus':
            self.algo.test(
                num_eps=self.config['minbatch_test'], eps_dataloader=self.test_dataloader)

        plotting_data = self.predict_example_tasks()
        plot_predictions(plotting_data, self.config['wandb'])
        # TODO: Calculate/Query all the statistics we want to know about...

    def predict_example_tasks(self):
        # load model
        model = self.algo.load_model(
            resume_epoch=self.config['num_epochs'], hyper_net_class=self.algo.hyper_net_class, eps_dataloader=self.test_dataloader)
        sample_indices = torch.randint(
            self.config['minbatch_test'], size=(self.config['num_example_tasks'],))
        plotting_data = [None] * self.config['num_example_tasks']
        for i in range(self.config['num_example_tasks']):
            example_task = self.test_dataloader.dataset[sample_indices[i]]
            x_test, sort_indices = torch.sort(example_task[0])
            y_test = example_task[1][sort_indices]
            split_data = self.config['train_val_split_function'](
                eps_data=example_task, k_shot=self.config['k_shot'])

            # move data to GPU (if there is a GPU)
            x_train = split_data['x_t'].to(self.config['device'])
            y_train = split_data['y_t'].to(self.config['device'])

            # predict mean and standard deviation for x_test
            y_pred_std = torch.zeros_like(y_test)
            if self.config['algorithm'] == 'maml':
                adapted_hyper_net = self.algo.adaptation(
                    x=x_train[:, None], y=y_train[:, None], model=model)
                y_pred_mean = self.algo.prediction(
                    x=x_test[:, None], adapted_hyper_net=adapted_hyper_net, model=model)
            elif self.config['algorithm'] == 'platipus':
                phi = self.algo.adaptation(
                    x=x_train[:, None], y=y_train[:, None], model=model)
                y_pred = self.algo.prediction(
                    x=x_test[:, None], phi=phi, model=model)
                y_pred = torch.stack(y_pred).squeeze()
                y_pred_std, y_pred_mean = torch.std_mean(
                    y_pred, dim=0, unbiased=False)
            elif self.config['algorithm'] == 'bmaml':
                pass
            # store plotting data
            plotting_data[i] = {
                'x_test': x_test.squeeze().cpu().detach().numpy(),
                'y_test': y_test.squeeze().cpu().detach().numpy(),
                'y_pred_mean': y_pred_mean.squeeze().cpu().detach().numpy(),
                'y_pred_std': y_pred_std.squeeze().cpu().detach().numpy(),
                'x_train': x_train.squeeze().cpu().detach().numpy(),
                'y_train': y_train.squeeze().cpu().detach().numpy(),
            }
        return plotting_data
