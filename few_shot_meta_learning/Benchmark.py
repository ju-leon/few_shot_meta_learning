from copy import deepcopy
import torch
import os
import numpy as np
import random

from few_shot_meta_learning import visualizer
from few_shot_meta_learning.fsml.algorithms.Baseline import Baseline
from few_shot_meta_learning.fsml.algorithms.Maml import Maml
from few_shot_meta_learning.fsml.algorithms.Platipus import Platipus
from few_shot_meta_learning.fsml.algorithms.Bmaml import Bmaml
from few_shot_meta_learning.fsml.HyperNetClasses import IdentityNet, NormalVariationalNet
from few_shot_meta_learning.benchmark_dataloader import create_benchmark_dataloaders


def apply_random_seed(num: int):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)


class Benchmark():
    def __init__(self, config) -> None:
        self.config = config

        apply_random_seed(config['seed'])

        self.train_dataloader, self.val_dataloader, self.test_dataloader = create_benchmark_dataloaders(
            config)

        algorithms = {
            'maml': Maml,
            'bmaml': Bmaml,
            'platipus': Platipus,
            'baseline': Baseline
        }
        self.algo = algorithms[config['algorithm']](config)

    def run(self) -> None:
        checkpoint_path = os.path.join(
            self.config['logdir_models'],
            f"Epoch_{self.config['evaluation_epoch']}.pt")

        # Only train if retraining is requested or no model with the current config exists
        if not self.config['reuse_models'] or not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader=self.train_dataloader,
                            val_dataloader=self.val_dataloader)
        self.algo.test(self.test_dataloader)

        # visualize the dataset (training, validation and testing)
        print(f"Plots are stored at {self.config['logdir_plots']}")
        visualizer.plot_tasks_initially(
            'Meta_Training_Tasks', self.algo, self.train_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Validation_Tasks', self.algo, self.val_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Testing_Tasks', self.algo, self.test_dataloader, self.config)
        # visualize predictions for training and validation tasks of each stored model
        for epoch in range(self.config['epochs_to_store'], self.config['evaluation_epoch']+1, self.config['epochs_to_store']):
            visualizer.plot_task_results(
                'Training', epoch, self.algo, self.train_dataloader, self.config)
            visualizer.plot_task_results(
                'Validation', epoch, self.algo, self.val_dataloader, self.config)
        # visualize Predictions for test tasks of only the last model
        visualizer.plot_task_results(
            'Testing', self.config['evaluation_epoch'], self.algo, self.test_dataloader, self.config)
        # TODO: Calculate/Query all the statistics we want to know about...
