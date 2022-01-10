import torch
import os
import numpy as np
import random

from few_shot_meta_learning import visualizer
from few_shot_meta_learning.fsml.algorithms.Maml import Maml
from few_shot_meta_learning.fsml.algorithms.Platipus import Platipus
from few_shot_meta_learning.fsml.algorithms.Bmaml import Bmaml
from few_shot_meta_learning.fsml.HyperNetClasses import IdentityNet, NormalVariationalNet
from few_shot_meta_learning.benchmark_dataloader import create_benchmark_dataloaders


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

        #if not os.path.exists(checkpoint_path):
        self.algo.train(train_dataloader=self.train_dataloader,
                        val_dataloader=None)
        self.algo.test(
            num_eps=self.config['minbatch_test'], eps_dataloader=self.test_dataloader)

        # visualize a few test tasks
        visualizer.plot_visualization_tasks(self.config, self.algo, self.test_dataloader)
        # TODO: Calculate/Query all the statistics we want to know about...
