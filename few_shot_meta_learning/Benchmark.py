import torch
import os
import numpy as np
import random

from few_shot_meta_learning import visualizer
import few_shot_meta_learning
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
            self.config['logdir_models'],
            f"Epoch_{self.config['evaluation_epoch']}.pt")

        print(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader=self.train_dataloader,
                            val_dataloader=None)
        self.algo.test(
            num_eps=self.config['minbatch_test'], eps_dataloader=self.test_dataloader)

        # visualize a few test tasks
        if self.config['plot_each_saved_model']:
            start = step = self.config['epochs_to_store']
        else:
            start = self.config['evaluation_epoch']
            step = 1
        stop = self.config['evaluation_epoch'] + step
        for epoch in range(start, stop, step):
            apply_random_seed(self.config['seed'])
            self.config['current_epoch'] = epoch
            visualizer.plot_visualization_tasks(
                self.config, self.algo, self.test_dataloader)

        # TODO: Calculate/Query all the statistics we want to know about...
