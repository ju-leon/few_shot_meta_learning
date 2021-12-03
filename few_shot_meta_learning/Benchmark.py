import torch
import os
import numpy as np

from few_shot_meta_learning.fsml.algorithms.Maml import Maml
from few_shot_meta_learning.fsml.HyperNetClasses import IdentityNet, NormalVariationalNet
from few_shot_meta_learning.benchmark_dataloader import create_benchmark_dataloaders


class Benchmark():
    def __init__(self, config) -> None:
        self.config = config

        # TODO: Add validation dataloader?
        self.train_dataloader, self.test_dataloader = create_benchmark_dataloaders(
            config)

        # TODO: select proper algorithm specified in config['algorithm']
        self.algo = Maml(config)

    def run(self) -> None:

        self.algo.train(train_dataloader=self.train_dataloader,
                        val_dataloader=None)
        model = self.algo.load_model(
            resume_epoch=self.config['resume_epoch'],
            eps_dataloader=self.train_dataloader,
            hyper_net_class=IdentityNet)

        # TODO: Calculate/Query all the statistics we want to know about...
