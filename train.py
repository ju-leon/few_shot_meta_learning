import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
import os
import argparse

from few_shot_meta_learning.fsml._utils import train_val_split_regression
from few_shot_meta_learning.Benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("--resume_epoch", default=0)
    parser.add_argument("--network_architecture", default="FcNet")
    parser.add_argument("--logdir", default="./saved_models/fsml/maml/FcNet")

    parser.add_argument("--num_ways", default=1)
    parser.add_argument("--k_shot", default=8)
    parser.add_argument("--v_shot", default=10)
    parser.add_argument("--num_models", default=16)
    parser.add_argument("--KL_weight", default=1e-5)

    parser.add_argument("--inner_lr", default=0.01)
    parser.add_argument("--num_inner_updates", default=5)
    parser.add_argument("--meta_lr", default=0.001)

    parser.add_argument("--train_flag", default=False)
    parser.add_argument("--num_episodes", default=100)
    parser.add_argument("--num_epochs", default=500)
    parser.add_argument("--num_episodes_per_epoch", default=8)
    parser.add_argument("--minibatch", default=8)
    parser.add_argument("--points_per_minibatch", default=16)
    parser.add_argument("--minbatch_test", default=128)
    parser.add_argument("--points_per_minibatch_test", default=128)
    parser.add_argument("--minibatch_print", default=1)
    parser.add_argument("--first_order", default=True,
                        help="Should always be true for MAML basd algos")

    parser.add_argument("--noise_stddev", default=0.01)
    parser.add_argument("--seed", default=123)
    parser.add_argument("--seed_offset", default=1234)
    parser.add_argument("--seed_offset_test", default=1234)
    parser.add_argument("--normalize_bm", default=True)
    parser.add_argument("--n_points_pred", default=100)
    parser.add_argument("--bm", default='Quadratic1D')
    

    parser.add_argument("--wandb", default=True,
                        help="Specifies if logs should be written to WandB")
    parser.add_argument("--algorithm", default='maml')

    args = parser.parse_args()

    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]

    # TODO: Allow loss function selection in params
    config['loss_function'] = torch.nn.MSELoss()

    # Add arguments that cannot be set
    config['device'] = torch.device(
        'cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    config['train_val_split_function'] = train_val_split_regression

    benchmark = Benchmark(config)

    benchmark.run()


if __name__ == "__main__":
    main()
