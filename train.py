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

    parser.add_argument("--resume_epoch", default=0,
                        help='0 means fresh training. >0 means training continues from a corresponding stored model.')
    parser.add_argument("--evaluation_epoch", default=500,
                        help='number of epochs the model used for evaluation was trained')
    parser.add_argument("--network_architecture", default="FcNet")
    parser.add_argument("--logdir", default=".",
                        help='default location to store the saved_models directory')

    parser.add_argument("--num_ways", default=1)
    parser.add_argument("--k_shot", default=8, help='number of datapoints in the context set')
    parser.add_argument("--v_shot", default=10)
    parser.add_argument("--num_models", default=16)
    parser.add_argument("--KL_weight", default=1e-5)

    parser.add_argument("--inner_lr", default=0.01)
    parser.add_argument("--num_inner_updates", default=5, help='number of SGD steps during adaptation')
    parser.add_argument("--meta_lr", default=0.001)

    parser.add_argument("--train_flag", default=False)
    parser.add_argument("--minibatch_validation", default=32, help='completely irrelevant if we have no validation dataloader')
    parser.add_argument("--num_epochs", default=500, help='number of training epochs. one epoch corresponds to one meta update for theta')
    parser.add_argument("--minibatch", default=8, help='number of meta training tasks ')
    parser.add_argument("--points_per_minibatch", default=16, help='number of datapoints in each meta training task')
    parser.add_argument("--minbatch_test", default=128, help='number of meta testing tasks')
    parser.add_argument("--points_per_minibatch_test", default=128, help='number of datapoints in each meta testing task')
    parser.add_argument("--minibatch_print", default=1, help='1 means training and validation loss are logged to wandb after each epoch')
    parser.add_argument("--first_order", default=True,
                        help="Should always be true for MAML basd algos")

    parser.add_argument("--noise_stddev", default=0.01, help='standard deviation of the white gaussian noise added to the data targets y')
    parser.add_argument("--seed", default=123, help='general seed for everything but data generation')
    parser.add_argument("--seed_offset", default=1234, help='data generation seed for the meta training tasks')
    parser.add_argument("--seed_offset_test", default=1234, help='data generation seed for the meta testing task')
    parser.add_argument("--normalize_bm", default=True)
    parser.add_argument("--bm", default='Sinusoid1D')

    parser.add_argument("--wandb", default=False,
                        help="Specifies if logs should be written to WandB")
    parser.add_argument("--algorithm", default='maml')

    args = parser.parse_args()
    print()

    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]

    # TODO: Allow loss function selection in params
    config['loss_function'] = torch.nn.MSELoss()

    # Add arguments that cannot be set
    config['device'] = torch.device(
        'cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    config['train_val_split_function'] = train_val_split_regression
    create_save_models_directory(config)
    config['num_episodes_per_epoch'] = config['minibatch']

    benchmark = Benchmark(config)
    benchmark.run()


def create_save_models_directory(config: dict):
    config['logdir'] = os.path.join(config['logdir'], 'saved_models',
                                    config['algorithm'].lower(), config['network_architecture'])
    if not os.path.exists(path=config['logdir']):
        from pathlib import Path
        Path(config['logdir']).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
