from cgi import test
import logging
import torch
import higher

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import typing
import os
import abc

import wandb
from few_shot_meta_learning import wandbManager

# --------------------------------------------------
# Default configuration
# --------------------------------------------------
config = {}  # initialize a configuration dictionary

# Hardware
config['device'] = torch.device('cuda:0' if torch.cuda.is_available()
                                else torch.device('cpu'))

# Dataset
config['datasource'] = 'omniglot-py'
config['suffix'] = 'png'  # extension of image file: png, jpg
config['image_size'] = (1, 64, 64)
config['ds_folder'] = './datasets'  # path to the folder containing the dataset
# load images on RAM for fast access. Set False for large dataset to avoid out-of-memory
config['load_images'] = True

# Meta-learning method
config['ml_algorithm'] = 'maml'  # either: maml and vampire
config['first_order'] = True  # applicable for MAML-like algorithms
# number of models used in Monte Carlo to approximate expectation
config['num_models'] = 1
config['KL_weight'] = 1e-4
config['dropout_prob'] = 0.2

# Task-related
config['max_way'] = 5
config['min_way'] = 5
config['k_shot'] = 1
config['v_shot'] = 15

# Training related parameters
# either CNN or ResNet18 specified in the CommonModels.py
config['network_architecture'] = 'CNN'
config['batchnorm'] = False
config['num_inner_updates'] = 5
config['inner_lr'] = 0.1
config['meta_lr'] = 1e-3
config['minibatch'] = 20  # mini-batch of tasks
config['minibatch_print'] = np.lcm(config['minibatch'], 500)
config['num_epochs'] = 1
config['resume_epoch'] = 0
# config['train_flag'] = True

# Testing
config['minibatch_validation'] = 100
# path to a csv file with row as episode name and column as list of classes that form an episode
config['episode_file'] = None

# Log
config['logdir_models'] = os.path.join('/media/n10/Data', 'meta_learning',
                                       config['ml_algorithm'], config['datasource'], config['network_architecture'])

# --------------------------------------------------
# Meta-learning class
# --------------------------------------------------


class MLBaseClass(object):
    """Meta-learning class for MAML and VAMPIRE
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, config: dict = config) -> None:
        """Initialize an instance of a meta-learning algorithm
        """
        if (config['wandb']):
            wandbManager.init(config)
        self.config = config
        return

    @abc.abstractmethod
    def load_model(self, resume_epoch: int, task_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """Load the model for meta-learning algorithm
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> typing.Union[higher.patch._MonkeyPatchBase, torch.Tensor]:
        """Task adaptation step that produces a task-specific model
        Args:
            x: training data of a task
            y: training labels of that task
            model: a dictionary consisting of
                - "hyper_net", "f_base_net", "optimizer" for MAML-like algorithms such as MAML, ABML, VAMPIRE
                - "protonet", "optimizer" for Prototypical Networks
        Returns: a task-specific model
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prediction(self, x: torch.Tensor, adapted_hyper_net: typing.Union[torch.Tensor, higher.patch._MonkeyPatchBase], model: dict) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
        """Calculate logits of data

        Args:
            x: data of a task
            adapted_hyper_net: either the prototypes of classes or the adapted hypernet
            model: dictionary consisting of the model and its optimizer

        Returns: prediction logits of data x
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: typing.Union[torch.Tensor, higher.patch._MonkeyPatchBase], model: dict) -> torch.Tensor:
        """Calculate the validation loss to update the meta-paramter

        Args:
            x: data in the validation subset
            y: corresponding labels in the validation subset
            adapted_hyper_net: either the prototypes of classes or the adapted hypernet
            model: dictionary consisting of the model and its optimizer

        Return: loss on the validation subset (might also include some regularization such as KL divergence)
        """
        raise NotImplementedError()

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: typing.Optional[torch.utils.data.DataLoader]) -> None:
        assert len(train_dataloader.dataset) == self.config['minibatch']
        print("Training is started.")
        print(f"Models are stored at {self.config['logdir_models']}.\n")

        print("{:<7} {:<10} {:<10}".format(
            'Epoch', 'NLL_train', 'NLL_validation'))
        # initialize/load model.
        model = self.load_model(
            resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, task_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        for epoch_id in range(self.config['resume_epoch'], self.config['evaluation_epoch'], 1):
            loss_monitor = [None] * self.config['minibatch']
            for task_id, task_data in enumerate(train_dataloader):
                # split data into train and validation
                split_data = self.config['train_val_split_function'](
                    task_data=task_data, k_shot=self.config['k_shot'])
                # move data to GPU (if there is a GPU)
                x_t = split_data['x_t'].to(self.config['device'])
                y_t = split_data['y_t'].to(self.config['device'])
                x_v = split_data['x_v'].to(self.config['device'])
                y_v = split_data['y_v'].to(self.config['device'])
                # adaptation on training subset
                adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)
                # loss on validation subset
                loss = self.validation_loss(x_v, y_v, adapted_hyper_net, model)
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN.")
                # calculate gradients w.r.t. hyper_net's parameters
                loss = loss / self.config['minibatch']
                loss.backward()
                loss_monitor[task_id] = loss.item()
            loss_monitor = np.sum(loss_monitor)
            # Validation
            if val_dataloader is not None:
                # turn on EVAL mode to disable dropout
                model["f_base_net"].eval()
                loss_val = np.mean(self.evaluate(val_dataloader, model))
                model["f_base_net"].train()
            # update hyper_net's parameter and reset gradient
            model["optimizer"].step()
            model["optimizer"].zero_grad()
            # Monitoring
            if self.config['wandb']:
                logging_data = {
                    'meta_train/epoch': epoch_id,
                    'meta_train/train_loss': loss_monitor
                }
                if val_dataloader is not None:
                    logging_data['meta_train/val_loss'] = loss_val
                wandb.log(logging_data)
            # Store the model and log losses in console
            if (epoch_id+1) % self.config['epochs_to_store'] == 0 or epoch_id == 0:
                loss_val = '-' if val_dataloader is None else loss_val
                print("{:<7} {:<10} {:<10}".format(epoch_id+1,
                      np.round(loss_monitor, 4), np.round(loss_val, 4)))
                checkpoint = {
                    "hyper_net_state_dict": model["hyper_net"].state_dict(),
                    "opt_state_dict": model["optimizer"].state_dict()
                }
                checkpoint_path = os.path.join(
                    self.config['logdir_models'], f'Epoch_{epoch_id + 1}.pt')
                torch.save(obj=checkpoint, f=checkpoint_path)
        print('Training is completed.\n')
        return None

    def evaluate(self, task_dataloader: torch.utils.data.DataLoader, model: dict) -> typing.List[float]:
        num_tasks = len(task_dataloader.dataset)
        task_losses = [float] * num_tasks

        for task_id, task_data in enumerate(task_dataloader):
            # split data into train and validation
            split_data = self.config['train_val_split_function'](
                task_data=task_data, k_shot=self.config['k_shot'])
            # move data to GPU (if there is a GPU)
            x_t = split_data['x_t'].to(self.config['device'])
            y_t = split_data['y_t'].to(self.config['device'])
            x_v = split_data['x_v'].to(self.config['device'])
            y_v = split_data['y_v'].to(self.config['device'])
            # adaptation on training subset
            adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)
            # loss on validation subset
            task_losses[task_id] = self.validation_loss(
                x_v, y_v, adapted_hyper_net, model).item()
        return task_losses

    def test(self, test_dataloader: torch.utils.data.DataLoader) -> None:
        assert len(test_dataloader.dataset) == self.config['minibatch_test']
        if self.config['minibatch_test'] == 0:
            return

        print("Evaluation is started.")
        model = self.load_model(
            resume_epoch=self.config['evaluation_epoch'], hyper_net_class=self.hyper_net_class, task_dataloader=test_dataloader)
        task_losses = self.evaluate(
            task_dataloader=test_dataloader, model=model)
        print(f'NLL_test = {np.round(np.mean(task_losses), 4)} \n')
        return None
