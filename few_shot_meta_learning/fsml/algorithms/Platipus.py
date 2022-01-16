"""
Training PLATIPUS is quite time-consuming. One might need to train MAML, then load such parameters obtained from MAML as mu_theta to speed up the training of PLATIPUS.
"""
# from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
import random
import typing
import wandb

from copy import deepcopy
from few_shot_meta_learning.fsml._utils import kl_divergence_gaussians
from few_shot_meta_learning.fsml.HyperNetClasses import PlatipusNet
from few_shot_meta_learning.fsml.algorithms.Maml import Maml


class Platipus(object):
    def __init__(self, config: dict) -> None:
        self.config = config

        self.hyper_net_class = PlatipusNet

    def load_model(self, resume_epoch: int, task_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, task_dataloader=task_dataloader, **kwargs)

    def adapt_params(self, x: torch.Tensor, y: torch.Tensor, params: typing.List[torch.Tensor], lr: torch.Tensor, model: dict) -> typing.List[torch.Tensor]:
        q_params = [p + 0. for p in params]

        for _ in range(self.config["num_inner_updates"]):
            # predict output logits
            logits = model["f_base_net"].forward(x, params=q_params)

            # calculate classification loss
            loss = self.config['loss_function'](input=logits, target=y)

            if self.config["first_order"]:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    retain_graph=True
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    create_graph=True
                )

            for i in range(len(q_params)):
                q_params[i] = q_params[i] - lr * \
                    torch.clamp(grads[i], min=-0.5, max=0.5)

        return q_params

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> typing.List[typing.List[torch.Tensor]]:
        """Correspond to Algorithm 2 for testing
        """
        # initialize phi
        phi = [None] * self.config["num_models"]

        # get meta-parameters
        params_dict = model["hyper_net"].forward()

        # step 1 - prior distribution
        mu_theta_t = self.adapt_params(
            x=x, y=y, params=params_dict["mu_theta"], lr=params_dict["gamma_p"], model=model)

        for model_id in range(self.config["num_models"]):
            # sample theta
            theta = [None] * len(params_dict["mu_theta"])
            for i in range(len(theta)):
                theta[i] = mu_theta_t[i] + torch.randn_like(
                    input=mu_theta_t[i], device=mu_theta_t[i].device) * torch.exp(input=params_dict["log_sigma_theta"][i])

            phi[model_id] = self.adapt_params(
                x=x, y=y, params=theta, lr=self.config["inner_lr"], model=model)

        return phi

    def prediction(self, x: torch.Tensor, phi: typing.List[typing.List[torch.Tensor]], model: dict) -> typing.List[torch.Tensor]:
        logits = [None] * len(phi)
        for model_id in range(len(phi)):
            logits[model_id] = model["f_base_net"].forward(
                x, params=phi[model_id])

        return logits

    def validation_loss(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> torch.Tensor:
        params_dict = model["hyper_net"].forward()

        # adapt mu_theta - step 7 in PLATIPUS paper
        mu_theta_v = self.adapt_params(
            x=x_v, y=y_v, params=params_dict["mu_theta"], lr=params_dict["gamma_q"], model=model)

        phi = [None] * self.config["num_models"]
        for model_id in range(self.config["num_models"]):
            # step 7: sample theta from N(mu_theta, v_q^2)
            theta = [None] * len(params_dict["mu_theta"])
            for i in range(len(theta)):
                theta[i] = mu_theta_v[i] + \
                    torch.randn_like(
                        input=mu_theta_v[i], device=mu_theta_v[i].device) * torch.exp(input=params_dict["log_v_q"][i])

            # steps 8 and 9
            phi[model_id] = self.adapt_params(
                x=x_t, y=y_t, params=theta, lr=self.config["inner_lr"], model=model)

        # step 10 - adapt mu_theta to training subset
        mu_theta_t = self.adapt_params(
            x=x_t, y=y_t, params=params_dict["mu_theta"], lr=params_dict["gamma_p"], model=model)

        # step 11 - validation loss
        loss = 0
        for i in range(len(phi)):
            logits = model["f_base_net"].forward(x_v, params=phi[i])
            loss_temp = self.config['loss_function'](input=logits, target=y_v)
            loss = loss + loss_temp

        loss = loss / len(phi)

        # KL loss
        KL_loss = kl_divergence_gaussians(
            p=[*mu_theta_v, *params_dict["log_v_q"]], q=[*mu_theta_t, *params_dict["log_sigma_theta"]])

        loss = loss + self.config["KL_weight"] * KL_loss

        return loss

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, **kwargs) -> None:
        assert len(train_dataloader.dataset) == self.config['minibatch']

        print("Training is started.")
        print(f"Models are stored at {self.config['logdir_models']}.\n")
        
        print("{:<7} {:<10} {:<10} {:<10}".format(
            'Epoch', 'KL_Loss', 'NLL_train', 'NLL_validation'))
        # initialize/load model.
        model = self.load_model(
            resume_epoch=self.config["resume_epoch"], hyper_net_class=self.hyper_net_class, task_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        for epoch_id in range(self.config["resume_epoch"], self.config["evaluation_epoch"], 1):
            # for platipus loss_monitor is the KL-Loss
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
                # loss on validation subset
                loss = self.validation_loss(
                    x_t=x_t, y_t=y_t, x_v=x_v, y_v=y_v, model=model)
                if torch.isnan(input=loss):
                    raise ValueError("Loss is NaN.")
                # calculate gradients w.r.t. hyper_net"s parameters
                loss.backward()
                loss_monitor[task_id] = loss.item()
            loss_monitor = np.mean(loss_monitor)
            loss_train = np.mean(self.evaluate(train_dataloader, model))
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
                    'meta_train/train_loss': loss_train,
                    'meta_train/KL_Loss': loss_monitor
                }
                if val_dataloader is not None:
                    logging_data['meta_train/val_loss'] = loss_val
                wandb.log(logging_data)
            # Store the model and log losses in console
            if (epoch_id+1) % self.config['epochs_to_store'] == 0 or epoch_id == 0:
                loss_val = '-' if val_dataloader is None else loss_val
                print("{:<7} {:<10} {:<10} {:<10}".format(epoch_id+1,
                      np.round(loss_monitor, 4), np.round(loss_train, 4), np.round(loss_val, 4)))
                checkpoint = {
                    "hyper_net_state_dict": model["hyper_net"].state_dict(),
                    "opt_state_dict": model["optimizer"].state_dict()
                }
                checkpoint_path = os.path.join(
                    self.config['logdir_models'], f'Epoch_{epoch_id + 1}.pt')
                torch.save(obj=checkpoint, f=checkpoint_path)
        print('Training is completed.')
        return None

    def evaluate(self, task_dataloader: torch.utils.data.DataLoader, model: dict) -> typing.List[float]:
        task_losses = [float] * len(task_dataloader.dataset)
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
            phi = self.adaptation(x=x_t, y=y_t, model=model)
            # loss on validation subset
            y_pred = self.prediction(x=x_v, phi=phi, model=model)
            sample_losses = [float] * len(phi)
            for s, y in enumerate(y_pred):
                sample_losses[s] = self.config['loss_function'](
                    y, y_v).detach().item()
            task_losses[task_id] = np.mean(sample_losses)
        return task_losses

    def test(self, test_dataloader: torch.utils.data.DataLoader) -> None:
        assert len(test_dataloader.dataset) == self.config['minibatch_test']
        if self.config['minibatch_test'] == 0:
            return
        print("Evaluation is started")
        model = self.load_model(
            resume_epoch=self.config['evaluation_epoch'], hyper_net_class=self.hyper_net_class, task_dataloader=test_dataloader)
        task_losses = self.evaluate(
            task_dataloader=test_dataloader, model=model)
        print(f'NLL_test = {np.round(np.mean(task_losses), 4)} \n')
        return None
