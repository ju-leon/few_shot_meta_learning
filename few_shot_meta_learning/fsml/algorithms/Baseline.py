import torch
import wandb
import typing
import os
import numpy as np
from few_shot_meta_learning.fsml.algorithms.Maml import Maml


class Baseline(Maml):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

    def task_loss(self, task_data: dict, model: dict) -> torch.Tensor:
        x = task_data[0].T
        y = task_data[1].T
        logits = self.prediction(
            x=x, adapted_hyper_net=model['hyper_net'], model=model)
        loss = self.config['loss_function'](input=logits, target=y)
        return loss

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: typing.Optional[torch.utils.data.DataLoader]) -> None:
        assert len(train_dataloader.dataset) == self.config['minibatch']
        print("Training is started.")
        print(f"Models are stored at {self.config['logdir_models']}.\n")

        print("{:<7} {:<10} {:<10} {:<10}".format(
            'Epoch', 'Base-Loss', 'NLL_train', 'NLL_validation'))
        # initialize/load model.
        model = self.load_model(
            resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, task_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        for epoch_id in range(self.config['resume_epoch'], self.config['evaluation_epoch'], 1):
            # for the Baseline loss_monitor is the loss on the whole task (not only on the validation set)
            loss_monitor = [float] * self.config['minibatch']
            for task_id, task_data in enumerate(train_dataloader):
                x = task_data[0].T
                y = task_data[1].T
                logits = self.prediction(
                    x=x, adapted_hyper_net=model['hyper_net'], model=model)
                loss = self.config['loss_function'](input=logits, target=y)
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN.")
                # calculate gradients w.r.t. hyper_net's parameters
                loss = loss / self.config['minibatch']
                loss.backward()
                loss_monitor[task_id] = loss.item()
            loss_monitor = np.sum(loss_monitor)
            loss_train = np.mean(self.evaluate(train_dataloader, model))
            # Validation
            if val_dataloader is not None:
                # turn on EVAL mode to disable dropout
                model["f_base_net"].eval()
                loss_val = np.mean(self.evaluate(val_dataloader,model))
                model["f_base_net"].train()
            # update hyper_net's parameter and reset gradient
            model["optimizer"].step()
            model["optimizer"].zero_grad()
            # Monitoring
            if self.config['wandb']:
                logging_data = {
                    'meta_train/epoch': epoch_id,
                    'meta_train/train_loss': loss_train
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
        print('Training is completed. \n')
        return None
