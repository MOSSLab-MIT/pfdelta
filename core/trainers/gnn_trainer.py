# Trainer for GNNs

from tqdm import tqdm

import torch
from torch_geometric.loader.dataloader import DataLoader

from core.trainers.base_trainer import BaseTrainer
from core.utils.registry import registry


@registry.register_trainer("gnn_trainer")
class GNNTrainer(BaseTrainer):
    def get_dataloader_class(self,):
        return DataLoader


    def train_one_epoch(self, train_dataloader):
        running_loss = [0.]*len(self.train_loss)
        losses = [0.]*len(self.train_loss)
        message = f"Epoch {self.epoch + 1} \U0001F3CB"
        for data in tqdm(train_dataloader, desc=message):
            # Move data to device
            data = data.to(self.device)

            # Calculate output and loss
            self.optimizer.zero_grad()
            outputs = self.model(data)
            for i, loss_func in enumerate(self.train_loss):
                losses[i] = loss_func(outputs, data)
                running_loss[i] += losses[i].item()

            # Backpropagate
            losses[0].backward()
            self.grad_manip(losses)
            self.optimizer.step()

            # Update train step
            self.update_train_step()

        return running_loss


    @torch.no_grad()
    def calc_one_val_error(self, val_dataloader, val_num):
        self.model.eval()
        running_losses = [0.]*len(self.val_loss)
        num_val_datasets = len(self.datasets[1:])
        message = f"Processing validation set {val_num+1}/{num_val_datasets}"
        for data in tqdm(val_dataloader, desc=message):
            # Move data to device
            data = data.to(self.device)

            # Calculate output and loss
            outputs = self.model(data)
            for i, loss_func in enumerate(self.val_loss):
                loss = loss_func(outputs, data)
                running_losses[i] += loss.item()

        return running_losses


    def customize_model_inputs(self, model_inputs):
        """We will use this method to pass data information to the model."""
        # First, verify that the model inputs are a dictionary
        assert type(model_inputs) == dict
        # Second, verify that the model is requiring data information
        ## This case is for the full training dataset
        if "dataset" in model_inputs and \
                model_inputs["dataset"] == "_include_":
            model_inputs["dataset"] = self.datasets[0]
        ## This case is for a single data sample
        if "data_sample" in model_inputs and \
                model_inputs["data_sample"] == "_include_":
            model_inputs["data_sample"] == self.datasets[0][0]
