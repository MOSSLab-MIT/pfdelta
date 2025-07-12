# Trainer for GNNs
import copy
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
        # import ipdb
        # ipdb.set_trace()

        running_loss = [0.]*len(self.train_loss)
        losses = [0.]*len(self.train_loss)
        message = f"Epoch {self.epoch + 1} \U0001F3CB"
        for data in tqdm(train_dataloader, desc=message):
            # Copy data in case models overwrite inputs
            data = copy.deepcopy(data)
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


    def modify_loss(self,):
        recycle_class = registry.get_loss_class("recycle_loss")
        # Add source to check if recycled is used in train
        losses = self.config["optim"]["train_params"]["train_loss"]
        names = [name if isinstance(name, str) else name["name"] for name in losses]
        if "combined_loss" in names:
            combined_loss_id = names.index("combined_loss")
            for i, loss in enumerate(self.train_loss):
                found = False
                if not isinstance(loss, recycle_class):
                    continue
                if loss.keyword == "canos_mse":
                    found = True
                    source = self.train_loss[combined_loss_id].loss1
                elif loss.keyword == "constraint_violation":
                    found = True
                    source = self.train_loss[combined_loss_id].loss2
                if found:
                    loss.source = source

        if "universal_power_balance" in names:
            universal_pbl_id = names.index("universal_power_balance")
            for i, loss in enumerate(self.train_loss):
                found = False
                if not isinstance(loss, recycle_class):
                    continue
                if loss.keyword == "pbl_pf":
                    found = True
                    source = self.train_loss[universal_pbl_id]
                if found:
                    loss.source = source

        # Add source to check if recycled is used in val
        losses = self.config["optim"]["val_params"]["val_loss"]
        names = [name if isinstance(name, str) else name["name"] for name in losses]
        if "combined_loss" in names:
            combined_loss_id = names.index("combined_loss")
            for i, loss in enumerate(self.val_loss):
                found = False
                if not isinstance(loss, recycle_class):
                    continue
                if loss.keyword == "canos_mse":
                    source = self.val_loss[combined_loss_id].loss1
                    found = True
                elif loss.keyword == "constraint_violation":
                    source = self.val_loss[combined_loss_id].loss2
                    found = True
                if found:
                    loss.source = source

        losses = self.config["optim"]["val_params"]["val_loss"]
        names = [name if isinstance(name, str) else name["name"] for name in losses]
        if "universal_power_balance" in names:
            universal_pbl_id = names.index("universal_power_balance")
            for i, loss in enumerate(self.val_loss):
                found = False
                if not isinstance(loss, recycle_class):
                    continue
                if loss.keyword == "pbl_pf":
                    source = self.val_loss[universal_pbl_id]
                    found = True
                if found:
                    loss.source = source


    def customize_model_init_inputs(self, model_inputs):
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
