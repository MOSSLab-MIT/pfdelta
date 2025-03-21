import time
import copy
import yaml
import types
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.loader.dataloader import DataLoader

import core.utils.other_losses as other_losses
from core.utils.registry import registry


def find_run(run_name, root="runs"):
    root = Path(root)
    all_runs = []
    for folder in root.rglob(run_name):
        if folder.is_dir():
            all_runs.append(folder)

    return all_runs

def load_model(run_location, device):
    config_location = run_location / "config.yaml"
    with open(config_location, 'r') as f:
        config = yaml.safe_load(f)

    # Gather model parameters
    model_parameters = copy.deepcopy(config["model"])
    model_name = model_parameters["name"]
    del model_parameters["name"]

    # Initialize model
    model_class = registry.get_model_class(model_name)
    model = model_class(**model_parameters)

    # Upload model weights
    model_location = run_location / "model.pt"
    model.load_state_dict(torch.load(model_location, map_location=device))

    return config, model


def load_test_datasets(config):
    test_datasets = []
    for parameters in config["dataset"]["datasets"][1:]:
        dataset_parameters = copy.deepcopy(parameters)
        dataset_name = dataset_parameters["name"]
        del dataset_parameters["name"]

        # Change mode for test
        dataset_parameters["mode"] = "test"
        dataset_parameters["data_avail"] = "full"

        dataset_class = registry.get_dataset_class(dataset_name)
        dataset = dataset_class(**dataset_parameters)
        test_datasets.append(dataset)

    return test_datasets


def load_losses(config):
    loss_funcs = []
    loss_names = []
    val_loss = config["optim"]["val_params"]["val_loss"]
    val_loss.append("PBL_max")
    for loss in val_loss:
        # Save name
        if isinstance(loss, str):
            name = loss
        else:
            name = loss["name"]
        # Save loss method
        loss = initialize_loss(loss)
        loss_funcs.append(loss)
        name = getattr(loss, "loss_name", name)
        loss_names.append(name)

    return loss_funcs, loss_names

def initialize_loss(loss):
    # To deal with losses with inputs
    if isinstance(loss, dict):
        assert "name" in loss,\
            "When loading loss with inputs, name needs to be specified!"
        loss_name = loss["name"]
        loss_inputs = copy.deepcopy(loss)
        del loss_inputs["name"]
    # To deal with losses without inputs
    else:
        assert isinstance(loss, str),\
            f"Invalid loss type {type(loss)}!"
        loss_name = loss
        loss_inputs = {}

    loss_class = getattr(torch.nn, loss_name, None)
    # In the case of a custom loss, we let the user initialize it
    if loss_class is None:
        initialized_loss = other_loss(loss_name, loss_inputs)
    else:
        initialized_loss = loss_class(**loss_inputs)

    return initialized_loss


def other_loss(loss_name, loss_inputs):
    r"""For more custom losses, inherit the base_trainer class, register it
    in the registry, and then redo this method."""
    loss_class = getattr(other_losses, loss_name, None)
    if loss_class is None:
        raise ValueError(f"Loss {loss_name} not saved in core/utils/other_losses!")

    if isinstance(loss_class, types.FunctionType):
        assert len(loss_inputs) == 0, \
            f"Custom loss {loss_name} is a function, but loss inputs were received!"
        return loss_class
    else:
        return loss_class(**loss_inputs)


@torch.no_grad()
def evaluate_model(model, dataset, loss_funcs):
    running_loss = [0.]*(len(loss_funcs) - 1)
    PBL_max = 0
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=128,)
    all_predictions = []
    for data in tqdm(dataloader):
        data = data.to(device)
        outputs = model(data)
        all_predictions(outputs.cpu())
        for i, loss_func in enumerate(loss_funcs):
            loss = loss_func(outputs, data)
            if i == len(loss_funcs) - 1:
                PBL_max = max(PBL_max, loss.item())
            else:
                running_loss[i] += loss.item()

    losses = [loss / len(dataloader) for loss in running_loss]
    losses.append(PBL_max)
    return losses, all_predictions


@torch.no_grad()
def model_inference(model, dataset):
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=1,)
    # Load all inputs into the GPU
    all_inputs = []
    for data in tqdm(dataloader):
        data = data.to(device)
        all_inputs.append(data)
    # Warm up the GPU
    for data in all_inputs[:50]:
        _ = model(data)
    # Calculate all outputs
    all_outputs = []
    tic = time.time()
    for data in all_inputs:
        outputs = model(data)
        all_outputs.append(outputs)
    toc = time.time()

    all_outputs = [output.cpu() for output in all_outputs]
    seconds_per_inference = (toc - tic) / len(all_outputs)

    return all_outputs, seconds_per_inference


