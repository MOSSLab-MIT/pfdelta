# Set of methods to make main.py tight and tidy

import os
import glob
import yaml
import copy
import json
import subprocess
from importlib import import_module

from core.utils.registry import registry


def load_registry():
    r"""This method loads every method in the code base. The main objective is
    to allow the registry to register all necessary items. It will load 
    everything starting from the working directory.

    It was inspired by
    https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
    """

    loaded = registry.loaded
    if loaded:
        print("Registry has already been loaded!")
        return None


    # We load all modules starting at root
    root = "core"
    exception_files = []
    exception_folders = ["non_ML_models"]
    python_files = glob.glob(f"{root}/**/*.py", recursive=True)
    for file in python_files:
        file_name = os.path.split(file)[1]
        if file_name in exception_files:
            continue
        bad_file = False
        for folder in exception_folders:
            if folder in file:
                bad_file = True
        if bad_file:
            continue
        module = file[:-3].replace("\\", ".").replace("/", ".")
        import_module(module)
    registry.loaded = True


def merge_dicts(base, overwrite):
    """Recursively merge two dictionaries.
    Values in override overwrites values in base. If base and overwrite 
    contain a dictionary as a value, this will call itself recursively to merge
    these dictionaries. This does not modify the input dictionaries (creates 
    an internal copy). Will not work recursively if dict is in a list.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    and https://github.com/RolnickLab/ocp/blob/disconnected_gnn/ocpmodels/common/utils.py

    Parameters
    ----------
    base_dict: dict
        Dictionary to be overriden.
    override_dict: dict
        Dictionary whose values overwrite and/or are created in base_dict.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(base, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)} {dict1}.")
    if not isinstance(overwrite, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)} {dict2}.")

    return_dict = copy.deepcopy(base)

    for k, v in overwrite.items():
        if k not in base:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(base[k], dict):
                return_dict[k] = merge_dicts(base[k], overwrite[k])
            elif isinstance(v, list):
                assert isinstance(base[k], list),\
                    "Expected in base dictionary a list, but found a {type(base[k])}."
                return_dict[k] = copy.deepcopy(v)
            else:
                return_dict[k] = overwrite[k]

    return return_dict


def load_config(parser):
    r"""Loads the config file. The config file contains the information for one
    or multiple jobs. If it is for one, then it contains all the information it
    needs to carry out training. If it is for multiple, then it submits them to
    slurm after loading the setup for each one individually."""

    args, override_args = parser.parse_known_args()

    # Preprocess config file name
    config_location = args.config
    assert config_location != "none", "No config file specified!"
    config_location = config_location.replace("\\", "/")
    if config_location[-5:] != ".yaml":
        config_location += ".yaml"
    config_location = "core/configs/" + config_location

    # Load config file
    assert os.path.exists(config_location),\
        f"Config file {config_location} doesn't exist!"
    with open(config_location, 'r') as f:
        config = yaml.safe_load(f)

    # Handle single vs batched jobs
    config_type = config.get("config_type", "experiment")
    if config_type == "experiment":
        config = single_config(config, args, override_args)
    elif config_type in ["batch", "local_batch"]:
        config = batch_config(config, args, override_args)
    elif config_type in ["simple_batch", "local_simple_batch"]:
        config = simple_batch(config, args, override_args)
    else:
        raise ValueError(f"Config type {config_type} not identified!")

    return config


def single_config(config, args, override_args, i=None):
    r"""Given a config for one single experiment and parser, it preprocess
    inputs to construct the config file for training."""

    if "functional" not in config:
        config["functional"] = {}

    # Load functional arguments
    config["functional"] = merge_dicts(config["functional"], vars(args))

    # Verify trainer exists and set up one if there is none
    if config["functional"].get("trainer_name", None) is None:
        config["functional"]["trainer_name"] = "base_trainer"
    trainer_name = config["functional"]["trainer_name"]
    assert registry.get_trainer_class(trainer_name) is not None,\
        f"Trainer {trainer_name} not found in registry!"
    # Set run name
    if "run_name" not in config["functional"]:
        config_name = config["functional"]["config"]
        config["functional"]["run_name"] = config_name
    if i is not None: # If it is not none, then this is for job submission
        config["functional"]["run_name"] += "_" + str(i)

    # Verify config file has model, model registered in registry
    assert "model" in config, "Model dictionary missing in config!"
    assert isinstance(config["model"], dict), "Model inputs need to be a dictionary!"
    assert "name" in config["model"], "Model name not found!"
    model_name = config["model"]["name"]
    assert registry.get_model_class(model_name) is not None,\
        f"Model {model_name} not found in registry!"

    # Verify config file has dataset
    assert "dataset" in config, "Dataset dictionary missing in config!"
    assert isinstance(config["dataset"], dict), "Dataset inputs need to be a dictionary!"
    assert "datasets" in config["dataset"], "Datasets info not found!"
    assert isinstance(config["dataset"]["datasets"], list),\
        "Datasets info needs to be passed as a list! First one is for trainning."

    # Verify dataset is in registry and add common arguments to the dataset classes
    for i, dataset in enumerate(config["dataset"]["datasets"]):
        # Make uniform the way in which datasets are saved, into dicts.
        if not isinstance(dataset, dict):
            assert isinstance(dataset, str), "Invalid dataset input!"
            dataset_name = dataset
            # Relabel dataset as empty dictionary
            config["dataset"]["datasets"][i] = {"name": dataset_name}

        # Add common arguments to the dataset dictionary
        common_arguments = copy.deepcopy(config["dataset"])
        del common_arguments["datasets"]
        dataset = merge_dicts(
            common_arguments, config["dataset"]["datasets"][i])
        config["dataset"]["datasets"][i] = dataset

        # Confirm dataset name has been specified and it has been registered
        assert "name" in dataset, "Dataset name missing!"
        dataset_name = dataset["name"]
        assert registry.get_dataset_class(dataset_name) is not None,\
            f"Dataset {dataset_name} not found in registry!"


    # Create optim folder if not available
    default_optim = {
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
        },
        "train_params": {
            "epochs": 10,
            "batch_size": 128,
            "train_loss": [
                "MSELoss", # MSE
                "L1Loss", # MAE
            ]
        },
        "val_params": {
            "report_every": 10,
            "batch_size": 128,
            "val_loss": [
                "MSELoss",
                "L1Loss",
            ],
        }
    }
    if "optim" not in config:
        config["optim"] = default_optim
    else:
        # If there is an optim already, then we merge the default into it
        # Epoch num is not added if train_steps is used.
        optim = config["optim"]
        if "train_params" in optim and "train_steps" in optim["train_params"]:
            del default_optim["train_params"]["epochs"]
        config["optim"] = merge_dicts(default_optim, config["optim"])

    train_params = config["optim"]["train_params"]
    val_params = config["optim"]["val_params"]
    # Verify it has important data
    assert "epochs" in train_params or "train_steps" in train_params,\
        "Number of epochs or number of train steps missing!"
    assert "batch_size" in train_params,\
        "Batch size of training dataset missing!"
    assert "batch_size" in val_params,\
        "Batch size of validation dataset(s) missing!"
    assert "train_loss" in train_params,\
        "Trainning dataset does not have a loss!"
    assert "val_loss" in val_params,\
        "Validation dataset(s) does not have a loss!"


    # Verify the number of batch sizes is correct
    num_datasets = len(config["dataset"]["datasets"])
    if isinstance(optim["val_params"]["batch_size"], list):
        n_batch_size = 1 + len(val_params["batch_size"])
    else:
        # This case means that all val datasets get the same batch size
        assert isinstance(val_params["batch_size"], int),\
            f"Invalid type of batch size input!"
        n_batch_size = num_datasets
    assert n_batch_size == num_datasets,\
        f"Only {n_batch_size} given, but there are {num_datasets} in total!" +\
        " (train and val combined)."

    # Load override arguments
    for arg in override_args:
        place_override(arg, config)

    return config


def place_override(arg, config):
    """Allows user to modify config file from the termina. Note that
    this CANNOT be used for hyperparameter search methods like
    _manual_list."""
    assert arg[:2] == "--", f"{arg} was read as an override argument." \
        + " Override arguments should start with --!"

    # Parse argument override
    location, value_modified = arg[2:].split("=")
    location = location.split(".")

    # Apply modification
    last_key = location.pop()
    curr_dict = config
    # Find location
    while len(location) > 0:
        key = location.pop(0)
        if key in curr_dict:
            curr_dict = curr_dict[key]
        else:
            curr_dict[key] = curr_dict
            curr_dict = curr_dict[key]
    # Cast input to int or float if possible
    if value_modified[0] == "[" and value_modified[-1] == "]":
        value_modified = value_modified[1:-1]
        list_of_values = value_modified.split(",")
        value_modified = []
        for value in list_of_values:
            value = parse_value(value)
            value_modified.append(value)
    else:
        value_modified = parse_value(value_modified)
    # Save value
    curr_dict[last_key] = value_modified


def parse_value(value):
    if value.isdigit():
        return int(value)
    else:
        try:
            return float(value)
        except ValueError:
            return value



def batch_config(config, args, override_args):
    job_parameters = config.get("job_parameters", {})
    default_values = config.get("default_values", {})
    jobs = config.get("jobs", None)
    assert jobs is not None, "No jobs listed! Have them under key name 'jobs'"
    assert isinstance(jobs, list), "Jobs need to be listed as a list!"

    # Process config in default values. Other config gets loaded and overwritten
    # by the default values specified in this job.
    default_config = default_values.get("config", None)
    if default_config is not None:
        default_values = load_other_configs(default_values, default_config)

    job_config_name = args.config
    batch_folder = create_job_folder(config, job_config_name)

    processed_configs = []
    sbatch_locations = []
    sbatch_scripts = []
    job_constants = {
        "default_values": default_values,
        "args": args,
        "override_args": override_args,
        "job_parameters": job_parameters,
        "batch_folder": batch_folder,
    }
    # Let's process the configs
    i = 0
    for raw_job in jobs:
        expanded_jobs = expand_raw_job(raw_job) # This returns a list for the cases with grid search
        for job in expanded_jobs:
            process_one_job(i, job, processed_configs, sbatch_locations, sbatch_scripts, job_constants)
            i += 1

    config["jobs"] = processed_configs
    config["sbatch_locations"] = sbatch_locations
    config["sbatch_scripts"] = sbatch_scripts

    return config


def expand_raw_job(raw_job):
    expanded_jobs = []
    expand_functions = ["_manual_list"]
    raw_job_str = json.dumps(raw_job)
    # We create a queue to be able to add more items as we go
    queue = [raw_job_str]
    while len(queue) > 0:
        current_dict = queue.pop(0)
        # We first determine if there are more expansion functions left
        finished = True
        curr_func = None
        for func in expand_functions:
            search_result = current_dict.find(func)
            if search_result != -1:
                curr_func = func
                finished = False
                break
        # If none, then we save the finished dict, and we remove from queue
        if finished:
            current_dict = json.loads(current_dict)
            expanded_jobs.append(current_dict)
            continue
        # Otherwise, we proceed to expand
        start = search_result + len(curr_func) + 1
        end = current_dict.find(")", start)
        inputs = current_dict[start:end].split()
        key_end = search_result - 4
        key_start = current_dict[:key_end].rfind('"') + 1
        key = current_dict[key_start:key_end]
        # We go type by type
        if curr_func == "_manual_list":
            new_dicts = manual_list_expansion(inputs, key, current_dict, search_result, end)
            queue = new_dicts + queue

    return expanded_jobs


def manual_list_expansion(inputs, key, current_dict, search_result, end):
    start_of_dict = current_dict[:search_result-1]
    end_of_dict = current_dict[end+2:]
    # Process run names if any and process instances
    if "|" in inputs:
        idx = inputs.index("|")
        inputs.pop(idx)
        assert (len(inputs) % 2) == 0, "In manual list, number of names needs to equal number of instances!"
        num_instances = len(inputs)//2
        instances = inputs[:num_instances]
        instances = [instance[:-1] for instance in instances[:-1]] + [instances[-1]] # Delete comma at end
        names = inputs[num_instances:]
        names = [name[:-1] for name in names[:-1]] + [names[-1]] # Delete comma at end
    # Default is the first 4 characters of the instance
    else:
        instances = inputs
        instances = [instance[:-1] for instance in instances[:-1]] + [instances[-1]]
        names = [instance[:4] for instance in instances]

    # Create dictionaries
    new_dicts = []
    for instance, name in zip(instances, names):
        new_dict = start_of_dict + instance + end_of_dict
        # We modify run name if necessary
        run_name_idx = new_dict.find("run_name")
        if run_name_idx != -1:
            # Gather run name
            run_name_start = run_name_idx + 12
            run_name_end = new_dict.rfind('"', run_name_start)
            run_name = new_dict[run_name_start:run_name_end]
            # Compute new run name
            new_run_name = run_name.replace(f"%{key}", name)
            new_dict = new_dict[:run_name_start] + new_run_name + new_dict[run_name_end:]
        new_dicts.append(new_dict)

    return new_dicts

# This method assumes that all forms of grid search have been used already
def process_one_job(i, job, processed_configs, sbatch_locations, sbatch_scripts, job_constants):
    # Gather job constants
    default_values = job_constants["default_values"]
    args = job_constants["args"]
    override_args = job_constants["override_args"]
    job_parameters = job_constants["job_parameters"]
    batch_folder = job_constants["batch_folder"]
    # Deal with other config
    this_job_other_config = job.get("config", None)
    if this_job_other_config is not None:
        job = load_other_configs(job, this_job_other_config)
    # Overwrite default values with job values
    job = merge_dicts(default_values, job)
    # Create config as if it was a single (to verify integrity)
    job = single_config(job, args, override_args, i)
    # Create job parameters
    override_job_parameters = job.get("job_parameters", {})
    this_job_parameters = merge_dicts(job_parameters, override_job_parameters)
    # Change config_type to experiment
    job["config_type"] = "experiment"
    # Save slurm and config files
    sbatch_location, script = save_config_n_slurm(
        job, this_job_parameters, i, batch_folder)
    sbatch_locations.append(sbatch_location)
    sbatch_scripts.append(script)
    processed_configs.append(job)


def simple_batch(config, args, override_args):
    # Convert simple batch to batch format
    one_job_batch = {}
    if "job_parameters" in config:
        one_job_batch["job_parameters"] = config["job_parameters"]
        del config["job_parameters"]
    if "batch_name" in config:
        one_job_batch["batch_name"] = config["batch_name"]
        del config["batch_name"]
    one_job_batch["config_type"] = config["config_type"]
    one_job_batch["jobs"] = [config]
    config = batch_config(one_job_batch, args, override_args)

    return config


def create_job_folder(config, job_config_name):
    # Create batch name
    batch_name = config.get("batch_name", None)
    if batch_name is None:
        batch_name = job_config_name
    # Verify job folder does not exists and create it
    submissions_location = os.path.join("core", "configs")
    batch_folder = os.path.join(submissions_location, batch_name)
    assert_message = "It seems like this batch of jobs has already been " + \
        "submitted! Be careful not to submit the same jobs twice.\nIf this" + \
        " is not the case, make sure to delete old config submissions OR " + \
        " give your batch a new name with an entry under 'batch_name'."
    assert not os.path.exists(batch_folder), assert_message
    os.mkdir(batch_folder)

    return batch_folder


def load_other_configs(override_config, other_config):
    configs_location = os.path.join("core", "configs")
    other_config_location = os.path.join(configs_location, other_config+".yaml")
    with open(other_config_location, 'r') as f:
        other_config = yaml.safe_load(f)
    # Delete other config name
    if "config" in override_config:
        del override_config["config"]
    # Values of the other config are overwritten by override_config
    final_config = merge_dicts(other_config, override_config)

    return final_config


def save_config_n_slurm(config, job_parameters, idx, batch_folder):
    batch_name = os.path.split(batch_folder)[1]
    # Save config file
    this_job = os.path.join(batch_folder, batch_name)
    with open(this_job + "_" + str(idx) + ".yaml", 'w') as f:
        yaml.dump(config, f)

    # Create sbatch script
    sbatch_script = "#!/bin/bash\n\n"
    for parameter, value in job_parameters.items():
        if parameter == "body":
            continue

        if len(parameter) > 1 and value == "__novalue__":
            sbatch_script += f"#SBATCH --{parameter}\n"
        elif len(parameter) > 1 and value != "__novalue__":
            sbatch_script += f"#SBATCH --{parameter}={value}\n"
        elif value == "__novalue__":
            sbatch_script += f"#SBATCH -{parameter}\n"
        else:
            sbatch_script += f"#SBATCH -{parameter} {value}\n"

    # Implement launch mechanism
    config_location = f"{batch_name}/{batch_name}_{idx}"
    launch_command = f"python main.py --config {config_location}"
    default_body = f"source ~/.bashrc\n__launch__"
    body = job_parameters.get("body", "__launch__")
    body_processed = body.replace("__launch__", launch_command)
    sbatch_script += "\n" + body_processed

    # Save sbatch script
    with open(this_job + f"_{str(idx)}.sh", 'w') as f:
        f.write(sbatch_script)

    return (this_job + f"_{str(idx)}.sh", sbatch_script)


def console(sbatch_locations, sbatch_scripts, jobs):
    message = r"""
Action                                           |  Command
-------------------------------------------------------------
Launch all jobs                                  |  Launch
Print config of ith job                          |  i
Print sbatch script of ith job                   |  script i
Abort all launches                               |  Exit

Input: """

    while True:
        command = input(message)
        if command.lower() == "exit":
            return 1
        elif command == "Launch":
            for sbatch_location in sbatch_locations:
                launch_command = f"sbatch {sbatch_location}"
                out = subprocess.run(launch_command, shell=True)
                if out.returncode != 0:
                    return out.returncode

            print("\nAll jobs launched!")
            return out.returncode
        elif command[:6] == "script":
            idx = command[7:]
            if idx.isdigit():
                idx = int(idx)
                if idx + 1 > len(jobs):
                    print(f"\nThere are only {len(jobs)} jobs!\n\n")
                    continue
            else:
                print(f"\nInvalid job idx {idx}!\n\n")
            script = sbatch_scripts[idx]
            print("\n\n")
            print(script)
            print("\n\n")
        elif command.isdigit():
            idx = int(command)
            if idx + 1 > len(jobs):
                print(f"\nThere are only {len(jobs)} jobs!\n\n")
                continue
            config = jobs[idx]
            print("\n\n")
            print(json.dumps(config, indent=4))
            print("\n\n")
        else:
            print("\nInvalid input!\n\n")
