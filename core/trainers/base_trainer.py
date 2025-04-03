import os
import copy
import yaml
import json
import sys
import types
import time
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from core.utils.registry import registry
from core.utils.trainer_utils import MultiPrinter

@registry.register_trainer("base_trainer")
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.is_debug = False
        self.datasets = []
        self.dataloaders = []
        self.train_loss = []
        self.train_loss_names = []
        self.val_loss = []
        self.val_loss_names = []
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.train_errors = {}
        self.val_errors = {}
        self.best_point = None # best_point is the best epoch or best train step
        self.best_model = None
        self.train_step = 0
        self.max_train_step = None
        self.epoch = 0
        self.max_epoch = None

        self.is_debug = self.config["functional"]["is_debug"]

        # Add unique identifier to run
        now = datetime.now()
        run_time = now.strftime("%y%m%d_%H%M%S")
        self.config["functional"]["run_name"] += "_" + run_time

        # Add prefix to run location
        run_location = self.config["functional"].get("run_location", "")
        run_location = os.path.join("runs", run_location)
        self.config["functional"]["run_location"] = run_location

        if not self.is_debug:
            self.create_run_location()

        # Save job id, if any
        jobid = os.environ.get("SLURM_JOBID", None)
        if jobid is not None:
            self.config["functional"]["slurm_jobid"] = jobid

        # Save git commit hash, if possible
        try:
            import git
            repo = git.Repo(".")
            commit_hash = repo.head.commit.hexsha
            config["functional"]["git_commit_hash"] = commit_hash
        except ModuleNotFoundError:
            print("Consider installing GitPython to be able to save" \
                + "git commit hash!")

        # Save config file
        run_location = self.config["functional"]["run_location"]
        config_location = os.path.join(run_location, "config.yaml")
        if not self.is_debug:
            with open(config_location, 'w') as f:
                yaml.dump(self.config, f, indent=2)

        # Set multiprinter
        out_location = os.path.join(run_location, "out.txt")
        if not self.is_debug:
            multiprinter = MultiPrinter(out_location)
            sys.stdout = multiprinter
            print("\U0001F4D1 Console output is now being recorded!")
        else:
            print("\U0001F4D1 is_debug flag passed, NO RESULTS WILL BE RECORDED!")

        # Report device being used
        cpu = self.config["functional"]["cpu"]
        if cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        print("\U0001F4D1 Device being used:", self.device)

        self.setup_training()

        # Report number of trainable parameters
        num_parameters = self.num_model_parameters()
        print(f"\n\U0001F4D1 Number of trainable parameters: {num_parameters}")

    def create_run_location(self,):
        run_location = self.config["functional"]["run_location"]
        run_name = self.config["functional"]["run_name"]
        run_path = os.path.join(run_location, run_name).replace("\\", "/")

        decomposed = run_path.split("/")
        path = ""
        for folder in decomposed:
            path = os.path.join(path, folder)
            if os.path.exists(path):
                continue
            try:
                os.mkdir(path)
            except FileExistsError:
                continue

        self.config["functional"]["run_location"] = run_path


    def setup_training(self,):
        self.load_dataset()
        self.load_dataloaders()
        print("\nDatasets loaded! \U00002705")
        self.load_model()
        print("Models loaded! \U00002705")
        self.load_optimizer()
        print("Optimizer loaded! \U00002705")
        self.load_loss_funcs()
        print("Loss function loaded! \U00002705")


    def load_dataset(self,):
        datasets = self.config["dataset"]["datasets"]
        for dataset in datasets:
            # Gather dataset class
            dataset_name = dataset["name"]
            dataset_class = registry.get_dataset_class(dataset_name)
            # Initialize dataset
            dataset_inputs = copy.deepcopy(dataset)
            del dataset_inputs["name"]
            dataset = dataset_class(**dataset_inputs)
            # Save dataset
            self.datasets.append(dataset)

        self.modify_datasets()


    def modify_datasets(self,):
        pass


    def load_dataloaders(self,):
        # First initialize train dataloader
        dataloader_class = self.get_dataloader_class()
        train_params = self.config["optim"]["train_params"]
        dataloader = dataloader_class(
            self.datasets[0],
            batch_size = train_params["batch_size"],
            shuffle = True,
        )
        self.dataloaders.append(dataloader)

        # Second initialize val dataloader
        val_params = self.config["optim"]["val_params"]
        if isinstance(val_params["batch_size"], int):
            batch_sizes = [val_params["batch_size"]]*len(self.datasets[1:])
        else:
            batch_sizes = val_params["batch_size"]
        for i, dataset in enumerate(self.datasets[1:]):
            batch_size = batch_sizes[i]
            dataloader = dataloader_class(
                dataset,
                batch_size = batch_size,
                shuffle = False
            )
            self.dataloaders.append(dataloader)

        self.modify_dataloaders()


    def modify_dataloaders(self,):
        pass


    def get_dataloader_class(self,):
        return DataLoader


    def load_model(self,):
        model = self.config["model"]
        # Gather model class
        model_name = model["name"]
        model_class = registry.get_model_class(model_name)
        # Gather inputs and modify them if necessary
        model_inputs = copy.deepcopy(model)
        del model_inputs["name"]
        self.customize_model_init_inputs(model_inputs)
        # Initialize model
        self.model = model_class(**model_inputs).to(self.device)


    def customize_model_init_inputs(self, model_inputs):
        r"""This method allows the user to modify the inputs before initiali-
        zation. In particular, it can be used to pass other trainer parameters
        to the model such as a data sample."""
        pass


    def load_optimizer(self,):
        # Gather optimizer class
        optimizer = self.config["optim"]["optimizer"]
        optim_name = optimizer["name"]
        optim_class = getattr(torch.optim, optim_name)
        # Initialize optimizer
        optim_inputs = copy.deepcopy(optimizer)
        del optim_inputs["name"]
        self.optimizer = optim_class(self.model.parameters(), **optim_inputs)
        # Save name for book keeping
        optim_inputs["name"] = optim_name

        # If necessary, load learning rate scheduler
        use_scheduler = "lr_scheduler" in self.config["optim"]
        if use_scheduler:
            scheduler_inputs = self.config["optim"]["lr_scheduler"]
            assert "name" in scheduler_inputs, \
                "No name found for lr scheduler!"
            scheduler_name = scheduler_inputs["name"]

            # Chained and Sequential need to be treated differently
            scheduler = None # Placeholder
            if scheduler_name in ["ChainedScheduler", "SequentialLR"]:
                assert "schedulers" in scheduler_inputs, \
                    "Chained scheduler needs scheduler lists!"
                schedulers_inputs = scheduler_inputs["schedulers"]
                # Chained schedulers need to be initialized one by one
                schedulers = []
                for sch_inputs in schedulers_inputs:
                    assert "name" in sch_inputs, \
                        "Scheduler in chain missing name!"
                    sch = self.initialize_scheduler(sch_inputs)
                    schedulers.append(sch)
                # Once all individual schedulers are initialized, we go by parts
                if scheduler_name == "ChainedScheduler":
                    sch_class = torch.optim.lr_scheduler.ChainedScheduler
                    scheduler = sch_class(schedulers)
                else:
                    processed_inputs = copy.deepcopy(scheduler_inputs)
                    processed_inputs["schedulers"] = schedulers
                    scheduler = self.initialize_scheduler(processed_inputs)
            else:
                # Single schedulers are just initialized
                scheduler = self.initialize_scheduler(scheduler_inputs)

            # Save scheduler
            self.lr_scheduler = scheduler

        self.modify_optimizer()


    def modify_optimizer(self,):
        pass


    def initialize_scheduler(self, scheduler_inputs):
        # Gather class of lr scheduler
        sch_name = scheduler_inputs["name"]
        sch_class = getattr(torch.optim.lr_scheduler, sch_name, None)
        assert sch_class is not None, f"Scheduler {sch_name} not found!"
        # Initialize class
        del scheduler_inputs["name"]
        scheduler = sch_class(optimizer=self.optimizer, **scheduler_inputs)
        # Save name back for book keeping
        scheduler_inputs["name"] = sch_name

        return scheduler


    def load_loss_funcs(self,):
        train_loss = self.config["optim"]["train_params"]["train_loss"]
        val_loss = self.config["optim"]["val_params"]["val_loss"]

        # Load train losses
        for loss in train_loss:
            # Save name
            if isinstance(loss, str):
                name = loss
            else:
                name = loss["name"]
            # Save loss method
            loss = self.initialize_loss(loss)
            self.train_loss.append(loss)
            # Update name if necessary
            name = getattr(loss, "loss_name", name)
            self.train_loss_names.append(name)

        # Load val losses
        for loss in val_loss:
            # Save name
            if isinstance(loss, str):
                name = loss
            else:
                name = loss["name"]
            # Save loss method
            loss = self.initialize_loss(loss)
            self.val_loss.append(loss)
            name = getattr(loss, "loss_name", name)
            self.val_loss_names.append(name)

        self.modify_loss()


    def modify_loss(self,):
        pass


    def initialize_loss(self, loss):
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
            initialized_loss = self.custom_loss(loss_name, loss_inputs)
        else:
            initialized_loss = loss_class(**loss_inputs)

        return initialized_loss


    def custom_loss(self, loss_name, loss_inputs):
        r"""For more custom losses, inherit the base_trainer class, register it
        in the registry, and then redo this method."""
        loss_class = registry.get_loss_class(loss_name)
        if loss_class is None:
            raise ValueError(f"Loss {loss_name} not saved in core/utils/other_losses!")

        if isinstance(loss_class, types.FunctionType):
            assert len(loss_inputs) == 0, \
                f"Custom loss {loss_name} is a function, but loss inputs were received!"
            return loss_class
        else:
            return loss_class(**loss_inputs)


    def num_model_parameters(self):
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return n


    def print_memory_reserved(self):
        gb_mreserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
        gb_reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
        print(f"\nMax GPU memory reserved: {gb_mreserved:.1f}GB\n")
        print(f"\nCurrent GPU memory reserved: {gb_reserved:.1f}GB\n")


    def train(self,):
        start = time.time()
        train_params = self.config["optim"]["train_params"]

        # Gather epoch or train step info
        assert "epochs" in train_params or "train_steps" in train_params, \
            "A max epoch or a max train_step need to be specified!"
        print()
        if "epochs" in train_params:
            assert "train_steps" not in train_params, \
                "Simulataneous epoch and train steps limit not supported!"
            self.max_epoch = train_params["epochs"]
            # print(f"\U0001F4D1 EPOCHS MODE ON. Max epoch: {self.max_epoch}")
        if "train_steps" in train_params:
            self.max_train_step = train_params["train_steps"]
            print(f"\U0001F4D1 TRAIN STEP MODE ON. Max train step: {self.max_train_step}")

        # Start training
        train_dataloader = self.dataloaders[0]
        while True:
            print()
            # Do any preliminary setup before each epoch
            self.setup_pre_epoch()
            self.model.train()

            # Train one epoch and gather losses
            running_losses = self.train_one_epoch(train_dataloader)
            losses = [loss / len(train_dataloader) for loss in running_losses]

            # Report and save results
            self.report_results(losses)

            # Do a LR scheduler step if epoch mode on
            if self.lr_scheduler is not None and self.max_epoch is not None:
                self.lr_scheduler.step()

            # Calc val error and save new best model if necessary
            self.is_val_error_time()

            # Do any post setup before each epoch
            self.setup_post_epoch()

            # End training condition
            if self.max_epoch is None:
                if self.train_step >= self.max_train_step - 1:
                    break
            else:
                if self.epoch >= self.max_epoch - 1:
                    break

            self.epoch += 1

        print("\nTRAINING FINISHED!!\U0001F9E0")
        print("Calculating final validation error...")
        self.calc_val_errors(last_time=True)
        end = time.time()
        print(f"Training finished in {(end-start):.2f} seconds.")
        print()


    def is_val_error_time(self, ):
        # Get status for either epoch or train step limit
        if self.max_epoch is None:
            current_point = self.train_step - 1
            max_point = self.max_train_step
        else:
            current_point = self.epoch
            max_point = self.max_epoch

        # Create validation error conditions
        report_every = self.config["optim"]["val_params"]["report_every"]
        is_report_time = (current_point + 1) % report_every == 0
        training_about_to_end = current_point == (max_point - 1)

        # Check if it is time to calculate val errors
        if is_report_time and not training_about_to_end:
            self.calc_val_errors()
            print("\nCONTINUE TRAINING!\U0001F9BF")


    def report_results(self, losses):
        # Print progress message
        if self.max_epoch is not None:
            current_point = self.epoch
        else:
            current_point = self.train_step

        print()
        if self.max_epoch is not None:
            message = f"Epoch {current_point+1}/{self.max_epoch} done!"
            message += f"\U0001F9BE Train step: {self.train_step}"
            print(message)
            print("-"*40)
        else:
            message = f"Train step {current_point}/{self.max_train_step} done!"
            message += f"\U0001F9BE Epoch: {self.epoch+1}"
            print(message)
            print("-"*40)

        # Record errors in train_errors
        self.train_errors[str(current_point)] = {}
        for loss_name, loss in zip(self.train_loss_names, losses):
            print(f"{loss_name}: {loss}")
            self.train_errors[str(current_point)][loss_name] = loss
        self.print_memory_reserved()

        # Save json file with errors
        run_location = self.config["functional"]["run_location"]
        train_location = os.path.join(run_location, "train.json")
        if not self.is_debug:
            with open(train_location, 'w') as f:
                json.dump(self.train_errors, f, indent=4)


    def setup_pre_epoch(self,):
        pass


    def setup_post_epoch(self,):
        pass


    def train_one_epoch(self, train_dataloader):
        message = f"Epoch {self.epoch + 1} \U0001F3CB"
        running_loss = [0.]*len(self.train_loss)
        losses = [0.]*len(self.train_loss)
        for data, labels in tqdm(train_dataloader, desc=message):
            # Move data to device
            data, labels = data.to(self.device), labels.to(self.device)

            # Calculate output and loss
            self.optimizer.zero_grad()
            outputs = self.model(data)
            for i, loss_func in enumerate(self.train_loss):
                losses[i] = loss_func(outputs, labels)
                running_loss[i] += losses[i].item()

            # Backpropagate
            losses[0].backward()
            self.grad_manip(losses)
            self.optimizer.step()

            # Update train step
            self.update_train_step()

        return running_loss


    def update_train_step(self,):
        # For per train step mode, we need two operations
        self.train_step += 1
        if self.max_epoch is None:
            # LR scheduler step if any
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Check if it is time to calc val error
            self.is_val_error_time()


    def grad_manip(self, losses):
        grad_manip = self.config["optim"]["train_params"].get("grad_manip", {})
        assert isinstance(grad_manip, dict), \
            "Grad manipulation controls are given through a dictionary!"
        if len(grad_manip) == 0:
            return None

        nan_to_zero = grad_manip.get("nan_to_zero", False)
        if nan_to_zero:
            for p in self.model.parameters():
                has_grad = p.requires_grad and p.grad is not None
                if has_grad and torch.isnan(p.grad).any():
                    p.grad = torch.nan_to_num(p.grad)

        clip_grad_norm = grad_manip.get("clip_grad", False)
        if clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)


    def calc_val_errors(self, last_time=False):
        val_dataloaders = self.dataloaders[1:]

        # Current point is either an epoch or a training step
        current_point = self.train_step if self.max_epoch is None else self.epoch

        print("\nTime to check validation error...\U0001F50D\n")
        val_params = self.config["optim"]["val_params"]
        loss_per_dataset = []
        self.val_errors[str(current_point)] = []
        for i, dataloader in enumerate(val_dataloaders):
            running_losses = self.calc_one_val_error(dataloader, i)
            losses = [loss / len(dataloader) for loss in running_losses]

            loss_per_dataset.append(losses)

        for i, val_loss in enumerate(loss_per_dataset):
            if not last_time:
                print(f"\nValidation dataset {i+1} errors \U0001F440\n"+"-"*30)
            one_val_error = {}
            for loss_name, loss in zip(self.val_loss_names, val_loss):
                one_val_error[loss_name] = loss
                if not last_time:
                    print(f"{loss_name}: {loss}")
            self.val_errors[str(current_point)].append(one_val_error)

        run_location = self.config["functional"]["run_location"]
        val_location = os.path.join(run_location, "val.json")
        if not self.is_debug:
            with open(val_location, 'w') as f:
                json.dump(self.val_errors, f, indent=2)

        self.save_model()
        self.save_summary(last_time)
        if last_time:
            self.print_summary()


    @torch.no_grad()
    def calc_one_val_error(self, val_dataloader, val_num):
        running_losses = [0.]*len(self.val_loss)
        num_val_datasets = len(self.datasets[1:])
        message = f"Processing validation set {val_num+1}/{num_val_datasets}"
        for data, labels in tqdm(val_dataloader, desc=message):
            # Move data to device
            data, labels = data.to(self.device), labels.to(self.device)

            # Calculate output and loss
            outputs = self.model(data)
            for i, loss_func in enumerate(self.val_loss):
                loss = loss_func(outputs, labels)
                running_losses[i] += loss.item()

        return running_losses


    def save_model(self,):
        first_time = self.best_point is None
        is_new_best, new_perf = self.is_new_best()            

        run_location = self.config["functional"]["run_location"]
        model_location = os.path.join(run_location, "model.pt")

        if is_new_best:
            if not self.is_debug:
                torch.save(self.model.state_dict(), model_location)
            if not first_time:
                print(f"\nNEW BEST VAL ERROR {new_perf:.3} FOUND!!\U0001F973 ")
                print("Model has best validation performance yet.")
            current_point = list(self.val_errors.keys())[-1]
            self.best_point = current_point
            self.best_model = copy.deepcopy(self.model)


    def is_new_best(self,):
        if self.best_point is None:
            only_point = list(self.val_errors.keys())[0]
            self.best_point = only_point

            return True, None

        last_point = list(self.val_errors.keys())[-1]
        old_best = self.val_errors[self.best_point]
        last_value = self.val_errors[last_point]

        # Gather leading errors
        new_perf, old_perf = self.interpret_val_performance(old_best, last_value)

        # Determine if new best has been achieved
        val_params = self.config["optim"]["val_params"]
        high_is_good = val_params.get("high_is_good", False)
        if high_is_good:
            return (new_perf >= old_perf), new_perf
        else:
            return (old_perf >= new_perf), new_perf


    def interpret_val_performance(self, old_best, last_value):
        """
        This method is used to calculate the performance of the model over the
        validation sets. The default one is to use only the leading val error
        and average it over all validation datasets.
        """
        val_params = self.config["optim"]["val_params"]
        use_first_only = val_params.get("use_first_only", False)

        # if this, then we only consider the first validation set
        if use_first_only:
            main_error = list(old_best[0].keys())[0]
            old_perf = old_best[0][main_error]
            new_perf = last_value[0][main_error]
            return new_perf, old_perf

        # else, we average over all val datasets
        main_error = list(old_best[0].keys())[0]
        old_perfs = [val_errors[main_error] for val_errors in old_best]
        old_perf_avg = sum(old_perfs) / len(old_perfs)
        new_perfs = [val_errors[main_error] for val_errors in last_value]
        new_perf_avg = sum(new_perfs) / len(new_perfs)

        return new_perf_avg, old_perf_avg


    def save_summary(self, last_time=False):
        val_errors = self.val_errors[self.best_point]
        # When train_step is on, we use the last training error, if any
        if self.max_epoch is None:
            train_entries = list(self.train_errors.keys())
            # If none, then we skip recording data.
            if len(train_entries) == 0:
                train_errors = None
            # Otherwise, gather the trian step immediatly before
            else:
                for i in range(len(train_entries)):
                    testing_point = train_entries[-(i+1)]
                    if int(testing_point) <= int(self.best_point):
                        train_errors = self.train_errors[testing_point]
                        break
        else:
            train_errors = self.train_errors[self.best_point]

        # Save results
        run_location = self.config["functional"]["run_location"]
        summary_location = os.path.join(run_location, "summary.json")

        summary = {
            "train": train_errors,
            "val": val_errors,
            "best_point": self.best_point,
            "finished": last_time
        }

        if not self.is_debug:
            with open(summary_location, 'w') as f:
                json.dump(summary, f, indent=4)


    def print_summary(self,):
        print("\n")
        print("-"*28 + " Results Summary \U0001F60E " + "-"*28)
        print()

        train_errors = self.train_errors[self.best_point]
        val_errors = self.val_errors[self.best_point]

        # Print train errors
        print("Training error\n" + "-"*18)
        for loss_name in self.train_loss_names:
            print(f"{loss_name}: {train_errors[loss_name]:.5}")

        # Print val errors
        print()
        print("Validation error(s)\n" + "-"*20)
        val_names_string = "\t\t"
        for loss_name in self.val_loss_names:
            short_name = loss_name[:14]
            if len(short_name) <= 8:
                short_name += " "*7
            val_names_string += short_name + "\t"
        print(val_names_string)

        for i, one_val_errors in enumerate(val_errors):
            val_string = f"Val set {i+1}\t"
            for loss_name in self.val_loss_names:
                val_string += f"{one_val_errors[loss_name]:.5}" + "\t\t"
            print(val_string)

        if self.max_epoch is None:
            point = "train step"
        else:
            point = "epoch"
        print(f"\nBest {point}: {int(self.best_point) + 1}")
