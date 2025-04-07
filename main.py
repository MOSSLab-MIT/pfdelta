# Script that starts jobs and submits them to slurm

import os
import sys
import time

from core.utils.flags import flags
from core.utils.registry import registry
from core.utils.main_utils import (
    load_registry,
    load_config,
    console,
)


def main():
    tic = time.time()
    print("Welcome!\U00002764  Loading lab...\U0000231B")

    load_registry()

    # Load arguments
    parser = flags.parser
    args, override_args = parser.parse_known_args()

    # Load the config file
    config = load_config(args, override_args)

    # Main proceeds according to the config type
    config_type = config["config_type"]
    if config_type == "experiment":
        print("Config file loaded!\U00002714\n")
        # Initialize trainer
        trainer_name = config["functional"].get("trainer_name", "base_trainer")
        trainer_class = registry.get_trainer_class(trainer_name)
        trainer = trainer_class(config)
        print("\nTrainer initialized!\U00002714")

        print("Starting training...")
        trainer.train()
    elif "local" in config_type:
        trainers = []
        n_jobs = len(config['jobs'])
        print("\nLocal batch of experiments detected! Good luck in your runs!" \
            + "\U0001F601\n")
        print(f"There are {n_jobs} jobs ready to train. " + \
            "They will be trained sequentially!\n")
        sbatch_scripts = config["sbatch_scripts"]
        for i, cfg in enumerate(config["jobs"]):
            print("\n" + "#" * 59)  # Top border
            print("###\t\t\t\t  \t\t\t###")
            print(f"###\t\t\t\u2728  RUN #{i + 1}  \u2728\t\t\t###")
            print("###\t\t\t\t  \t\t\t###")
            print("#" * 59 + "\n")

            script = sbatch_scripts[i]
            config_name = script.split("--config ")[-1]
            sys.argv = ["main.py", "--config", config_name]
            trainer = main()
            trainers.append(trainer)
        # This last line is to not have to change the last part
        trainer = trainers
    else:
        print("\nBatch of experiments detected! " + \
            "Good luck in your runs!\U0001F601\n")
        sbatch_locations = config["sbatch_locations"]
        sbatch_scripts = config["sbatch_scripts"]
        n_jobs = len(config['jobs'])
        print(f"There are {n_jobs} jobs ready to launch. " + \
            "What do you want to do?\n")
        result = console(sbatch_locations, sbatch_scripts, config["jobs"])
        if result != 0:
            print("\nFailed to launch jobs! Deleting job submission files...")
            config_folder = os.path.split(sbatch_locations[0])[0]
            for file in os.listdir(config_folder):
                os.remove(os.path.join(config_folder, file))
            os.rmdir(config_folder)

    toc = time.time()
    print(f"Script took {toc-tic} seconds to run.")
    print("Closing lab...\U0001F63A Goodbye!")
    if config_type == "experiment":
        return trainer


if __name__ == "__main__":
    main()
