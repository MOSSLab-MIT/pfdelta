PFDelta Dataset
=================================================================

This dataset contains JSON-formatted AC power flow samples generated under topological perturbations, generator cost permulations and varying  load conditions. It includes both just-feasible and close-to-infeasible cases.

Directory Structure:
--------------------

```
case_name/
└── topology_perturb/
    ├── raw.tar.gz
    ├── raw_shuffle.json
    ├── close2inf_train_nose.tar.gz
    ├── close2inf_train_around_nose.tar.gz
    ├── close2inf_test_nose.tar.gz
```

Available Cases and Perturbations:
----------------------------------
The following base cases and topological perturbation types are included:

Cases:
- case14
- case30
- case57
- case118
- case500
- case2000 Note: for case2000 only raw.tar.gz and its corresponding shuffle file is included.

Topological Perturbations:
- none      : Original topology (no line removed)
- n-1       : Single-component outage scenarios
- n-2       : Double-component outage scenarios

How to Extract:
---------------
Use the following command to extract any archive:

    tar -xzvf <archive_name>.tar.gz -C <destination_folder>

Example:

    tar -xzvf raw.tar.gz -C .

Contents and Format:
--------------------
All files are in `.json` format.

1. raw.tar.gz
   - Contains just-feasible samples.
   - Each file includes:
     - "network": a PowerModels.jl-compatible network dictionary.
     - "solution": the result of solving the AC optimal power flow.

2. close2inf_train_nose.tar.gz and close2inf_test_nose.tar.gz
   - Contains samples at the voltage stability limit (nose point) for the train and test sets, respectively.
   - Each file includes a single dictionary:
     - "solved_net": a PowerModels network dictionary that has already been updated with the solution.

3. close2inf_train_around_nose.tar.gz
   - Contains samples near the nose point.
   - Each file includes:
     - "solved_net": the updated network dictionary.
     - "lambda": the load scaling parameter used to generate this sample.

4. raw_shuffle.json
   - A shuffled mapping of sample indices, used to consistently split raw samples into training and test sets.

How to Train the Models
-----------------------

We've included configuration files in the repository that allow you to reproduce our model results easily.

To retrain a model on **Task 1.3**, simply run the following command:

```bash
python main.py --config config_<model_name>
```

Replace <model_name> with the specific model you want to train (e.g., CANOS, GNS, PFNet.). If you’d like to train on a different task, open the corresponding config file (located in `core/configs/`) and modify the lines: `task: 1.3`.

How to Use PF$\Delta$:
----------------------
### Using the Data

We provide a PyTorch dataset class to download and preprocess the raw data stored in HuggingFace. This dataset class can be prompted to load the train/val/test set for a given task in the benchmark and it preprocesses the data to enable both supervised and unsupervised methods. Each dataset item is preprocessed as a graph that contains the sufficient network data to run a standard Newton-Raphson algorithm, and to calculate Power Balance Loss, as well as a single solution that can be used as ground truth by supervised losses. An in-depth description of the data structure is located in Appendix A.6 of the PF$\Delta$ paper, and is replicated below:

    HeteroData(
      bus={
        x=[14, 2],
        y=[14, 2],
        bus_gen=[14, 2],
        bus_demand=[14, 2],
        bus_voltages=[14, 2],
        bus_type=[14],
        shunt=[14, 2],
        limits=[14, 2],
      },
      gen={
        limits=[5, 4],
        generation=[5, 2],
        slack_gen=[5],
      },
      load={ demand=[11, 2] },
      (bus, branch, bus)={
        edge_index=[2, 20],
        edge_attr=[20, 8],
        edge_label=[20, 4],
        edge_limits=[20, 1],
      },
      (gen, gen_link, bus)={ edge_index=[2, 5] },
      (bus, gen_link, gen)={ edge_index=[2, 5] },
      (load, load_link, bus)={ edge_index=[2, 11] },
      (bus, load_link, load)={ edge_index=[2, 11] }
    )


The dataset class is located in `core/datasets/pfdelta_dataset.py`, saved under the name PFDeltaDataset. A thorough description of how to use the dataset class is included in the docstrings of the class. In addition to this, we have included a notebook with examples of how to use the dataset class. This includes an example for how to modify the class to adapt the preprocessing to different models’ needs, such as homogeneous GNNs and standard feedforward networks.

Notes:
------
- All values are in per-unit (p.u.).
- All JSON files are compatible with PowerModels.jl for loading, solving, or further analysis.