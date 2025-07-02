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

Notes:
------
- All values are in per-unit (p.u.).
- All JSON files are compatible with PowerModels.jl for loading, solving, or further analysis.