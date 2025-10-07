from core.datasets.pfdelta_dataset import PFDeltaDataset

case14_data = PFDeltaDataset(
    root_dir="data",
    case_name="case14",
    perturbation="none",
    task=3.1,
    feasibility_type="feasible",
    n_samples=10,
    force_reload=True,
)

print(case14_data[0])

print("Number of samples:", len(case14_data))
print("Data successfully loaded!")
