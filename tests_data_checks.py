from core.datasets.pfdelta_dataset import PFDeltaDataset
from notebooks.dataset_validation_utils import (
    add_ids_to_base,
    samples_with_slack_gen_outage,
)

case_name = "case118"
root_dir = "data"
case_n = PFDeltaDataset(
    root_dir=root_dir, case_name=case_name, perturbation="n", task="analysis"
)

case_n_1 = PFDeltaDataset(
    root_dir=root_dir, case_name=case_name, perturbation="n-1", task="analysis"
)

case_n_2 = PFDeltaDataset(
    root_dir=root_dir, case_name=case_name, perturbation="n-2", task="analysis"
)

base_case = add_ids_to_base(case_n[0])

bad_samples = samples_with_slack_gen_outage(base_case, [case_n_1[13]])
print(f"Samples with slack generator outage: {bad_samples}")
print(f"Number of samples with slack generator outage: {len(bad_samples)}")
