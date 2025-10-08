from notebooks.utils import check_violations
from core.datasets.pfdelta_dataset import PFDeltaDataset

case14_dataset = PFDeltaDataset(
    case_name="case14",
    task="analysis",
    feasibility_type="near infeasible",
    perturbation="n",
)

violations = check_violations(case14_dataset)
