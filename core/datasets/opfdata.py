import os

from torch_geometric.datasets import OPFDataset

from core.utils.registry import registry


@registry.register_dataset("opfdata")
class OPFData(OPFDataset):
    def __init__(
            self,
            split="train",
            case_name="pglib_opf_case14_ieee",
            num_groups=1,
            topological_perturbations=False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=False,
        ):
        self.root = os.path.join("data", "opfdata")
        super().__init__(
            root=self.root,
            split="train",
            case_name="pglib_opf_case14_ieee",
            num_groups=1,
            topological_perturbations=False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=False,
        )
