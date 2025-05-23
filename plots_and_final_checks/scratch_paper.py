import sys
import os
from classes_for_analysis import PFDeltaDataset

import torch
from torch_geometric.datasets import OPFDataset
import pandas 
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random 
import matplotlib.patches as mpatches
from data_utils import loadcase 
from idx import *
import seaborn as sns

# Load new_PFDelta without seed expansion
pwd = os.getcwd()
print(pwd)
root = 'data/pfdelta' # double-check path
print("Looking for raw data in:", os.path.join(root, "case118", "none", 'raw'))
pfdelta = PFDeltaDataset(
    root_dir=root,
    split='all', 
    case_name="case118",
    topo_perturb='none',
    force_reload=False
)

