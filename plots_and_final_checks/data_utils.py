from idx import *
import numpy as np
import torch

def loadcase(fname):
    """
    Loads a MATPOWER case file (.m format) and returns a dictionary in PyTorch tensor format.
    """
    ppc = {
        'version': '2',
        'baseMVA': None,
        'bus': None,
        'gen': None,
        'branch': None,
        'gencost': None
    }

    with open(fname, 'r') as f:
        lines = f.readlines()
    
    current_section = None
    data_dict = {
        'bus': [],
        'gen': [],
        'branch': [],
        'gencost': []
    }

    for line in lines:
        line = line.strip()
        if line.startswith('%') or not line:
            continue

        if line.startswith('mpc.baseMVA'):
            ppc['baseMVA'] = float(line.split('=')[1].strip().replace(';', ''))
        elif line.startswith('mpc.bus'):
            current_section = 'bus'
            continue
        elif line.startswith('mpc.gen '):  # Avoid match with gencost
            current_section = 'gen'
            continue
        elif line.startswith('mpc.branch'):
            current_section = 'branch'
            continue
        elif line.startswith('mpc.gencost'):
            current_section = 'gencost'
            continue
        elif '];' in line:
            current_section = None
            continue
        
        if current_section in data_dict:
            line = line.split('%')[0].strip()
            values = line.replace(';', '').split()
            try:
                data_dict[current_section].append([float(x) for x in values])
            except ValueError:
                print(f"Warning: Skipping invalid line in {current_section}: {line}")
    
    # Convert and post-process
    for key in data_dict:
        if data_dict[key]:  # if not empty
            arr = np.array(data_dict[key], dtype=np.float32)
            ppc[key] = arr
        else:
            ppc[key] = None

    # Apply baseMVA scaling
    ppc["bus"][:, [PD, QD]] /= ppc['baseMVA']
    ppc["gen"][:, [PG, QG, QMAX, QMIN, PMAX, PMIN]] /= ppc['baseMVA']
    ppc["branch"][:, [RATE_A, RATE_B, RATE_C]] /= ppc['baseMVA']

    # Apply zero-indexing and cast index columns to int
    ppc["bus"][:, 0] = ppc["bus"][:, 0].astype(np.int32) - 1
    ppc["branch"][:, 0] = ppc["branch"][:, 0].astype(np.int32) - 1
    ppc["branch"][:, 1] = ppc["branch"][:, 1].astype(np.int32) - 1
    ppc["gen"][:, GEN_BUS] = ppc["gen"][:, GEN_BUS].astype(np.int32) - 1

    # Convert all numpy arrays to torch tensors
    for key in ['bus', 'gen', 'branch', 'gencost']:
        if ppc[key] is not None:
            ppc[key] = torch.tensor(ppc[key])

    return ppc