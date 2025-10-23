

import os

best_runs_folders = {
    "CANOS_mse": [
        "canos_pf_hd128_depth15_lr0.00_seedseed1_1_250802_144426",
        "canos_pf_hd128_depth15_lr0.00_seedseed2_4_250802_160910",
        "canos_pf_hd128_depth15_lr0.00_seedseed3_7_250802_161324"
    ],
    "CANOS_constraint": [
        "canos_pf_hd128_depth15_lr0.00_seedseed1_2_250802_174118",
        "canos_pf_hd128_depth15_lr0.00_seedseed2_5_250802_174255",
        "canos_pf_hd128_depth15_lr0.00_seedseed3_8_250802_184548"
    ],
    "CANOS_constraint_with_mse": [
        "canos_pf_hd128_depth15_lr5e-4_seedseed1_0_250802_143822",
        "canos_pf_hd128_depth15_lr5e-4_seedseed2_1_250802_143822",
        "canos_pf_hd128_depth15_lr5e-4_seedseed3_2_250802_143822"
    ],
    "CANOS_pbl": [
        "canos_pf_hd128_depth15_lr0.00_seedseed1_2_250813_120644",
        "canos_pf_hd128_depth15_lr0.00_seedseed2_5_250813_133631",
        "canos_pf_hd128_depth15_lr0.00_seedseed3_8_250813_133707"
    ],
    "PFNet_mse": [
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed1_2_250730_181142",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed2_5_250730_192611",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed3_8_250730_200747"
    ],
    "PFNet_constraint": [
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed1_2_250811_175545",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed2_5_250811_181518",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed3_8_250811_183815"
    ],
    "PFNet_constraint_with_mse": [
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed1_2_250812_094938",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed2_5_250812_094938",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed3_8_250812_101830"
    ],
    "PFNet_pbl": [
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed1_PFNet_power_balance_violation_1_250820_151252",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed2_PFNet_power_balance_violation_4_250820_151252",
        "pfnet_hd256_lr0.00_layers_4_K_3_seedseed3_PFNet_power_balance_violation_7_250820_161101"
    ]
    
}
for key, model in best_runs_folders.items():
    if "CANOS" in key:
        best_runs_folders[key] = [os.path.join("runs/runs/canos_pf_task_1_3", folder_name) for folder_name in model]
    elif "PFNet" in key:
        best_runs_folders[key] = [os.path.join("runs/runs/pfnet_task_2_3", folder_name) for folder_name in model]

