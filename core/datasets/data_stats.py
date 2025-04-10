import torch


opfdata_stats = {
    "pglib_opf_case14_ieee": {
        "n_minus_one": {
            "mean": {
                "bus": {
                    "x": torch.tensor([1.0, 1.4286, 0.94, 1.06]),
                    "y": torch.tensor([-0.2389, 1.0263])
                },
                "generator": {
                    "x": torch.tensor([100., 0.43353, 0., 0.86707, 0.085126,
                                        -0.084505, 0.25476, 1., 0., 657.94, 0.]),
                    "y": torch.tensor([0.5998, 0.2096])
                },
                "load": {
                    "x": torch.tensor([0.2340, 0.0666])
                },
                "shunt": {
                    "x": torch.tensor([0.19, 0.])
                },
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor([-0.5236, 0.5236, 0.0067,
                                            0.0067, 0.0722, 0.1768,
                                            2.0632, 2.0632, 2.0632]),
                    "edge_label": torch.tensor([-0.2514, -0.004, 0.2615, 0.0247])
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor([-0.5236, 0.5236, 0., 0.3391,
                                            1.0366, 1.0366, 1.0366, 0.9597,
                                            0., 0., 0.,]),
                    "edge_label": torch.tensor([-0.3042, 0.0031, 0.3042, 0.0255])
                }
            },
            "std": {
                "bus": {
                    "x": torch.tensor([0., 0.6227, 0., 0.]),
                    "y": torch.tensor([0.0946, 0.0234])
                },
                "generator": {
                    "x": torch.tensor([0., 0.68025, 0., 1.3605, 0.067095,
                                    0.11368, 0.10043, 0., 0., 912.14, 0.]),
                    "y": torch.tensor([1.1174, 0.1209])
                },
                "load": {
                    "x": torch.tensor([0.2543, 0.0666])
                },
                "shunt": {
                    "x": torch.tensor([0., 0.])
                },
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor([0., 0., 0.0099, 0.0099, 0.0583,
                                            0.0749, 1.4880, 14880, 1.4880]),
                    "edge_label": torch.tensor([0.5278, 0.0798, 0.5431, 0.0658])
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor([0., 0., 0., 0.1545, 0.3714,
                                            0.3714, 0.3714, 0.0199, 0., 0., 0.]),
                    "edge_label": torch.tensor([0.135, 0.0705, 0.135, 0.0833])
                }
            }
        }
    },
    "pglib_opf_case500_goc": {
        "n_minus_one": {
            "mean": {
                "bus": {
                    "x": torch.tensor([135.5771, 1.228, 0.9, 1.1]),
                    "y": torch.tensor([-0.2813, 1.0682])
                },
                "generator": {
                    "x": torch.tensor([136.3, 0.9016, 0.44018, 1.363,
                                    0.24351, -0.14274, 0.62975, 1., 101.8,
                                    2637., -4.1048]),
                    "y": torch.tensor([1.0618, 0.1712])
                },
                "load": {
                    "x": torch.tensor([0.6325, 0.1633])
                },
                "shunt": {
                    "x": torch.tensor([0.5205, 0.])
                },
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor([-0.5236, 0.5236, 0.0281, 0.0281,
                                            0.0062308, 0.035974, 81.549,
                                            81.549, 81.549]),
                    "edge_label": torch.tensor([0.0369, 0.0172, -0.0314, -0.0419])
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor([-0.5236, -0.5236, -0.0035322,
                                            0.12216, 4.4074, 4.4074, 1.0188,
                                            0., 0., 0.]),
                    "edge_label": torch.tensor([1.4672, 0.2505, -1.4656, -0.1681])
                }
            },
            "std": {
                "bus": {
                    "x": torch.tensor([91.959, 0.4243, 0., 0.]),
                    "y": torch.tensor([0.1406, 0.0214])
                },
                "generator": {
                    "x": torch.tensor([187.16, 1.2559, 0.68147, 1.8716, 0.3027,
                                    0.19755, 0.80136, 0., 187.88, 830.61, 9.7501]),
                    "y": torch.tensor([1.539, 0.3083])
                },
                "load": {
                    "x": torch.tensor([0.3890, 0.1387])
                },
                "shunt": {
                    "x": torch.tensor([0.7599, 0.])
                },
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor([0., 0., 0.063017, 0.063017,
                                            0.0049197, 0.025097, 267.88,
                                            267.88, 267.88]),
                    "edge_label": torch.tensor([1.4603, 0.361, 1.4602, 0.3714])
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor([0., 0., 0.0266, 0.8397, 2.2261,
                                            2.2261, 2.2261, 0.0282, 0., 0., 0.]),
                    "edge_label": torch.tensor([1.4838, 0.3519, 1.482, 0.2942])
                }
            }
        }
    }
}
