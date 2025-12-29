import torch


pfnet_pfdata_stats = {
    "case14": {
        "casename": "case14",
        "mean": {
            "bus": {
                "y": torch.tensor([0.9890, -0.3531, 0.3332, 0.1638, 0.0000, 0.0136]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0615, 0.2012, 0.0114, 0.9940, 0.0000])
            },
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0430, 0.1772, 0.8586, 0.3212, 0.0000, 0.0489]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0597, 0.1081, 0.0188, 0.0163, 0.0000])
            },
        },
    },
    "case30": {
        "casename": "case30",
        "mean": {
            "bus": {
                "y": torch.tensor([0.9678, -0.3932, 0.1893, 0.0974, 0.0000, 0.0078]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0757, 0.2000, 0.0080, 0.9963, 0.0000])
            },
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0451, 0.1488, 0.5893, 0.2596, 0.0000, 0.0347]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0794, 0.1344, 0.0150, 0.0126, 0.0000])
            },
        },
    },
    "case57": {
        "casename": "case57",
        "mean": {
            "bus": {
                "y": torch.tensor([0.9945, -0.3734, 0.3465, 0.1747, 0.0000, 0.0039]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0695, 0.2430, 0.0144, 0.9931, 0.0000])
            },
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0626, 0.3506, 1.6059, 0.5894, 0.0000, 0.0171]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.1095, 0.3016, 0.0255, 0.0225, 0.0000])
            },
        },
    },
    "case118": {
        "casename": "case118",
        "mean": {
            "bus": {
                "y": torch.tensor([1.0294, -0.6576, 0.5330, 0.2933, 0.0000, 0.0075]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0274, 0.1068, 0.0720, 0.9978, 0.0000])
            },
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0341, 0.3067, 2.4606, 0.7816, 0.0000, 0.0610]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0210, 0.0756, 0.1777, 0.0107, 0.0000])
            },
        },
    },
    "case500": {
        "casename": "case500",
        "mean": {
            "bus": {
                "y": torch.tensor([1.0776, -0.4102, 0.7608, 0.3084, 0.0000, 0.0323]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0055, 0.0587, 0.0414, 1.0050, 0.0000])
            },
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0330, 0.4644, 1.8176, 0.8199, 0.0000, 0.2271]),
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0143, 0.4335, 0.1109, 0.0167, 0.0000])
            },
        },
    },
}


opfdata_stats = {
    "pglib_opf_case14_ieee": {
        "n_minus_one": {
            "mean": {
                "bus": {
                    "x": torch.tensor([1.0, 1.4286, 0.94, 1.06]),
                    "y": torch.tensor([-0.2389, 1.0263]),
                },
                "generator": {
                    "x": torch.tensor(
                        [
                            100.0,
                            0.43353,
                            0.0,
                            0.86707,
                            0.085126,
                            -0.084505,
                            0.25476,
                            1.0,
                            0.0,
                            657.94,
                            0.0,
                        ]
                    ),
                    "y": torch.tensor([0.5998, 0.2096]),
                },
                "load": {"x": torch.tensor([0.2340, 0.0666])},
                "shunt": {"x": torch.tensor([0.19, 0.0])},
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            -0.5236,
                            0.5236,
                            0.0067,
                            0.0067,
                            0.0722,
                            0.1768,
                            2.0632,
                            2.0632,
                            2.0632,
                        ]
                    ),
                    "edge_label": torch.tensor([-0.2514, -0.004, 0.2615, 0.0247]),
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            -0.5236,
                            0.5236,
                            0.0,
                            0.3391,
                            1.0366,
                            1.0366,
                            1.0366,
                            0.9597,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ),
                    "edge_label": torch.tensor([-0.3042, 0.0031, 0.3042, 0.0255]),
                },
            },
            "std": {
                "bus": {
                    "x": torch.tensor([0.0, 0.6227, 0.0, 0.0]),
                    "y": torch.tensor([0.0946, 0.0234]),
                },
                "generator": {
                    "x": torch.tensor(
                        [
                            0.0,
                            0.68025,
                            0.0,
                            1.3605,
                            0.067095,
                            0.11368,
                            0.10043,
                            0.0,
                            0.0,
                            912.14,
                            0.0,
                        ]
                    ),
                    "y": torch.tensor([1.1174, 0.1209]),
                },
                "load": {"x": torch.tensor([0.2543, 0.0666])},
                "shunt": {"x": torch.tensor([0.0, 0.0])},
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            0.0,
                            0.0,
                            0.0099,
                            0.0099,
                            0.0583,
                            0.0749,
                            1.4880,
                            14880,
                            1.4880,
                        ]
                    ),
                    "edge_label": torch.tensor([0.5278, 0.0798, 0.5431, 0.0658]),
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.1545,
                            0.3714,
                            0.3714,
                            0.3714,
                            0.0199,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ),
                    "edge_label": torch.tensor([0.135, 0.0705, 0.135, 0.0833]),
                },
            },
        }
    },
    "pglib_opf_case500_goc": {
        "n_minus_one": {
            "mean": {
                "bus": {
                    "x": torch.tensor([135.5771, 1.228, 0.9, 1.1]),
                    "y": torch.tensor([-0.2813, 1.0682]),
                },
                "generator": {
                    "x": torch.tensor(
                        [
                            136.3,
                            0.9016,
                            0.44018,
                            1.363,
                            0.24351,
                            -0.14274,
                            0.62975,
                            1.0,
                            101.8,
                            2637.0,
                            -4.1048,
                        ]
                    ),
                    "y": torch.tensor([1.0618, 0.1712]),
                },
                "load": {"x": torch.tensor([0.6325, 0.1633])},
                "shunt": {"x": torch.tensor([0.5205, 0.0])},
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            -0.5236,
                            0.5236,
                            0.0281,
                            0.0281,
                            0.0062308,
                            0.035974,
                            81.549,
                            81.549,
                            81.549,
                        ]
                    ),
                    "edge_label": torch.tensor([0.0369, 0.0172, -0.0314, -0.0419]),
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            -0.5236,
                            0.5236,
                            0.003522,
                            0.12216,
                            4.4074,
                            4.4074,
                            4.4074,
                            1.0188,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ),
                    "edge_label": torch.tensor([1.4672, 0.2505, -1.4656, -0.1681]),
                },
            },
            "std": {
                "bus": {
                    "x": torch.tensor([91.959, 0.4243, 0.0, 0.0]),
                    "y": torch.tensor([0.1406, 0.0214]),
                },
                "generator": {
                    "x": torch.tensor(
                        [
                            187.16,
                            1.2559,
                            0.68147,
                            1.8716,
                            0.3027,
                            0.19755,
                            0.80136,
                            0.0,
                            187.88,
                            830.61,
                            9.7501,
                        ]
                    ),
                    "y": torch.tensor([1.539, 0.3083]),
                },
                "load": {"x": torch.tensor([0.3890, 0.1387])},
                "shunt": {"x": torch.tensor([0.7599, 0.0])},
                ("bus", "ac_line", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            0.0,
                            0.0,
                            0.063017,
                            0.063017,
                            0.0049197,
                            0.025097,
                            267.88,
                            267.88,
                            267.88,
                        ]
                    ),
                    "edge_label": torch.tensor([1.4603, 0.361, 1.4602, 0.3714]),
                },
                ("bus", "transformer", "bus"): {
                    "edge_attr": torch.tensor(
                        [
                            0.0,
                            0.0,
                            0.0266,
                            0.8397,
                            2.2261,
                            2.2261,
                            2.2261,
                            0.0282,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ),
                    "edge_label": torch.tensor([1.4838, 0.3519, 1.482, 0.2942]),
                },
            },
        }
    },
}
