import torch


pfnet_pfdata_stats = {
    "case14_seeds": {
        "casename": "case14_seeds",
        "mean": {
            "bus": {
                "y": torch.tensor([ 0.9929, -0.3463,  0.3324,  0.1616,  0.0000,  0.0136])
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0615, 0.2012, 0.0114, 0.9939, 0.0000])
            }
        },
        "std": {
            "bus": {
                "y": torch.tensor([0.0394, 0.1720, 0.8527, 0.3247, 0.0000, 0.0489])
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0597, 0.1081, 0.0188, 0.0163, 0.0000])
            }
        }
    },
    "case118_seeds": {
        "casename": "case118_seeds", 
        "mean": {
            "bus": {
                "x": torch.tensor([4.7212e-01, 6.6688e-27, 2.8586e-01, 9.0312e-02, 0.0000e+00, 7.4576e-03]),
                "y": torch.tensor([ 1.0280, -0.5947,  0.4654,  0.2504,  0.0000,  0.0075])
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0274, 0.1068, 0.0720, 0.9978, 0.0000])
            }
        },
        "std": {
            "bus": {
                "x": torch.tensor([5.1877e-01, 1.1657e-23, 1.6960e+00, 1.3489e-01, 0.0000e+00, 6.1039e-02]),
                "y": torch.tensor([0.0328, 0.3659, 2.6608, 0.8317, 0.0000, 0.0610])
            },
            ("bus", "branch", "bus"): {
                "edge_attr": torch.tensor([0.0210, 0.0756, 0.1777, 0.0107, 0.0000])
            }
        }
    }
}

canos_pfdelta_stats = {
    "case118_seeds": {
        "mean":{
            "bus": {
                    "x": torch.tensor([0.3307, 0.5845]),
                    "y": torch.tensor([0.0151, 0.2691]), 
                    "bus_gen": torch.tensor([0.5525, 0.3173]),
                    "bus_demand": torch.tensor([0.5162, 0.2398]),
                    "bus_voltages": torch.tensor([-0.6396, 1.0290]),
                    "shunt": torch.tensor([0.0, 0.0075]),
                },
            "PQ": {
                "x": torch.tensor([0.4347, 0.2022]), 
                "y": torch.tensor([-0.6596,  1.0152]),
            }, 
            "PV": {
                "x": torch.tensor([0.2091, 1.0454]), 
                "y": torch.tensor([0.4380, -0.6271]),
                "generation": torch.tensor([0.8353, 0.7286]), 
                "demand": torch.tensor([0.6262, 0.2907]),
            }, 
            "slack": {
                "x": torch.tensor([3.4637e-18, 1.0583e+00]), 
                "y": torch.tensor([21.3432, -0.8125]),
                "generation": torch.tensor([21.3432, -0.8125]), 
                "demand": torch.tensor([0., 0.])
            },
            ("bus", "ac_line", "bus"): {
                "edge_attr": torch.tensor([0.0274, 0.1068, 0.0000, 
                                            0.0360, 0.0000, 0.0360, 
                                            0.9978, 0.0000]),
                "edge_label": torch.tensor([ 0.0920,  0.0767, 
                                             -0.0690, -0.0228])
            },
        },
        "std": {
            "bus": {
                    "x": torch.tensor([1.4630, 0.4419]),
                    "y": torch.tensor([2.2695, 0.85781]), 
                    "bus_gen": torch.tensor([2.3803, 1.0612]),
                    "bus_demand": torch.tensor([0.9875, 0.5016]),
                    "bus_voltages": torch.tensor([0.3231, 0.0352]),
                    "shunt": torch.tensor([0.0, 0.0610]),
                },
            "PQ": {
                "x": torch.tensor([0.3378, 0.1852]),
                "y": torch.tensor([0.3126, 0.0378]),
            }, 
            "PV": {
                "x": torch.tensor([2.1543, 0.0224]), 
                "y": torch.tensor([1.1555, 0.3258]),
                "generation": torch.tensor([1.8244, 1.4756]), 
                "demand": torch.tensor([1.4236, 0.7197]),
            }, 
            "slack": {
                "x": torch.tensor([7.6358e-16, 9.7875e-03]), 
                "y": torch.tensor([6.1415, 1.3867]),
                "generation": torch.tensor([6.1415, 1.3867]), 
                "demand": torch.tensor([0., 0.]),
            }, 
            ("bus", "ac_line", "bus"): {
                "edge_attr": torch.tensor([0.0210, 0.0756, 0.0000, 
                                            0.0888, 0.0000, 0.0888, 
                                            0.0107, 0.0000]),
                "edge_label": torch.tensor([1.5430, 0.5323, 
                                             1.5317, 0.5184])
            },
        } 
    }
}


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
                    "edge_attr": torch.tensor([-0.5236, 0.5236, 0.003522,
                                            0.12216, 4.4074, 4.4074, 4.4074,
                                            1.0188, 0., 0., 0.]),
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
