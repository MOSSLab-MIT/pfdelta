# These are all classes for data analysis
import os
import json
import torch
from torch_geometric.data import InMemoryDataset, HeteroData


# Do only nose cases first, then extend the class for other types of data
class PFDeltaDataset(InMemoryDataset):
    def __init__(
        self,
        root_dir="data",
        case_name="",
        topology_perturb="n",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self.root = root_dir
        self.case_name = case_name
        self.topology_perturb = topology_perturb
        self.add_bus_type = True  # get rid of this later maybe

        super().__init__(
            self.root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        # Load data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(
            self.root, self.case_name, self.topology_perturb, "close2inf_test", "nose"
        )  # how did we do this in main?

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".json")])

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.case_name, "processed")

    @property
    def processed_file_names(self):
        return "nose_cases.pt"

    def process(self):
        data_list = []

        for raw_path in self.raw_paths:
            with open(raw_path, "r") as f:
                pm_case = json.load(f)
            data = self.build_heterodata(pm_case)
            data_list.append(data)

        # Convert list of data objects into (data, slices) for efficient access
        data, slices = self.collate(data_list)  # understand this later

        # Save to disk
        torch.save((data, slices), self.processed_paths[0])

    def build_heterodata(self, pm_case: dict):
        data = HeteroData()
        network_data = pm_case["solved_net"]
        solution_data = pm_case["solved_net"]
        feasibility = (
            True  # get rid of this later. Maybe have all checks happen elsewhere.
        )

        # Bus nodes
        pf_x, pf_y = [], []
        bus_voltages = []
        bus_type = []
        bus_shunts = []
        bus_gen, bus_demand = [], []

        PQ_bus_x, PQ_bus_y = [], []
        PV_bus_x, PV_bus_y = [], []
        PV_demand, PV_generation = [], []
        slack_x, slack_y = [], []
        slack_demand, slack_generation = [], []
        PV_to_bus, PQ_to_bus, slack_to_bus = [], [], []
        pq_idx, pv_idx, slack_idx = 0, 0, 0

        for bus_id_str, bus in sorted(
            network_data["bus"].items(), key=lambda x: int(x[0])
        ):
            bus_id = int(bus_id_str)
            bus_idx = bus_id - 1
            bus_sol = solution_data["bus"][bus_id_str]

            va, vm = bus_sol["va"], bus_sol["vm"]
            bus_voltages.append(torch.tensor([va, vm]))

            # Shunts
            gs, bs = 0.0, 0.0
            for shunt in network_data["shunt"].values():
                if int(shunt["shunt_bus"]) == bus_id:
                    gs += shunt["gs"]
                    bs += shunt["bs"]
            bus_shunts.append(torch.tensor([gs, bs]))

            # Load
            pd, qd = 0.0, 0.0
            for load in network_data["load"].values():
                if int(load["load_bus"]) == bus_id:
                    pd += load["pd"]
                    qd += load["qd"]

            bus_demand.append(torch.tensor([pd, qd]))

            # Gen
            pg, qg = 0.0, 0.0
            for gen_id, gen in sorted(
                network_data["gen"].items(), key=lambda x: int(x[0])
            ):
                if int(gen["gen_bus"]) == bus_id:
                    if gen["gen_status"] == 1:
                        gen_sol = solution_data["gen"][gen_id]
                        pg += gen_sol["pg"]
                        qg += gen_sol["qg"]
                    else:
                        if feasibility:
                            assert gen["pg"] == 0 and gen["qg"] == 0, (
                                f"Expected gen {gen_id} to be off"
                            )
                        else:
                            assert solution_data["gen"].get(gen_id) is None, (
                                f"Expected gen {gen_id} to be off."
                            )

            bus_gen.append(torch.tensor([pg, qg]))

            # Now decide final bus type
            bus_type_now = bus["bus_type"]

            if bus_type_now == 2 and pg == 0.0 and qg == 0.0:
                bus_type_now = 1  # PV bus with no gen --> becomes PQ
                # maybe add an assert here to check if all gens were off.

            bus_type.append(torch.tensor(bus_type_now))

            if bus_type_now == 1:
                pf_x.append(torch.tensor([pd, qd]))
                pf_y.append(torch.tensor([va, vm]))

                PQ_bus_x.append(torch.tensor([pd, qd]))
                PQ_bus_y.append(torch.tensor([va, vm]))
                PQ_to_bus.append(torch.tensor([pq_idx, bus_idx]))
                pq_idx += 1
            elif bus_type_now == 2:
                pf_x.append(torch.tensor([pg - pd, vm]))
                pf_y.append(torch.tensor([qg - qd, va]))

                PV_bus_x.append(torch.tensor([pg - pd, vm]))
                PV_bus_y.append(torch.tensor([qg - qd, va]))
                PV_demand.append(torch.tensor([pd, qd]))
                PV_generation.append(torch.tensor([pg, qg]))
                PV_to_bus.append(torch.tensor([pv_idx, bus_idx]))
                pv_idx += 1
            elif bus_type_now == 3:
                pf_x.append(torch.tensor([va, vm]))
                pf_y.append(torch.tensor([pg - pd, qg - qd]))

                slack_x.append(torch.tensor([va, vm]))
                slack_y.append(torch.tensor([pg - pd, qg - qd]))
                slack_demand.append(torch.tensor([pd, qd]))
                slack_generation.append(torch.tensor([pg, qg]))
                slack_to_bus.append(torch.tensor([slack_idx, bus_idx]))
                slack_idx += 1

        generation, limits, slack_gen = [], [], []

        # Generator nodes
        for gen_id, gen in sorted(network_data["gen"].items(), key=lambda x: int(x[0])):
            if gen["gen_status"] == 1:
                gen_sol = solution_data["gen"][gen_id]
                pmin, pmax, qmin, qmax = (
                    gen["pmin"],
                    gen["pmax"],
                    gen["qmin"],
                    gen["qmax"],
                )
                pgen, qgen = gen_sol["pg"], gen_sol["qg"]
                limits.append(torch.tensor([pmin, pmax, qmin, qmax]))
                generation.append(torch.tensor([pgen, qgen]))
                is_slack = torch.tensor(
                    1
                    if network_data["bus"][str(gen["gen_bus"])]["bus_type"] == 3
                    else 0,
                    dtype=torch.bool,
                )
                slack_gen.append(is_slack)
            else:
                if feasibility:
                    assert (
                        solution_data["gen"][gen_id]["pg"] == 0
                        and solution_data["gen"][gen_id]["qg"] == 0
                    ), f"Expected gen {gen_id} to be off"
                else:
                    assert solution_data["gen"].get(gen_id) is None, (
                        f"Expected gen {gen_id} to be off."
                    )

        # Load nodes
        demand = []
        for load_id, load in sorted(
            network_data["load"].items(), key=lambda x: int(x[0])
        ):
            pd, qd = load["pd"], load["qd"]
            demand.append(torch.tensor([pd, qd]))

        # Edges
        # bus to bus edges
        edge_index, edge_attr, edge_label = [], [], []
        for branch_id_str, branch in sorted(
            network_data["branch"].items(), key=lambda x: int(x[0])
        ):
            if branch["br_status"] == 0:
                continue  # Skip inactive branches

            from_bus = int(branch["f_bus"]) - 1
            to_bus = int(branch["t_bus"]) - 1
            edge_index.append(torch.tensor([from_bus, to_bus]))
            edge_attr.append(
                torch.tensor(
                    [
                        branch["br_r"],
                        branch["br_x"],
                        branch["g_fr"],
                        branch["b_fr"],
                        branch["g_to"],
                        branch["b_to"],
                        branch["tap"],
                        branch["shift"],
                    ]
                )
            )

            branch_sol = solution_data["branch"].get(branch_id_str)
            if feasibility == None:
                assert branch_sol is not None, (
                    f"Missing solution for active branch {branch_id_str}"
                )

            if branch_sol:
                edge_label.append(
                    torch.tensor(
                        [
                            branch_sol["pf"],
                            branch_sol["qf"],
                            branch_sol["pt"],
                            branch_sol["qt"],
                        ]
                    )
                )

        # bus to gen edges
        gen_to_bus_index = []
        for gen_id, gen in sorted(network_data["gen"].items(), key=lambda x: int(x[0])):
            if gen["gen_status"] == 1:
                gen_bus = torch.tensor(gen["gen_bus"]) - 1
                gen_to_bus_index.append(torch.tensor([int(gen_id) - 1, gen_bus]))

        # bus to load edges
        load_to_bus_index = []
        for load_id, load in sorted(
            network_data["load"].items(), key=lambda x: int(x[0])
        ):
            load_bus = torch.tensor(load["load_bus"]) - 1
            load_to_bus_index.append(torch.tensor([int(load_id) - 1, load_bus]))

        # Create graph nodes and edges
        data["bus"].x = torch.stack(pf_x)
        data["bus"].y = torch.stack(pf_y)
        data["bus"].bus_gen = torch.stack(bus_gen)  # aggregated
        data["bus"].bus_demand = torch.stack(bus_demand)  # aggregated
        data["bus"].bus_voltages = torch.stack(bus_voltages)
        data["bus"].bus_type = torch.stack(bus_type)
        data["bus"].shunt = torch.stack(bus_shunts)

        data["gen"].limits = torch.stack(limits)
        data["gen"].generation = torch.stack(generation)
        data["gen"].slack_gen = torch.stack(slack_gen)

        data["load"].demand = torch.stack(demand)

        if self.add_bus_type:
            data["PQ"].x = torch.stack(PQ_bus_x)
            data["PQ"].y = torch.stack(PQ_bus_y)

            data["PV"].x = torch.stack(PV_bus_x)
            data["PV"].y = torch.stack(PV_bus_y)
            data["PV"].generation = torch.stack(PV_generation)
            data["PV"].demand = torch.stack(PV_demand)

            data["slack"].x = torch.stack(slack_x)
            data["slack"].y = torch.stack(slack_y)
            data["slack"].generation = torch.stack(slack_generation)
            data["slack"].demand = torch.stack(slack_demand)

        for link_name, edges in {
            ("bus", "branch", "bus"): edge_index,
            ("gen", "gen_link", "bus"): gen_to_bus_index,
            ("load", "load_link", "bus"): load_to_bus_index,
        }.items():
            edge_tensor = torch.stack(edges, dim=1)
            data[link_name].edge_index = edge_tensor
            if link_name != ("bus", "branch", "bus"):
                data[
                    (link_name[2], link_name[1], link_name[0])
                ].edge_index = edge_tensor.flip(0)
            if link_name == ("bus", "branch", "bus"):
                data[link_name].edge_attr = torch.stack(edge_attr)
                data[link_name].edge_label = torch.stack(edge_label)

        if self.add_bus_type:
            for link_name, edges in {
                ("PV", "PV_link", "bus"): PV_to_bus,
                ("PQ", "PQ_link", "bus"): PQ_to_bus,
                ("slack", "slack_link", "bus"): slack_to_bus,
            }.items():
                edge_tensor = torch.stack(edges, dim=1)
                data[link_name].edge_index = edge_tensor
                data[
                    (link_name[2], link_name[1], link_name[0])
                ].edge_index = edge_tensor.flip(0)

        return data


class PowerBalanceLoss:  # should this even be a class?
    def __init__(self):
        self.power_balance_mean = None
        self.power_balance_max = None
        self.power_balance_l2 = None
        self.loss_name = "PBL Mean"

    def __call__(self, data):
        # Physical quantities from ground truth
        V = data["bus"].bus_voltages[:, 1]  # vm
        theta = data["bus"].bus_voltages[:, 0]  # va
        Pnet = data["bus"].bus_gen[:, 0] - data["bus"].bus_demand[:, 0]
        Qnet = data["bus"].bus_gen[:, 1] - data["bus"].bus_demand[:, 1]

        shunt_g = data["bus"].shunt[:, 0]
        shunt_b = data["bus"].shunt[:, 1]

        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        r = edge_attr[:, 0]
        x = edge_attr[:, 1]
        g_fr = edge_attr[:, 2]
        b_fr = edge_attr[:, 3]
        g_to = edge_attr[:, 4]
        b_to = edge_attr[:, 5]
        tau = edge_attr[:, 6]
        theta_shift = edge_attr[:, 7]

        src, dst = edge_index

        Y = 1 / (r + 1j * x)
        Y_real = torch.real(Y)
        Y_imag = torch.imag(Y)

        delta_theta1 = theta[src] - theta[dst]
        delta_theta2 = theta[dst] - theta[src]

        # Active power flow
        P_flow_src = (
            V[src]
            * V[dst]
            / tau
            * (
                -Y_real * torch.cos(delta_theta1 - theta_shift)
                - Y_imag * torch.sin(delta_theta1 - theta_shift)
            )
            + Y_real * (V[src] / tau) ** 2
        )

        P_flow_dst = (
            V[dst]
            * V[src]
            / tau
            * (
                -Y_real * torch.cos(delta_theta2 - theta_shift)
                - Y_imag * torch.sin(delta_theta2 - theta_shift)
            )
            + Y_real * V[dst] ** 2
        )

        # Reactive power flow
        Q_flow_src = (
            V[src]
            * V[dst]
            / tau
            * (
                -Y_real * torch.sin(delta_theta1 - theta_shift)
                + Y_imag * torch.cos(delta_theta1 - theta_shift)
            )
            - (Y_imag + b_fr) * (V[src] / tau) ** 2
        )

        Q_flow_dst = (
            V[dst]
            * V[src]
            / tau
            * (
                -Y_real * torch.sin(delta_theta2 - theta_shift)
                + Y_imag * torch.cos(delta_theta2 - theta_shift)
            )
            - (Y_imag + b_to) * V[dst] ** 2
        )

        # Aggregate to buses
        Pbus_pred = torch.zeros_like(V).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)

        Qbus_pred = torch.zeros_like(V).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        # Power mismatch
        delta_P = Pnet - Pbus_pred - V**2 * shunt_g
        delta_Q = Qnet - Qbus_pred + V**2 * shunt_b
        delta_PQ_2 = delta_P**2 + delta_Q**2

        delta_PQ_magnitude = torch.sqrt(delta_PQ_2)

        return delta_PQ_magnitude, delta_P, delta_Q
