import types

import torch
from torch import nn
from torch_geometric.data import HeteroData

from core.utils.registry import registry
from core.utils.pf_losses_utils import PowerBalanceLoss


def loss_loader(class_name, class_inputs, class_type):
    loss_class = registry.get_loss_class(class_name)
    assert loss_class is not None, \
        f"{class_type} {class_name} is not registered in the registry!"
    if isinstance(loss_class, types.FunctionType):
        loss = loss_class
    else:
        loss = loss_class(**class_inputs)

    return loss


@registry.register_loss("GNNTorchLoss")
class GNNTorchLoss:
    def __init__(self, torch_nn_name, output_name, loss_inputs={}):
        loss_class = getattr(torch.nn, torch_nn_name, None)
        assert loss_class is not None, f"Loss {torch_nn_name} not found in torch.nn!"
        assert isinstance(loss_inputs, dict), f"Loss inputs need to be a dictionary!"
        self.loss = loss_class(**loss_inputs)
        self.output_name = output_name
        self.loss_name = torch_nn_name


    def __call__(self, outputs, data):
        if "__" in self.output_name:
            key, output_name = self.output_name.split("__")
            truth = data[key].get(output_name, None)
        else:
            truth = data.get(self.output_name, None)
        assert truth is not None, f"Data does not contain {output_name}!"
        return self.loss(outputs, truth)


@registry.register_loss("PBL_mean")
def PBL_mean(predictions, data):
    """
    Custom loss function for the PowerFlowTypedGNN to enforce power
    balance at each node. The loss minimizes the mismatch in active power
    (ΔP) and reactive power (ΔQ) at each node.

    The power balance equations are:
        ΔP(t) = Pg - Pd - Pbus(V(t), θ(t))
        ΔQ(t) = Qg - Qd - Qbus(V(t), θ(t))

    The goal is to minimize the sum of squared ΔP and ΔQ across all nodes.
    """
    per_node_loss = PowerBalanceLoss(predictions, data)[0]
    return per_node_loss.mean()


@registry.register_loss("PBL_l2norm")
def PBL_l2norm(predictions, data):
    """
    Same as PBL_mean, but this time we calculate the L2 norm
    """
    batch = data["bus"].batch
    batch_size = batch.max().item() + 1
    per_node_loss = PowerBalanceLoss(predictions, data)[0]
    l2_normed = torch.zeros(batch_size, device=predictions.device)
    l2_normed = l2_normed.scatter_add_(dim=0, index=batch, src=per_node_loss**2)
    l2_normed = torch.sqrt(l2_normed)
    return l2_normed.mean()


@registry.register_loss("PBL_max")
def PBL_max(predictions, data):
    """
    Returns the maximum power balance loss
    """
    per_node_loss = PowerBalanceLoss(predictions, data)[0]
    max_loss = per_node_loss.max()
    return max_loss


@registry.register_loss("combined_loss")
class CombinedLoss:
    def __init__(self, loss1, loss2, lamb=1, inp1={}, inp2={}):
        self.loss1_name = loss1
        self.loss2_name = loss2
        self.lamb = lamb

        self.loss1 = self.initialize_loss(loss1, inp1)
        self.loss2 = self.initialize_loss(loss2, inp2)

        loss1_printname = getattr(self.loss1, "loss_name", loss1)
        loss2_printname = getattr(self.loss2, "loss_name", loss2)
        self.loss_name = loss1_printname + "+" + loss2_printname

    def initialize_loss(self, loss_name, loss_inputs):
        # This is for pytorch losses
        if getattr(nn, loss_name, None) is not None:
            loss_class = getattr(nn, loss_name)
            return loss_class(**inputs)

        # This is for custom loss
        loss_class = registry.get_loss_class(loss_name)
        assert loss_class is not None, f"Loss {loss_name} not found!"

        if isinstance(loss_class, types.FunctionType):
            assert len(loss_inputs) == 0, \
                f"Custom loss {loss_name} is a function, but loss inputs were received!"
            return loss_class
        else:
            return loss_class(**loss_inputs)


    def __call__(self, predictions, labels):
        loss1 = self.loss1(predictions, labels)
        loss2 = self.loss2(predictions, labels)
        weighted_loss = loss1 + self.lamb*loss2
        return weighted_loss


@registry.register_loss("Objective_n_Penalty")
class Objective_n_Penalty:
    def __init__(
        self, obj_name=None, ineq_name=None, eq_name=None,
        obj_inputs={}, ineq_inputs={}, eq_inputs={}
    ):
        self.obj_name = obj_name
        self.ineq_name = ineq_name
        self.eq_name = eq_name
        self.obj_fn = None
        self.ineq_fn = None
        self.eq_fn = None

        obj_active = obj_name is not None
        ineq_active = ineq_name is not None
        eq_active = eq_name is not None
        assert obj_active or ineq_active or eq_active, \
            "No objective, equality or inequality declared!!"

        # Load objective function
        if self.obj_name is not None:
            self.obj_fn = loss_loader(obj_name, obj_inputs, "Objective function")

        # Load inequality functions
        if self.ineq_name is not None:
            self.ineq_fn = loss_loader(ineq_name, ineq_inputs, "Inequality functions")

        # Load equality functions
        if self.eq_name is not None:
            self.eq_fn = loss_loader(eq_name, eq_inputs, "Equality functions")

        self.create_name()

    def create_name(self,):
        name = ""
        if self.obj_name is not None:
            obj_name = getattr(self.obj_fn, "loss_name", "Obj")
            name += obj_name
        if self.ineq_name is not None:
            ineq_name = getattr(self.ineq_fn, "loss_name", "Ineq")
            if len(name) == 0:
                name += ineq_name
            else:
                name += "+" + ineq_name
        if self.eq_name is not None:
            eq_name = getattr(self.eq_fn, "loss_name", "Eq")
            if len(name) == 0:
                name += eq_name
            else:
                name += "+" + eq_name
        self.loss_name = name

    def __call__(self, predictions, data):
        loss = 0
        if self.obj_fn is not None:
            loss += self.obj_fn(predictions, data)
        if self.ineq_fn is not None:
            loss += self.ineq_fn(predictions, data)
        if self.eq_fn is not None:
            loss += self.eq_fn(predictions, data)
        return loss.mean()


@registry.register_loss("pfn_masked_mse")
class Masked_L2_loss:
    """
    Custom loss function for the masked L2 loss.

    Args:
        output (torch.Tensor): The output of the neural network model.
        target (torch.Tensor): The target values.
        mask (torch.Tensor): The mask for the target values.

    Returns:
        torch.Tensor: The masked L2 loss.
    """

    def __init__(self, regularize=True, regcoeff=1):
        super(Masked_L2_loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.regularize = regularize
        self.regcoeff = regcoeff
        self.loss_name = "Masked MSE"
        if self.regularize:
            self.loss_name += ", reg."


    def __call__(self, output, data):
        if isinstance(data, HeteroData):
            target = data["bus"].y
            mask = data["bus"].x[:, 10:]
        else:
            target = data.y
            mask = data.x[:, 10:]

        masked = mask.type(torch.bool)

        # output = output * mask
        # target = target * mask
        outputl = torch.masked_select(output, masked)
        targetl = torch.masked_select(target, masked)

        loss = self.criterion(outputl, targetl)

        if self.regularize:
            masked = (1 - mask).type(torch.bool)
            output_reg = torch.masked_select(output, masked)
            target_reg = torch.masked_select(target, masked)
            loss = loss + self.regcoeff * self.criterion(output_reg, target_reg)

        return loss
