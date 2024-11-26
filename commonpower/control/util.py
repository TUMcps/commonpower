"""
Helper functions for control module.
"""


from pyomo.core import ConcreteModel, Objective

from commonpower.core import System
from commonpower.modeling.base import ModelEntity


def t2n(x):
    """
    Transform a torch tensor to a numpy array.

    Args:
        x: torch tensor

    Returns:
        (np.array): numpy array

    """
    return x.detach().cpu().numpy()


def predicted_cost_callback(ctrl, sys_inst) -> float:
    """
    Computes the cost for a controller as the actual cost for the current time step and the predicted cost over the rest
    of the forecast horizon. The "prediction" is actually the solution to the optimization problem solved when updating
    the sys_inst, so it basically assumes an optimal controller for the future time steps. This is not really in line
    with the RL paradigm, so use with care!

    Args:
        ctrl (BaseController): controller for which we will compute the cost
        sys_inst (System): the system within which the controller operates some subsystem

    Returns:
        float: cost of controlling the subsystem
    """
    # ToDo: Need to adjust if we ever have an action horizon > 1 time step
    cost_values = [n.get_value(sys_inst, "cost") for n in ctrl.top_level_nodes]
    cost_values = [item for sublist in cost_values for item in sublist]
    ctrl_cost = sum(cost_values)
    return ctrl_cost


def single_step_cost_callback(ctrl, sys_inst) -> float:
    """
    Computes the cost for a controller as the actual cost for the current time step.

    Args:
        ctrl (BaseController): controller for which we will compute the cost
        sys_inst (System): the system within which the controller operates some subsystem

    Returns:
        float: cost of controlling the subsystem
    """
    # ToDo: Need to adjust if we ever have an action horizon > 1 time step
    cost_values = [n.get_value(sys_inst, "cost") for n in ctrl.top_level_nodes]
    cost_values = [cost[0] for cost in cost_values]
    ctrl_cost = sum(cost_values)
    return ctrl_cost


def clone_from_top_level_nodes(
    nodes: list[ModelEntity], model_instance: ConcreteModel, clone_objectives: bool = False
) -> ConcreteModel:
    """
    Generates a new model instance from the given top-level nodes while also cloning their children.
    """
    mdl = ConcreteModel()

    for node in nodes:
        if isinstance(node, System):
            mdl = model_instance.clone()
        else:
            setattr(mdl, node.id.split(".")[-1], node.get_self_as_pyomo_block(model_instance).clone())

    if not clone_objectives:
        for objective in mdl.component_objects(Objective, descend_into=True):
            mdl.del_component(objective)

    return mdl
