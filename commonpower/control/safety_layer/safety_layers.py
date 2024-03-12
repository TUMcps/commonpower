"""
Collection of safety layers.
"""
from copy import deepcopy
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.core import ConcreteModel, Objective, quicksum
from pyomo.environ import value
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver

from commonpower.core import System
from commonpower.modelling import ModelEntity
from commonpower.utils.cp_exceptions import EntityError
from commonpower.utils.default_solver import get_default_solver


class BaseSafetyLayer:
    def __init__(self, solver: OptSolver = get_default_solver()):
        """
        Base class for safety layers. A safety layer checks whether the action selected by a controller violates any
        constraints of the controlled entities and adjusts the actions if necessary.

        Args:
            solver (OptSolver, optional): solver for optimization problem

        Returns:
            BaseSafetyLayer

        """
        self.nodes = None
        self.top_level_nodes = None
        self.obj_fcn = None
        self.sys_inst = None
        self.model = None
        self.unsafe_action = None
        self.solver = solver

    def initialize(self, nodes: List[ModelEntity], top_level_nodes: List[ModelEntity]):
        """
        Initializes the safety layer

        Args:
            nodes (List[ModelEntity]): list of controlled entities to be safeguarded
            top_level_nodes (List[ModelEntity]): list of controlled entities in highest level of model tree
            solver (OptSolver): solver for optimization problem which will be called by Pyomo

        Returns:
            None

        """
        self.nodes = nodes
        self.top_level_nodes = top_level_nodes

    def compute_safe_action(self, action: Dict = None) -> Tuple[Dict, bool, float]:
        """
        Checks whether the actions proposed by the controller satisfy the constraints of the controlled entities and
        modifies them if necessary

        Args:
            action (dict): action suggested by the controller

        Returns:
            dict: verified action
            bool: whether the action was corrected or not
            float: penalty for action correction (0 if action was not corrected)

        """
        # ToDo: rewrite this to make it more general?
        # store action
        self.unsafe_action = action
        # get current system pyomo instance
        self.sys_inst = self.nodes[0].instance
        mdl = ConcreteModel()

        for node in self.top_level_nodes:
            if isinstance(node, System):
                mdl = self.sys_inst.clone()
            else:
                setattr(mdl, node.id.split(".")[-1], node.get_self_as_pyomo_block(self.sys_inst).clone())

        # we want to delete existing objectives from the original system and define our own
        for objective in mdl.component_objects(pyo.Objective, descend_into=True):
            mdl.del_component(objective)

        self.model = mdl

        # set objective function
        self.model.safety_obj = Objective(expr=self.get_objective_function())

        results = self.solver.solve(self.model, warmstart=True)
        self.model.solutions.store_to(results)
        # catch infeasible solution
        if results.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            raise EntityError(self.top_level_nodes[0], "Cannot find a safe input")

        # retrieve inputs from solved optimization problem
        safe_action = deepcopy(action)  # copy or deepcopy?
        node_actions = {}
        for node in self.nodes:
            node_action = node.get_inputs(self.model)
            if node_action is not None:
                node_actions[node.id] = node_action

        for node_id, actions in safe_action.items():
            for el_id, el_action in actions.items():
                for i in range(el_action.shape[0]):
                    safe_action[node_id][el_id][i] = node_actions[node_id][el_id][i]

        # correction penalty (can be used in RL reward function)
        correction_penalty = self.get_correction_penalty()
        action_corrected = value(self.model.safety_obj) > 1e-5
        return safe_action, action_corrected, correction_penalty

    def get_objective_function(self):
        # needs to be implemented by subclasses
        raise NotImplementedError

    def get_correction_penalty(self):
        # needs to be implemented by subclasses
        raise NotImplementedError


class ActionProjectionSafetyLayer(BaseSafetyLayer):
    def __init__(self, penalty_factor: float = 1.0, solver: OptSolver = get_default_solver()):
        """
        Computes safe action by minimizing the distance between the RL action and the safe action while also satisfying
        constraints.

        Args:
            penalty_factor (float): factor to control magnitude of penalty
            solver (OptSolver, optional): solver for optimization problem
        """
        super().__init__(solver=solver)
        self.penalty_factor = penalty_factor

    def get_objective_function(self):
        """
        Objective function: minimal distance between action suggested by RL controller and safe action.

        Returns:
            Callable: objective function rule for Pyomo model

        """

        def obj_fcn_rl(model):
            obj_fcn_elements = []
            for node in self.nodes:
                node_input_ids = node.get_input_ids(self.model)
                if node_input_ids is not None:
                    # separate the input element name and the node id
                    el_names = [n_id.split(".")[-1] for n_id in node_input_ids]
                    global_node_ids = [".".join(n_id.split(".")[:-1]) for n_id in node_input_ids]
                    # obtain action horizon (for how many time steps does the RL agent predict the action)
                    action_horizon = list(range(len(self.unsafe_action[global_node_ids[0]][el_names[0]])))
                    # action projection objective function: (a_RL[t] - a_safe[t])^2 for all t in action_horizon
                    # first step: (a_RL[t] - a_safe[t]) for all t and for all input elements of the current node
                    node_fcn = [
                        self.unsafe_action[global_node_ids[i]][el_names[i]][t]
                        - node.get_pyomo_element(el_names[i], self.model)[t]
                        for t in action_horizon
                        for i in range(len(el_names))
                    ]
                    # second step: ()^2
                    node_fcn = [item**2 for item in node_fcn]

                    obj_fcn_elements.append(node_fcn)
            # flatten list
            obj_fcn_elements = [item for sublist in obj_fcn_elements for item in sublist]
            # sum over all time steps and all input elements of all nodes
            obj = quicksum(obj_fcn_elements)
            return obj

        return obj_fcn_rl

    def get_correction_penalty(self):
        """
        In this case, the penalty for the action correction is the objective resulting from solving the
        minimal-adjustment constrained optimization problem.

        Returns:
            float: penalty

        """
        return self.penalty_factor * value(self.model.safety_obj)
