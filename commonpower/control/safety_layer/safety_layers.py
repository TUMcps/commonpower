"""
Collection of safety layers.
"""
import logging
from copy import deepcopy
from typing import Dict, List, Tuple

from pyomo.core import ConcreteModel, Objective, quicksum
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import value
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver

from commonpower.control.safety_layer.penalties import BasePenalty, DistanceDependingPenalty
from commonpower.control.util import clone_from_top_level_nodes
from commonpower.modeling.base import ModelEntity
from commonpower.utils.cp_exceptions import EntityError
from commonpower.utils.default_solver import get_default_solver

logging.getLogger().setLevel(logging.ERROR)


class BaseSafetyLayer:
    def __init__(self):
        """
        Base class for safety layers. A safety layer checks whether the action selected by a controller violates any
        constraints of the controlled entities and adjusts the actions if necessary.

        Returns:
            BaseSafetyLayer

        """
        self.nodes = None
        self.top_level_nodes = None
        self.obj_fcn = None
        self.unsafe_action = None

    def initialize(self, nodes: List[ModelEntity], top_level_nodes: List[ModelEntity]):
        """
        Initializes the safety layer
        Args:
            nodes (List[ModelEntity]):
                list of controlled entities to be safeguarded
            top_level_nodes (List[ModelEntity]):
                list of controlled entities in highest level of model tree
            solver (OptSolver):
                solver for optimization problem which will be called by Pyomo
        Returns:
            None

        """
        self.nodes = nodes
        self.top_level_nodes = top_level_nodes

    def compute_safe_action(self, action: Dict = None) -> Tuple[Dict, bool, float]:
        """
        Checks whether the actions proposed by the controller satisfy the constraints
        of the controlled entities and modifies them if necessary.

        Args:
            action (dict): action suggested by the controller

        Returns:
            safe_action (dict): verified action
            action_corrected (bool): whether the action was corrected or not
            correction_penalty (float): penalty for action correction (0 if action was not corrected)

        """
        raise NotImplementedError("Safety layers need to implement this method, do not use BaseClass directly.")


class ActionReplacementWithOptSafetyLayer(BaseSafetyLayer):
    # distance used to determine if the action was corrected or not
    DISTANCE_EPS = 1e-5

    def __init__(self, penalty: BasePenalty, solver: OptSolver = None):
        """
        Action replacement safety layer. Action violating the constraints is replaced
        by safe action determined through an optimization method.

        Args:
            penalty (BasePenalty): class defining the penalty behavior for unsafe actions
            solver (OptSolver, optional):
                solver for optimization problem, defaults to direct gurobi
                https://pyomo.readthedocs.io/en/stable/library_reference/solvers/gurobi_direct.html

        Returns:
            ActionReplacementWithOptSafetyLayer

        """
        super().__init__()
        if solver is None:
            solver = get_default_solver()
        self.penalty = penalty
        self.solver = solver

    def __del__(self):
        """Called when the class is destroyed, takes care to release Gurobi resources"""
        self.solver.close()

    def compute_safe_action(self, action: Dict = None) -> Tuple[Dict, bool, float]:
        """
        Checks whether the actions proposed by the controller satisfy the constraints of the controlled entities and
        replaces by pyomo-generated output if necessary.

        Args:
            action (dict): action suggested by the controller

        Returns:
            safe_action (dict): verified action
            action_corrected (bool): whether the action was corrected or not
            correction_penalty (float): penalty for action correction (0 if action was not corrected)

        """
        # store action
        self.unsafe_action = action

        action_feasible = self.is_action_feasible(action)

        if action_feasible:
            return action, False, 0.0

        model = self.prepare_model()
        model = self.set_action_in_model(model, action)  # initializes optimization with current action
        action_distance = self.solve_model(model)
        if action_distance is None:
            raise EntityError(self.top_level_nodes[0], "Cannot find a safe input")
        # second check is needed for the action projection safety layer as it
        # avoids having to do the costly feasibility check
        action_corrected = action_distance > self.DISTANCE_EPS

        safe_action = self.set_action_from_model(model, action)
        if not action_corrected:
            return safe_action, action_corrected, 0.0
        correction_penalty = self.get_penalty(action_distance)

        return safe_action, action_corrected, correction_penalty

    def get_penalty(self, action_distance: float) -> float:
        """
        Get penalty depending on the penalty class used.

        Args:
            action_distance (float): distance between the safe and unsafe action,
                                     only used if penalty is distance based

        Returns:
            float: computed penalty for the action
        """
        # correction penalty (can be used in RL reward function) -> depends on used penalty type
        # If the penalty is distance depending, the penalty is computed based on the distance between the unsafe action
        # and the safe action. Therefore, the penalty needs the value of the safety objective function.
        if isinstance(self.penalty, DistanceDependingPenalty):
            return self.penalty.get_correction_penalty(action_distance)
        else:
            return self.penalty.get_correction_penalty()

    def is_action_feasible(self, action: dict) -> dict:
        """Check if action is feasible.

        Args:
            action (dict): action to check

        Returns:
            bool: True if action is feasible
        """
        model = self.prepare_model()
        self.set_action_in_model(model, action, fix_values=True)
        res = self.solve_model(model)
        return res is not None

    def prepare_model(self) -> ConcreteModel:
        """
        Clone model from sys to run local optimization over

        Returns:
            ConcreteModel: pyomo optimization model, representing
                           the part of the system under supervision of the safety
        """
        # get current system pyomo instance
        sys_inst = self.nodes[0].instance

        mdl = clone_from_top_level_nodes(self.top_level_nodes, sys_inst)

        return mdl

    def solve_model(self, model: ConcreteModel) -> float:
        """
        Finds a feasible action by solving a local model.
        Returns the distance of the original unsafe action to the solved model.

        Args:
            model (ConcreteModel): model to solve
        Returns:
            float: distance from action or None
        """
        results = self.solver.solve(model, warmstart=True)
        if results.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            with open("infeasible_safety_model.log", "w") as f:
                model.pprint(f)
            return None

        distance_from_action = value(self.action_distance(model, self.unsafe_action))

        return distance_from_action

    def action_distance(self, model: ConcreteModel, action: dict) -> SumExpression:
        """
        Euclidean norm between action expressed by the model and the action in dict

        Args:
            model (ConcreteModel): pyomo optimization model
            action (dict): action description
        Returns:
            SumExpression: distance between the action and the model values expressed
                           in pyomo class, convert to float by running value() over it

        """
        obj_fcn_elements = []
        for node in self.nodes:
            node_input_ids = node.get_input_ids(model)
            if node_input_ids is not None:
                # separate the input element name and the node id
                el_names = [n_id.split(".")[-1] for n_id in node_input_ids]
                global_node_ids = [".".join(n_id.split(".")[:-1]) for n_id in node_input_ids]
                # obtain action horizon (for how many time steps does the RL agent predict the action)
                action_horizon = list(range(len(action[global_node_ids[0]][el_names[0]])))
                # action projection objective function: (a_RL[t] - a_safe[t])^2 for all t in action_horizon
                # first step: (a_RL[t] - a_safe[t]) for all t and for all input elements of the current node
                node_fcn = [
                    action[global_node_ids[i]][el_names[i]][t] - node.get_pyomo_element(el_names[i], model)[t]
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

    def set_action_from_model(self, model: ConcreteModel, action: dict) -> dict:
        """Corrects an action according to a model.

        Args:
            model (ConcreteModel): pyomo model to ge the values from
            action (dict): action to correct

        Returns:
            dict: safe action according to the model
        """
        safe_action = deepcopy(action)
        node_actions = {}
        for node in self.nodes:
            node_action = node.get_inputs(model)
            if node_action is not None:
                node_actions[node.id] = node_action

        for node_id, actions in safe_action.items():
            for el_id, el_action in actions.items():
                for i in range(el_action.shape[0]):
                    safe_action[node_id][el_id][i] = node_actions[node_id][el_id][i]
        return safe_action

    def set_action_in_model(self, model: ConcreteModel, action: dict, fix_values: bool = False) -> ConcreteModel:
        """
        Sets the model values to the action to initialize the optimization.
        If fix values is True, the model values will be fixed, which is useful
        to check the feasibility of the given action.

        Args:
            model (ConcreteModel): pyomo optimization model
            action (dict): action to set into the model
            fix_values (bool): optionally fixes the values for the optimization problem

        Returns:
            ConcreteModel: model with values set
        """
        for node in self.nodes:
            node_input_ids = node.get_input_ids(model)
            if node_input_ids is not None:
                # separate the input element name and the node id
                el_names = [n_id.split(".")[-1] for n_id in node_input_ids]
                global_node_ids = [".".join(n_id.split(".")[:-1]) for n_id in node_input_ids]
                for i in range(len(el_names)):
                    node.set_value(model, el_names[i], action[global_node_ids[i]][el_names[i]], fix_value=fix_values)
        return model


class ActionProjectionSafetyLayer(ActionReplacementWithOptSafetyLayer):
    def __init__(self, penalty: BasePenalty, solver: OptSolver = None):
        """
        Computes safe action by minimizing the
        distance between the RL action and the safe
        action while also satisfying
        constraints.

        Args:
            penalty (BasePenalty): class defining the penalty behavior for unsafe actions
            solver (OptSolver, optional):
                solver for optimization problem, defaults to direct gurobi
                https://pyomo.readthedocs.io/en/stable/library_reference/solvers/gurobi_direct.html

        Returns:
            ActionReplacementSafetyLayer

        """
        super().__init__(penalty, solver)

    def prepare_model(self) -> ConcreteModel:
        """Prepare model. Additionally projection criterion is added

        Returns:
            ConcreteModel: pyomo optimization model, representing
                           the part of the system under supervision of the safety
        """
        model = super().prepare_model()

        model.safety_obj = Objective(expr=lambda model: self.action_distance(model, self.unsafe_action))
        return model

    def is_action_feasible(self, action) -> bool:
        """
        As feasibility check is quite costly, we can skip for projection layer
        as we check action distance to the original action after optimization
        """
        return False
