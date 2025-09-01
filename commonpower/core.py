"""
Core power system entities.
"""
from __future__ import annotations

import pickle
import random
import re
from collections import OrderedDict
from copy import copy
from datetime import datetime, timedelta
from typing import Callable, List, Tuple, Union

import gymnasium as gym
import numpy as np
from pyomo.core import ConcreteModel, Expression, Objective, Set, quicksum, value
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver

from commonpower.control.environments import ControlEnv
from commonpower.control.observation_handling import Observer
from commonpower.data_forecasting import DataProvider
from commonpower.modeling.base import ControllableModelEntity, ElementTypes, ModelElement, ModelEntity
from commonpower.modeling.history import ModelHistory
from commonpower.modeling.param_initialization import ParamInitializer
from commonpower.modeling.robust_constraints import RobustConstraintBuilder
from commonpower.modeling.robust_cost import BaseRobustCost, CostScenario, NominalCost
from commonpower.utils import rsetattr
from commonpower.utils.cp_exceptions import EntityError, InstanceError
from commonpower.utils.default_solver import get_default_solver


class PowerFlowModel:
    """
    Generic class to model power flow constraints.
    """

    def empty_copy(self) -> PowerFlowModel:
        """
        Creates a fresh copy of the power flow model.

        Returns:
            PowerFlowModel: Cloned power flow model instance.
        """
        return self.__class__()

    def add_to_model(self, model: ConcreteModel, nodes: List[Node], lines: List[Line]) -> None:
        """
        Specifies the power flow constraints and adds them to the given model instance.
        This method is called by system.add_to_model().

        Args:
            model (ConcreteModel): Pyomo root model (sys).
            nodes (List[Node]): Nodes to consider.
            lines (List[Line]): Lines to consider.
        """

        self._set_sys_constraints(model, nodes, lines)

        for nid, node in enumerate(nodes):
            connected_lines = [line for line in lines if node in [line.src, line.dst]]
            self._set_bus_constraint(model, nid, node, connected_lines)

        for lid, line in enumerate(lines):
            self._set_line_constraint(model, lid, line)

    def _set_sys_constraints(self, model: ConcreteModel, nodes: List[Node], lines: List[Line]) -> None:
        """
        Adds system-wide constraint(s) to the given model instance.
        Optional for subclasses.

        Args:
            model (ConcreteModel): Pyomo root model (sys).
            nodes (List[Node]): Nodes to consider.
            lines (List[Line]): Lines to consider.
        """

    def _set_bus_constraint(self, model: ConcreteModel, nid: int, node: Node, connected_lines: list[Line]) -> None:
        """
        Adds constraint for the given bus to the given model instance.
        Optional for subclasses.

        Args:
            model (ConcreteModel): Pyomo root model (sys).
            nid (int): Node index.
            node (Node): Node instance.
            connected_lines (list[Line]): List of lines connected to the node.
        """

    def _set_line_constraint(self, model: ConcreteModel, lid: int, line: Line):
        """
        Adds constraint for the given bus to the given model instance.
        Optional for subclasses.

        Args:
            model (ConcreteModel): Pyomo root model (sys).
            lid (int): Line index.
            line (Line): Line instance.
        """


class System(ControllableModelEntity):
    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        return [ModelElement("cost", ElementTypes.COST, "dispatch cost")]

    def __init__(self, power_flow_model: PowerFlowModel) -> System:
        """
        Singleton class to serve as root of the model hierarchy.
        The System manages all nodes/lines and provides the interfaces to simulate/control them.

        Args:
            power_flow_model (PowerFlowModel): Power flow model to include in the system's constraints.
        """

        super().__init__("System")

        self.nodes = []
        self.lines = []
        self.controllers = {}

        self.t = None  # current time
        self.tau = None  # time step
        self.horizon = None  # control horizon
        self.horizon_int = None

        self.start_time = None  # start of simulation time

        self.date_range = None  # date range of data

        self.power_flow_model = power_flow_model

        self.env_func = None

        self.solver = None  # solver for optimization problem

        self._cost_builder: BaseRobustCost = None

        self.observer = Observer()

    def empty_copy(self, with_entities: bool = True) -> System:
        """
        Creates a fresh copy of the system.

        Returns:
            System: Cloned system instance.
        """
        new_sys = System(self.power_flow_model.empty_copy())

        if with_entities:
            for node in self.nodes:
                new_sys.add_node(node.empty_copy(with_children=True))
            for line in self.lines:
                new_sys.add_line(line.empty_copy())

        return new_sys

    def add_node(self, node: Node, at_index: Union[None, int] = None) -> System:
        """
        Adds a node to the system.
        Here, the node's id is set according to its position in the model hierarchy.

        Args:
            node (Node): Node istance to add.
            at_index (int, optional): Specifies to override the existing node at this index in the system's node list.
        Returns:
            System: System instance.
        """
        if at_index is None:
            self.nodes.append(node)
        else:
            self.nodes[at_index] = node
        return self

    def add_line(self, line: Line) -> System:
        """
        Adds a line to the system.

        Args:
            line (Line): Line instance to add.

        Returns:
            System: System instance.
        """
        self.lines.append(line)
        return self

    def initialize(
        self,
        episode_horizon: timedelta = timedelta(hours=0),
        horizon: timedelta = timedelta(hours=24),
        tau: timedelta = timedelta(hours=1),
        solver: OptSolver = get_default_solver(),
    ) -> None:
        """
        Initializes the system.
        This constructs the pyomo model of the system by traversing through the
        object tree (nodes, lines, power flow), calling self.add_to_model().
        It validates if required data providers are present and
        if controllers have been defined appropriately.

        Args:
            episode_horizon (timedelta): Specifies how long RL agents simulate the system before resetting.
                Mainly needed to adjust the date_range of the system.
            horizon (timedelta, optional): This specifies the time period for which
                the controllers "look into the future". Defaults to 24h.
            tau (timedelta, optional): Sample time, i.e., the period of time between to control actions.
                This needs to match the frequency of data providers. Defaults to timedelta(hours=1).
            solver (OptSolver, optional): Solver instance for the optimization problem that will be called by Pyomo.
        """
        self.tau = tau
        self.horizon = horizon
        self.horizon_int = int(self.horizon / self.tau)
        self.episode_horizon = episode_horizon
        self.solver = solver

        # check if all data providers have appropriate forecast horizon and data frequency
        for node in self.nodes:
            node.validate_data_providers(horizon, tau)

        self.add_to_model(ConcreteModel())

        # check if all nodes with input elements have a controller assigned
        # add unique control to controller dictionary
        ctrl_ids = []
        ctrl = OrderedDict()

        for node in self.nodes:
            ctrl_ids, ctrl = node.validate_controller(ctrl_ids, ctrl)
        # change order such that the system controller will be the last one
        sys_controller_id = self.controller.get_id()
        if sys_controller_id in ctrl:
            ctrl.move_to_end(sys_controller_id, last=True)
        self.controllers = ctrl
        # initialize control
        for ctrl in self.controllers.values():
            ctrl.initialize()

        self.date_range = self._calc_date_range()

    def reset(self, at_time: datetime) -> None:
        """
        Resets the system model to the state at a certain timestamp.
        It creates a clone of the system's "raw" pyomo model which will then be used for simulation.
        Furthermore, entities' parameters are initialized according to their configuration.

        Args:
            at_time (datetime): Timestamp to begin the simulation from.
        """
        self.t = at_time
        self.start_time = self.t
        self.instance = self.model.clone()

        for node in self.nodes:
            node.t = self.t
            node.reset(self.instance, at_time)

        # reset training history of RL controllers
        for ctrl in self.controllers.values():
            from commonpower.control.controllers import RLBaseController

            ctrl.reset_history()
            if isinstance(ctrl, RLBaseController):
                ctrl.obs_handler.reset()

    def unmodeled_update(self):
        """
        Executes the unmodeled updates of all system nodes.
        """

        for node in self.nodes:
            node.unmodeled_update()

    def update_data(self, at_time: datetime):
        """
        Updates the data sources of the system for the given timestamp.

        Args:
            at_time (datetime): Timestamp of "now".
        """
        for node in self.nodes:
            node.update_data(at_time)

    def update(self):
        """
        Moves the system one time step forward.
        Loads new values from data providers and updates state variables.
        """
        self.t = self.t + self.tau

        for node in self.nodes:
            node.t = self.t
            node.update(self.t)

    def save_to_file(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_from_file(self, path: str) -> System:
        with open(path, "rb") as f:
            return pickle.load(f)

    def add_to_model(self, model: ConcreteModel) -> None:
        """
        This method adds the system to the global pyomo model.
        Accordingly, it adds all specified nodes, lines and power flow definitions.

        Args:
            model (ConcreteModel): Root (global) pyomo model.
        """
        self.model = model  # store reference to model internally

        self.model.t = Set(initialize=range(0, self.horizon_int + 1))

        # the tau value in the model is a float indicating tau / 1h, i.e., the fraction/multiple of one hour.
        tau_float = self.tau / timedelta(hours=1)

        # self.model.tau = Param(initialize=tau_float)
        # self.model.horizon = Param(initialize=self.horizon, domain=pyo.NonNegativeIntegers)

        for idx, node in enumerate(self.nodes):
            node.set_id("", idx)
            node.add_to_model(model, tau=tau_float, horizon=self.horizon_int)

        for line in self.lines:
            line.add_to_model(model)

        # for power flow we extract all child nodes from StructureNodes
        self.power_flow_model.add_to_model(model, self.get_first_level_non_structure_nodes(), self.lines)

        self.model_elements = self._augment_model_elements(self._get_model_elements())

        for el in self.model_elements:
            self._add_model_element(el)

        # this cost function does not really matter, since all step 0 inputs are computed by controllers
        # however, using a reasonable cost function makes this more robust against modeling errors,
        # helps with warmstarting the solver,
        # and allows for better analysis of predicted behavior at different time steps.
        self._cost_builder = NominalCost(
            discount_factor=1.0,
        )
        self._cost_builder.initialize(self.cost_fcn, self.horizon_int)
        self.model.obj1 = Objective(expr=self._cost_builder.obj_fcn(model))

    def get_controllers(self, ctrl_types: list = None) -> dict:
        """
        Get dictionary of {controller_id : controller} based on the type of controllers (returns all controllers if no
        types are specified).

        Args:
            ctrl_types (list): list of controller types to be included (if None, all controllers are included).

        Returns:
            dict: dictionary of {controller_id: controller} of specified types

        """
        if not ctrl_types:
            # return all controllers
            return self.controllers
        else:
            return {ctrl_id: ctrl for ctrl_id, ctrl in self.controllers.items() if isinstance(ctrl, tuple(ctrl_types))}

    def create_env_func(
        self,
        episode_length,
        wrapper: gym.Wrapper = None,
        fixed_start: datetime = None,
        normalize_actions: bool = True,
        history: ModelHistory = None,
    ):
        """
        Creates an environment which encapsulates the power system in a way that RL algorithms can interact with it.
        Based on the OpenAI Gym environment API.

        Args:
            episode_length (int): how many environment interaction steps to complete before resetting the environment
            wrapper (gym.Wrapper): any class to wrap around the standard ControlEnv API provided within this repository
            (used for example to map from multi-agent environment to single-agent environment)
            fixed_start (datetime): whether to run on a fixed given day
            normalize_actions (bool): whether or not to normalize the action space
            history (ModelHistory): logger

        Returns:
            wrapper(ControlEnv): environment instance

        """
        # ToDo: multiple threads using SubprocVecEnv, one thread using DummyVecEnv?

        def init_env():
            env = ControlEnv(
                system=self,
                episode_length=episode_length,
                fixed_start=fixed_start,
                normalize_action_space=normalize_actions,
                history=history,
            )
            if wrapper:
                env = wrapper(env)
            return env

        self.env_func = init_env()
        return self.env_func

    def limit_date_range(self, start: datetime, end: datetime):
        if not start >= self.date_range[0]:
            raise ValueError(f"Start time has to be after {self.date_range[0]}.")
        if not end <= self.date_range[1]:
            raise ValueError(f"End time has to be before {self.date_range[1]}.")
        self.date_range[0] = start
        self.date_range[1] = end
        if start + self.episode_horizon >= end:
            raise ValueError(
                f"Start time {start} + episode horizon {self.episode_horizon} has to be before end time {end}."
            )
        self.date_range[1] = end - self.episode_horizon

    def _calc_date_range(self) -> list[datetime]:

        # check the overlap of date ranges of all data providers
        date_range = []
        data_providers = [child.data_providers for child in self.get_children() if hasattr(child, "data_providers")]
        data_providers = [item for sublist in data_providers for item in sublist]
        if data_providers:
            for dp in data_providers:
                dp_date_range = dp.get_date_range()
                date_range = dp_date_range if not date_range else date_range
                date_range[0] = max(date_range[0], dp_date_range[0])
                date_range[1] = min(date_range[1], dp_date_range[1])

                if date_range[0] >= date_range[1]:
                    raise EntityError(
                        self,
                        "Some of the given DataProviders have no overlap in their date ranges.",
                    )

            # upper limit reduced by control horizon to not run into problems during update() and forecasting
            date_range[1] = date_range[1] - self.episode_horizon

        else:  # if no data providers are defined
            date_range = [datetime(1900, 1, 1), datetime(2100, 12, 31)]

        return date_range

    def sample_start_date(self, fixed_start: datetime = None) -> str:
        """
        Get start date and time for a power system simulation. Can be a fixed start time or a start time
        sampled randomly from the date range for which the data sources within the system are configured.
        Currently, the start time will always be at the beginning of the day.

        Args:
            fixed_start (datetime): Specific timestamp to start the simulation.
                If None, a random start time will be sampled.

        Returns:
            datetime: day, month, year, hour, minutes when to reset the system

        """
        if fixed_start:
            if not self.date_range[0] <= fixed_start <= self.date_range[1]:
                raise EntityError(self, "Fixed start is not within the date range of the provided data providers.")
            date = fixed_start
        else:
            date = self.date_range[0] + timedelta(
                # Get a random amount of seconds between `start` and `end`
                seconds=random.randint(0, int((self.date_range[1] - self.date_range[0]).total_seconds())),
            )
            date = date.replace(hour=0, minute=0, second=0, microsecond=0)  # set to start of day
        return date

    def step(
        self,
        obs: dict = None,
        rl_action_callback: Callable = None,
        history: ModelHistory = None,
    ) -> tuple[dict, dict, dict]:
        """
        Runs one time step of the power system simulation. This includes fixing the actions computed by the system's
        controllers within the Pyomo model, solving the Pyomo model, and updating the states and data sources within
        the model. The return values of this function adhere to the OpenAI Gym API, such that the method can be called
        within the ControlEnv step() function to obtain the information required for RL training.

        Args:
            obs (dict): dictionary of {controller_id: controller_observation}
            rl_action_callback (Callable): callback used to retrieve actions from RL controllers
            model_history (ModelHistroy, optional): Instance of ModelHistory to log the system model.

        Returns:
            dict: dictionary of observations of all controllers {controller_id: controller_observation} AFTER applying
            the actions to the system
            dict: dictionary of rewards of all controllers {controller_id: controller_observation} AFTER applying
            the actions to the system. The rewards depend on the current state of the system and the action applied
            in this state, as well as on whether the action had to be corrected due to safety constraints.
            dict: additional information

        """
        # write actions into model instance
        penalties = {}
        for ctrl_id, ctrl in self.controllers.items():
            # step through controllers, compute actions, fix actions --> ToDo: inside compute_action?
            ctrl_obs = obs[ctrl_id] if obs else None
            verified_action, verification_penalty = ctrl.compute_control_input(
                obs=ctrl_obs, input_callback=rl_action_callback
            )
            penalties[ctrl_id] = verification_penalty  # will be 0 for optimal controllers, dummy controllers
            # fix computed inputs in model instance
            nodes = ctrl.get_nodes()
            for node in nodes:
                if node.id in verified_action.keys():
                    node_actions = verified_action[node.id]
                    node.fix_inputs(node_actions)

        # step the model (corresponding to the environment)
        inst = self.instance
        results = self.solver.solve(inst, warmstart=True)
        # inst.pprint()
        # inst.solutions.load_from(results)
        # inst.solutions.store_to(results)  # ToDo: necessary?
        # catch error if model solving is infeasible
        if results.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            with open("infeasible_model.log", "w") as f:
                inst.pprint(f)
            raise InstanceError(self, "Solving the model with current inputs is infeasible or unbounded")

        # execute unmodeled updates
        self.unmodeled_update()

        # get objective values
        self.compute_cost()

        costs = {}
        for ctrl_id, ctrl in self.controllers.items():
            costs[ctrl_id] = ctrl.get_cost(inst)

        # add verification costs
        costs = {agent: cost + penalties[agent] for agent, cost in costs.items()}

        if history:
            history.log(inst, self.t)

        # at this point we imply: t = t+1

        # unfix all variables - we fix the current state during update()
        inst.unfix_all_vars()

        # advance data sources etc.
        self.update()

        # get observations
        obs, _ = self.observe()

        info = {}
        return obs, costs, info

    def observe(self) -> dict:
        """
        Observe all system states and external variables

        Returns:
            dict: dictionary of {controller_id: controller_observation}
        """
        return self.observer.observe(self.controllers)

    def terminal_step(
        self,
        history: ModelHistory = None,
    ) -> None:
        """
        Terminal step of the simulation.
        Here, we

        (1) log the final state of the system to the history

        (2) compute a perfect knowledge optimal control trajectory
            and store it in the prediction horizon of the last history log.
            This can be used to estimate the value of the terminal system state.
            For example, the perfect knowledge trajectory would in the standard case discharge all
            existing ESS within the horizon.
            Accordingly, the predicted final state of t_terminal + horizon is
            identical across different controllers, which allows for better comparability.

        Args:
            model_history (ModelHistroy, optional): Instance of ModelHistory to log the system model.
        """
        if not history:
            return

        # temporarily set all data providers to perfect knowledge
        data_providers = [
            dp for child in self.get_children() if hasattr(child, "data_providers") for dp in child.data_providers
        ]
        for dp in data_providers:
            dp.set_perfect_knowledge(True)

        # override forecasts with perfect knowledge predictions
        self.update_data(self.t)

        for dp in data_providers:
            dp.set_perfect_knowledge(False)

        # solve the model
        # all inputs will be set by the global solver
        inst = self.instance
        results = self.solver.solve(inst, warmstart=True)
        # catch error if model solving is infeasible
        if results.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            raise InstanceError(self, "Solving the model with current inputs is infeasible or unbounded")

        # get objective values
        self.compute_cost()

        if history:
            history.log(inst, self.t)

    def pprint(self) -> None:
        """
        Prints an overview of the system members.
        """

        print_indentation = "   "

        def print_node_tree(node: Node, indentation: str) -> str:
            cntrlr = f"-- {node.controller.name}" if node.controller else ""
            output = f"{indentation}{node.id} ({node.__class__.__name__}): {node.name} {cntrlr} \n"
            for n in node.nodes:
                output += print_node_tree(n, indentation + print_indentation)

            return output

        output = "\nSYSTEM OVERVIEW \n \n"
        output += "Nodes: \n"
        for n in self.nodes:
            output += print_node_tree(n, print_indentation)

        output += "\nLines: \n"
        for line in self.lines:
            output += f"{print_indentation}{line.id}: {line.src.name} -- {line.dst.name} \n"

        print(output)

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int = 0) -> Expression:
        return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])

    def compute_cost(self) -> None:
        """
        Computes the cost based on the specified cost_fcn and stores the result in the systems' cost parameter.
        """
        # We compute the cost only for the nominal scenario here
        # This is because we are only interested in the realized cost, i.e, at index 0.
        for t in range(self.horizon_int):
            self.set_value(self.instance, "cost", value(self.cost_fcn(CostScenario(), self.instance, t)), idx=t)

        for node in self.nodes:
            node.compute_cost()

    def get_children(self) -> list[ModelEntity]:
        children = copy(self.nodes) + copy(self.lines)
        all_children = copy(children)
        for child in children:
            temp = child.get_children()
            if temp:
                all_children += temp if isinstance(temp, list) else [temp]

        return all_children

    def get_first_level_non_structure_nodes(self) -> List[ModelEntity]:
        """
        This "unpacks" all nodes/buses which are first level children of StructureNodes.
        Used for determining all buses relevant for power flow caluculations.

        Returns:
            list[ModelEntity]: List of child entities.
        """

        def unpack_structure_node(node) -> List[ModelEntity]:
            return [unpack_structure_node(n) if isinstance(n, StructureNode) else n for n in node.nodes]

        children = copy(self.nodes)
        all_children = []
        for child in children:
            if isinstance(child, StructureNode):
                all_children += unpack_structure_node(child)
            else:
                all_children.append(child)

        return all_children

    def get_all_data_providers(self, node=None, provider=None) -> list[DataProvider]:
        """
        Recursively retrieves all data providers associated with a given node and its children.

        Args:
            node (Node): The node to retrieve data providers from.
            provider (list[DataProvider], optional): The list of data providers \
                (only used for recursion). Defaults to None.

        Returns:
            list[DataProvider]: A list of all data providers associated with the node.
        """
        providers = []
        node = self if node is None else node
        if provider is None:
            provider = []
        if hasattr(node, "data_providers"):
            provider.extend(node.data_providers)
        if len(node.get_children()) > 0:
            for child in node.get_children():
                providers.extend(self.get_all_data_providers(child, provider[:]))
        else:
            providers.extend(provider)
        return providers


class Line(ControllableModelEntity):
    CLASS_INDEX = "l"

    def __init__(self, src: Node, dst: Node, config: dict = {}, name: str = "line") -> Line:
        """
        Power transmission line.
        Sublasses have to implement specific variables and parameters.
        We consider lines to be undirected in principle, however,
        the sign of current flow would be based on the src-dst convention.

        Args:
            src (Node): Source node.
            dst (Node): Target node.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
            name (str, optional): Name of the line object. Defaults to "line".
        """
        super().__init__(name, config)

        # we consider lines to be undirected in principle
        # however, the sign of current flow is based on the src-dst convention
        self.src = src
        self.dst = dst

        # self.id = self.CLASS_INDEX + "_" + src.id + "_" + dst.id
        # This does not work anymore since the ids are set on sys init now
        self.id = self.CLASS_INDEX + "_" + "%05x" % random.randrange(16**5)

    def empty_copy(self) -> Line:
        """
        Creates a fresh copy of the line.

        Returns:
            Line: Cloned line instance.
        """
        return self.__class__(self.src, self.dst, self.config, self.name)


class Node(ControllableModelEntity):
    CLASS_INDEX = "nx"

    @classmethod
    def _augment_model_elements(cls, model_elements: List[ModelElement]) -> List[ModelElement]:
        """
        This method adds initial state variables and a cost variable
        All of this is appended to the given model_elements list and returned.

        Args:
            model_elements (List[ModelElement]): Model elements list so far.

        Returns:
            List[ModelElement]: Model elements list with added limit variables and cost variable
        """
        new_model_elements = []
        for el in model_elements:
            if el.type in [ElementTypes.STATE]:
                new_model_elements.append(
                    ModelElement(
                        f"{el.name}_init",
                        ElementTypes.CONSTANT,
                        f"{el.doc} initial value",
                        domain=el.domain,
                        bounds=el.bounds,
                    )
                )

        cst = ModelElement("cost", ElementTypes.COST, "dispatch cost")
        new_model_elements.append(cst)

        return model_elements + new_model_elements

    def __init__(self, name: str, config: dict = {}) -> Node:
        """
        Base class providing functionality for busses and components.

        Args:
            name (str): Name of the Node object.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
        """
        super().__init__(name, config)

        self.controller = None
        self.is_valid = False

        self.t = None  # current time
        self.tau = None
        self.horizon = None

        self.robust_constraint_builder = None  # set in add_to_model()

        self.nodes: list[Node] = []

    def empty_copy(self, with_children: bool = True, with_data_providers: bool = True) -> Node:
        """
        Creates a fresh copy of the node.

        Args:
            with_children (bool): Whether to clone the node's children.
            with_data_providers (bool): Whether to clone the node's data providers.

        Returns:
            Node: Cloned node instance.
        """
        new_node = self.__class__(self.name, self.config)

        if with_data_providers:
            for dp in self.data_providers:
                new_node.add_data_provider(dp.empty_copy())

        if with_children:
            for node in self.nodes:
                new_node.add_node(node.empty_copy(with_children=with_children, with_data_providers=with_data_providers))

        return new_node

    def set_id(self, parent_identity: str = "", number: int = 0) -> None:
        """
        Generates and sets the node id.
        This is called by the parent entity, i.e., the next higher entity in the object tree.

        Args:
            parent_identity (str, optional): Id of the parent entity. Defaults to "".
            number (int, optional): Number assigned by the parent entity. Defaults to 0.
        """

        parent_number = re.findall(r"\d+", parent_identity.split(".")[-1])
        parent_number = parent_number[0] if len(parent_number) > 0 else ""

        own_id = self.CLASS_INDEX + str(parent_number) + str(number)
        # possible alternative: own_id = self.CLASS_INDEX + "_" + '%03x' % random.randrange(16**3)
        self.id = parent_identity + "." + own_id if parent_identity != "" else own_id

    def add_to_model(self, model: ConcreteModel, **kwargs) -> None:
        """
        This method adds the calling entity to the given (global) pyomo model.
        To this end, we
            - declare and add a new pyomo block named by self.id (the entity's global id).
            - call _get_model_elements() to retrieve the entity's model elements (variables and parameters).
            - call _augment_model_elements() to add additional model elements (constraints etc.).
            - check the configuration dict for completeness based on the defined model elements.
            - add all model elements to the previously declared pyomo block.

        We also store a reference to the global model in self.model.

        Args:
            model (ConcreteModel): Global pyomo model.
            **kwargs
        """
        self.model = model  # store reference to overall model internally

        # Attention: For the System "tau", and "horizon" are timedelta, for Nodes they are numeric (float, int).
        self.tau = kwargs["tau"]
        self.horizon = kwargs["horizon"]

        rsetattr(self.model, self.id, ConcreteModel())

        for idx, node in enumerate(self.nodes):
            node.set_id(self.id, idx)
            node.add_to_model(self.model, tau=self.tau, horizon=self.horizon)

        self.model_elements = self._augment_model_elements(self._get_model_elements())
        self.model_elements = self._add_constraints(self.model_elements)

        self._check_config(self.config)

        self.robust_constraint_builder = RobustConstraintBuilder(self)
        self.robust_constraint_builder.expand_robust_constraints()

        for el in self.model_elements:
            self._add_model_element(el)

    def validate_data_providers(self, horizon: timedelta, tau: timedelta) -> None:
        """
        Validates if data providers have compatible configurations.

        Args:
            horizon (timedelta): Forecast horizon.
            tau (timedelta): Sample time.
        """

        # check if all dataproviders have an appropriate forecast horizon, data frequency
        for dp in self.data_providers:
            if horizon != dp.horizon:
                raise EntityError(
                    self,
                    f"The Data Provider providing {dp.get_variables()} must implement a forecast horizon of"
                    f" {horizon} instead of {dp.horizon}",
                )
            if dp.frequency != tau:
                raise EntityError(
                    self,
                    f"The Data Provider providing {dp.get_variables()} must implement data frequency {tau} instead of"
                    f" {dp.frequency}",
                )

        for node in self.nodes:
            node.validate_data_providers(horizon, tau)

        self.is_valid = True

    def validate_controller(self, controller_ids: list, controllers: dict) -> Tuple[list, dict]:
        """
        Used to check whether all nodes which require a controller have one assigned. All unique controllers are
        added to a list of controllers maintained by the power system.

        Args:
            controller_ids (list): unique controller IDs already registered by the system
            controllers (list): unique controllers already registered by the system

        Returns:
            list: IDs of controllers within the system
            dict: dictionary of {controller_id: controller} within the system

        """
        if self.n_inputs() > 0:
            if self.controller is None:
                raise EntityError(self, "Controller required!")
            else:
                id = self.controller.get_id()
                # add unique control to list
                if id:
                    if id not in controller_ids:
                        controller_ids.append(id)
                        controllers[id] = self.controller
        for node in self.nodes:
            controller_ids, controllers = node.validate_controller(controller_ids, controllers)

        return controller_ids, controllers

    def add_node(self, node: Node) -> Node:
        """
        Adds a subordinate node.
        The added node's id is set according to its position in the model hierarchy.

        Args:
            node (Node): Node istance to add.

        Returns:
            Node: Node instance.
        """

        self.nodes.append(node)
        return self

    def reset(self, instance: ConcreteModel, at_time: datetime) -> None:
        """
        Stores the current global model instance and initializes parameters according to their configuration.
        It additionally loads the "current" values from data providers.

        Args:
            instance (ConcreteModel): Global model instance.
            at_time (datetime): Timestamp of "now".
        """
        if self.is_valid is False:
            raise EntityError(self, "Node has not been validated")

        self.instance = instance

        # override param values if ParamInitializers have been provided
        for el in self.model_elements:
            if el.type == ElementTypes.CONSTANT and el.initialize is None:
                val = self.config[el.name]
                if isinstance(val, ParamInitializer):
                    self.set_value(self.instance, el.name, val.get_init_val(at_time))

        # write state init values to the state variables (index 0) so that they correctly show up
        # in the first RL observation
        for el in self.model_elements:
            if el.type == ElementTypes.STATE:
                self.set_value(
                    self.instance, el.name, self.get_value(self.instance, f"{el.name}_init"), idx=0, fix_value=True
                )
                try:
                    # try to set the bound variables of uncertain states
                    self.set_value(
                        self.instance,
                        f"{el.name}_lb",
                        self.get_value(self.instance, f"{el.name}_init"),
                        idx=0,
                        fix_value=True,
                    )
                    self.set_value(
                        self.instance,
                        f"{el.name}_ub",
                        self.get_value(self.instance, f"{el.name}_init"),
                        idx=0,
                        fix_value=True,
                    )
                except EntityError:
                    pass

        self._update_data(at_time)

        for node in self.nodes:
            node.reset(instance, at_time)

    def update(self, at_time: datetime) -> None:
        """
        This reads data providers and executes the dynamics of self and all subordinate nodes.
        Results are written to current model instance.
        It also calls _additional_updates().

        Args:
            at_time (datetime): Timestamp of "now".
        """

        self._update_state()
        self._step_solution()
        self._update_data(at_time)

        for node in self.nodes:
            node.update(at_time)

    def unmodeled_update(self) -> None:
        """
        System updates that are not modeled.
        This could for example be the "true" dynamics of the system, or maniputations of parameters.
        """
        self._unmodeled_updates()
        for node in self.nodes:
            node.unmodeled_update()

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Returns the node's cost as pyomo expression at time t.

        .. math::
            cost = \\sum_{i \\in children} cost_i

        Returns:
            Expression: Cost.
        """

        if self.nodes:
            return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])
        else:
            return 0.0

    def compute_cost(self) -> None:
        """
        Computes the cost based on the specified cost_fcn and stores the result in the node's cost parameter.
        """
        for t in range(self.horizon):
            self.set_value(self.instance, "cost", value(self.cost_fcn(CostScenario(), self.instance, t)), idx=t)

        for node in self.nodes:
            node.compute_cost()

    def get_children(self) -> list[ModelEntity]:
        children = copy(self.nodes)
        all_children = copy(children)
        for child in children:
            temp = child.get_children()
            if temp:
                all_children += temp if isinstance(temp, list) else [temp]

        return all_children

    def _add_constraints(self, model_elements: List[ModelElement]) -> List[ModelElement]:
        """This method adds all node constraints
           All of this is appended to the given model_elements list and returned.
        Args:
            model_elements (List[ModelElement]): Model elements list so far

        Returns:
            List[ModelElement]: Model elements list with appended constraints
        """
        new_model_elements = []

        new_model_elements += self._get_additional_constraints()
        new_model_elements += self._get_dynamic_fcn()

        return model_elements + new_model_elements

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        Returns constraints on all state variables representing how the states change between timesteps.

        Returns:
            List[ModelElement]: Generated constraints.
        """
        return []

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Returns additional constraints on model variables.
        This is a utility to keep the _get_model_elements() method clean.

        Returns:
            List[ModelElement]: List of additional constraint elements.
        """
        return []

    def _update_state(self):
        """
        Updates states by moving one timestep "forward", i.e., state[t] <- state[t+1].
        This method can be overwritten by subclasses to implement the "true" dynamics of the system.
        """
        for el in [el for el in self.model_elements if el.type == ElementTypes.STATE and el.indexed is True]:
            # one timestep forward, i.e. state[t] <- state[t+1]
            values = self.get_value(self.instance, el.name)
            values = np.round(values, 5)  # round to avoid warnings / numerical issues
            for t in range(0, self.horizon):
                self.set_value(
                    self.instance,
                    el.name,
                    values[t + 1],
                    idx=t,
                    fix_value=(t == 0),
                )

    def _step_solution(self):
        """
        Steps all variables (except states) forward by one timestep.
        This is mainly to provide a better warm-start for the solver in the next iteration.
        """
        for el in self.model_elements:
            if el.type in [ElementTypes.VAR, ElementTypes.INPUT, ElementTypes.COST] and el.indexed is True:
                # one timestep forward, i.e. var[t] <- var[t+1]
                values = self.get_value(self.instance, el.name)
                values = np.round(values, 5)  # round to avoid warnings / numerical issues
                for t in range(0, self.horizon):
                    self.set_value(
                        self.instance,
                        el.name,
                        values[t + 1],
                        idx=t,
                    )

    def update_data(self, at_time: datetime):
        """
        Reads data providers for self and all subnodes.

        Args:
            at_time (datetime): Timestamp of "now".
        """
        self._update_data(at_time)
        for node in self.nodes:
            node.update_data(at_time)

    def _update_data(self, at_time: datetime):
        """
        Reads node data providers.

        Args:
            at_time (datetime): Timestamp of "now".
        """
        obs_dict = {}
        uncertainty_bounds_dict = {}

        for dp in self.data_providers:
            obs_dict.update(dp.observe(at_time))
            if dp.forecaster.is_uncertain:
                uncertainty_bounds_dict.update(dp.observation_bounds(at_time))

        # update model
        for el in self.model_elements:
            if el.type == ElementTypes.DATA:
                self.set_value(self.instance, el.name, obs_dict[el.name])

                # update uncertainty if (1) uncertain forecast and (2) variable is used in robust constraint(s)
                # We might only need condition (2) ?
                if el.name in uncertainty_bounds_dict.keys() and self.has_pyomo_element(f"{el.name}_lb", self.instance):
                    self.set_value(
                        self.instance, f"{el.name}_lb", [bound[0] for bound in uncertainty_bounds_dict[el.name]]
                    )
                    self.set_value(
                        self.instance, f"{el.name}_ub", [bound[1] for bound in uncertainty_bounds_dict[el.name]]
                    )

    def _unmodeled_updates(self) -> None:
        """
        Unmodeled update actions can be defined here.
        This could for example be the "true" dynamics of the system, or maniputations of parameters.
        """


class StructureNode(Node):
    CLASS_INDEX = "sn"

    def __init__(self, name: str, config: dict = {}) -> StructureNode:
        """
        Structure nodes model abstract entities such as energy communities, P2P markets, etc.
        They do not implement any physical characteristcs and are ignored in the power flow constraints.

        Args:
            name (str): Name of the StructureNode object.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
        """
        super().__init__(name, config)


class Bus(Node):
    CLASS_INDEX = "n"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        """
        Returns primary model elements.
        Busses specify active power (p), reactive power (q), voltage magnitude (v), and voltage angle (d).

        Returns:
            List[ModelElement]: Model elements.
        """
        model_elements = [
            ModelElement("p", ElementTypes.VAR, "active power", bounds=[-1e6, 1e6]),
            ModelElement("q", ElementTypes.VAR, "reactive power", bounds=[-1e6, 1e6]),
            ModelElement("v", ElementTypes.VAR, "voltage magnitude", bounds=[0.9, 1.1]),
            ModelElement("d", ElementTypes.VAR, "voltage angle", bounds=[-15, 15]),
        ]
        return model_elements

    def __init__(self, name: str, config: dict = {}) -> Bus:
        """
        Bus.

        Args:
            name (str): Name of the Bus object.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
        """
        super().__init__(name, config)

        self.stand_alone = True  # indicates if the bus is child of a StructureNode (energy community, P2P market)

    def set_as_structure_member(self) -> None:
        """
        Sets a flag indicating that the bus is a member of some structure (e.g., energy community, P2P market).
        """
        self.stand_alone = False

    def set_as_stand_alone(self) -> None:
        """
        Sets a flag indicating that the bus is stand-alone.
        """
        self.stand_alone = True

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Returns additional constraints on model variables.
        By default only internal power balance constraints.

        Returns:
            List[ModelElement]: List of additional constraint elements.
        """
        return self._get_internal_power_balance_constraints()

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        """
        Returns the internal power balance constraints of the node.

        Returns:
            List[ModelElement]: Generated constraints.
        """

        def bus_pb(model, t):
            if self.nodes:
                return (
                    quicksum(
                        [
                            c.get_pyomo_element("p", model)[t] if c.has_pyomo_element("p", model) else 0.0
                            for c in self.nodes
                        ]
                    )
                    == self.get_pyomo_element("p", model)[t]
                )
            else:
                return 0.0 == self.get_pyomo_element("p", model)[t]

        def bus_qb(model, t):
            if self.nodes:
                return (
                    quicksum(
                        [
                            c.get_pyomo_element("q", model)[t] if c.has_pyomo_element("q", model) else 0.0
                            for c in self.nodes
                        ]
                    )
                    == self.get_pyomo_element("q", model)[t]
                )
            else:
                return 0.0 == self.get_pyomo_element("q", model)[t]

        c_pb = ModelElement("c_pb", ElementTypes.CONSTRAINT, "active power balance", expr=bus_pb)
        c_qb = ModelElement("c_qb", ElementTypes.CONSTRAINT, "reactive power balance", expr=bus_qb)

        return [c_pb, c_qb]


class Component(Node):
    CLASS_INDEX = "x"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        """
        Returns primary model elements.

        Returns:
            List[ModelElement]: Model elements.
        """
        raise NotImplementedError

    def __init__(self, name: str, config: dict = {}) -> Component:
        """
        Generic power system device.
        We use this to model generators, loads, storage systems, etc.

        Args:
            name (str): Name of the Component object.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
        """
        super().__init__(name, config)

    def add_node(self, node: Node) -> Node:
        """
        Components cannot have subordinate nodes.

        Raises:
            EntityError
        """
        raise EntityError(self, "Components cannot have sub-nodes")
