"""
Core power system entities.
"""
from __future__ import annotations
from typing import Union, Tuple, Callable, Dict, List
import re
from copy import copy
from pyomo.core import ConcreteModel, Set, Objective, value, quicksum, Expression
from pyomo.opt import TerminationCondition
from pyomo.environ import SolverFactory
from pyomo.opt.solver import OptSolver
from datetime import datetime, timedelta
from randomtimestamp import randomtimestamp
import pickle
import gymnasium as gym
import logging
from collections import OrderedDict

from commonpower.data_forecasting import DataProvider
from commonpower.modelling import ElementTypes, ModelElement, ModelEntity, ModelHistory, ControllableModelEntity
from commonpower.data_forecasting import *
from commonpower.utils.param_initialization import ParamInitializer
from commonpower.utils import rsetattr
from commonpower.utils.cp_exceptions import EntityError, InstanceError
from commonpower.control.environments import ControlEnv


class PowerFlowModel:
    """
    Generic class to model power flow constraints.
    """

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
            if not connected_lines and len(nodes) > 1:
                logging.warning(f"The node {node.name} has no power lines connected to it")
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
        self.horizon = None  # forecast horizon
        self.start_time = None  # start of simulation time
        self.continuous_control = None  # whether to consider an infinite control horizon

        self.date_range = None  # date range of data

        self.power_flow_model = power_flow_model

        self.env_func = None

        self.solver = None  # solver for optimization problem

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
            self.nodes[-1].set_id("", len(self.nodes) - 1)
        else:
            node.set_id("", at_index)
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
        horizon: int = 24,
        tau: timedelta = timedelta(hours=1),
        continuous_control: bool = False,
        solver: OptSolver = SolverFactory("gurobi"),
    ) -> None:
        """
        Initializes the system.
        This constructs the pyomo model of the system by traversing through the
        object tree (nodes, lines, power flow), calling self.add_to_model().
        It validates if data providers have compatible configurations and
        if controllers have been defined appropriately.

        Args:
            horizon (int, optional): Control horizon. This specifies the number of timesteps
                the controllers "look into the future". Defaults to 24.
            tau (timedelta, optional): Sample time, i.e., the period of time between to control actions.
                This needs to match the frequency of data providers. Defaults to timedelta(hours=1).
            continuous_control (bool): whether to use an infinite control horizon
            solver (OptSolver, optional): Name of the solver for the optimization problem that will be called by Pyomo.
        """
        self.tau = tau
        self.horizon = horizon
        self.continuous_control = continuous_control
        self.solver = solver

        self.add_to_model(ConcreteModel())

        # check if all dataproviders and constants have been defined
        for node in self.nodes:
            node.validate(horizon, tau)
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

        # check if all data sources have the same date range
        date_range = []
        data_providers = [child.data_providers for child in self.get_children() if hasattr(child, "data_providers")]
        data_providers = [item for sublist in data_providers for item in sublist]
        if data_providers:
            for dp in data_providers:
                dp_date_range = dp.get_date_range()
                date_range = dp_date_range if not date_range else date_range
                if (dp_date_range[0] != date_range[0]) or (dp_date_range[1] != date_range[1]):
                    raise EntityError(
                        self,
                        f"The Data Provider providing {dp.get_variables()} is defined over an inconsistent date range"
                        f" {dp_date_range}",
                    )
            # upper limit reduced by one day to not run into problems during update()
            date_range[1] = date_range[1] - timedelta(days=1)

        self.date_range = date_range

    def reset(self, at_time: Union[str, datetime]) -> None:
        """
        Resets the system model to the state at a certain timestamp.
        It creates a clone of the system's "raw" pyomo model which will then be used for simulation.
        Furthermore, entities' parameters are initialized according to their configuration.

        Args:
            at_time (Union[str, datetime]): Timestamp to begin the simulation from.
        """
        self.t = pd.to_datetime(at_time) if isinstance(at_time, str) else at_time
        self.start_time = self.t
        self.instance = self.model.clone()

        for node in self.nodes:
            node.t = self.t
            node.reset(self.instance, at_time)

        # reset training history of RL controllers
        for ctrl in self.controllers.values():
            ctrl.reset_history()

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

        self.model.t = Set(initialize=range(0, self.horizon + 1))

        # the tau value in the model is a float indicating tau / 1h, i.e., the faction/multiple of one hour.
        tau_float = self.tau / timedelta(hours=1)

        # self.model.tau = Param(initialize=tau_float)
        # self.model.horizon = Param(initialize=self.horizon, domain=pyo.NonNegativeIntegers)

        for node in self.nodes:
            node.add_to_model(model, tau=tau_float, horizon=self.horizon)

        for line in self.lines:
            line.add_to_model(model)

        self.power_flow_model.add_to_model(model, self.nodes, self.lines)

        self.model_elements = self._augment_model_elements(self._get_model_elements())

        for el in self.model_elements:
            self._add_model_element(el)

        def obj_fcn(model):
            return quicksum([n.cost_fcn(model, t) for t in range(self.horizon) for n in self.nodes])

        self.model.obj1 = Objective(expr=obj_fcn)

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

    def create_env_func(self, wrapper: gym.Wrapper = None, fixed_day: datetime = None, normalize_actions: bool = True):
        """
        Creates an environment which encapsulates the power system in a way that RL algorithms can interact with it.
        Based on the OpenAI Gym environment API.

        Args:
            wrapper (gym.Wrapper): any class to wrap around the standard ControlEnv API provided within this repository
            (used for example to map from multi-agent environment to single-agent environment)
            fixed_day (datetime): whether to run on a fixed given day
            normalize_actions (bool): whether or not to normalize the action space

        Returns:
            wrapper(ControlEnv): environment instance

        """
        # ToDo: multiple threads using SubprocVecEnv, one thread using DummyVecEnv?
        if not self.controllers:
            raise SystemError("No control assigned to system")

        def init_env():
            env = ControlEnv(
                system=self,
                continuous_control=self.continuous_control,
                fixed_day=fixed_day,
                normalize_action_space=normalize_actions,
            )
            if wrapper:
                env = wrapper(env)
            return env

        self.env_func = init_env()
        return init_env()

    def global_observation_space(self, global_obs_mask: List[Tuple[Union[ModelEntity, list]]]):
        """
        Translates a list of either model entities or strings (identifiers of model entities) to the respective
        observation space corresponding to the bounds of the observed quantities.

        Args:
            global_obs_mask (ist[Tuple[Union[ModelEntity, list]]]): list of either model entities or strings
            (identifiers of model entities) to be added to the observation of a controller

        Returns:
            gym.spaces.Dict: observation space for the quantities in the global_obs_mask

        """
        global_obs_space = {}
        # rewrite global obs_mask to local obs_maks:
        local_obs_mask = {}
        nodes = [item[0] for item in global_obs_mask]
        obs_el = [item[1] for item in global_obs_mask]
        for count, node in enumerate(nodes):
            local_obs_mask[node.id] = obs_el[count]
        for node in nodes:
            node_obs_space = node.observation_space(local_obs_mask)
            global_obs_space[node.id] = node_obs_space
        global_obs_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in global_obs_space.items()})
        return global_obs_space

    def global_obs(self, global_obs_mask: List[Tuple[Union[ModelEntity, list]]]) -> Dict:
        """
        Gets values of model elements in global_obs_mask which should be added to the observation of a controller

        Args:
            global_obs_mask List[Tuple[Union[ModelEntity, list]]]: model elements which should be added to the
            observation of a controller

        Returns:
            dict: dictionary of {entity_id: entity_observation} of all model elements in global_obs_mask

        """
        obs = OrderedDict()
        # rewrite global obs_mask to local obs_maks:
        local_obs_mask = {}
        nodes = [item[0] for item in global_obs_mask]
        obs_el = [item[1] for item in global_obs_mask]
        for count, node in enumerate(nodes):
            local_obs_mask[node.id] = obs_el[count]
        for node in nodes:
            node_obs = node.observe(local_obs_mask)
            obs[node.id] = node_obs
        return obs

    def sample_start_date(self, fixed_day: datetime = None) -> str:
        """
        Get start date and time for a power system simulation. Can be a fixed day or a day sampled randomly from the
        date range for which the data sources within the system are configured. Currently, the start time will always
        be at the beginning of the day.

        Args:
            fixed_day (datetime): Fixed day in the pattern "%d.%m.%Y", e.g., "27.11.2016"

        Returns:
            datetime: day, month, year, hour, minutes where to reset the system

        """
        if fixed_day:
            date = fixed_day
        else:
            date = randomtimestamp(start=self.date_range[0], end=self.date_range[1], text=True, pattern="%d.%m.%Y")
        date = date + " 00:00"
        return date

    def step(
        self, obs: dict = None, rl_action_callback: Callable = None, history: ModelHistory = None
    ) -> (dict, dict, bool, bool, dict):
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
            bool: whether the episode has terminated (we assume that all agents terminate an episode at the same time,
            as we have a centralized time management). Always false for continuous control
            bool: same as above (but the gymnasium API makes a difference between terminated and truncated, which
            can be useful for other environments but is not needed in our case)
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
            raise InstanceError(self, "Solving the model with current inputs is infeasible or unbounded")

        # get objective values
        self.compute_cost()

        costs = {}
        for ctrl_id, ctrl in self.controllers.items():
            costs[ctrl_id] = ctrl.get_cost(inst)

        if history:
            history.log(inst, self.t)

        # at this point we imply: t = t+1

        # unfix all variables - we fix the current state during update()
        inst.unfix_all_vars()

        # advance data sources etc.
        self.update()

        # get observations
        obs, _ = self.observe()
        # add verification costs
        costs = {agent: cost + penalties[agent] for agent, cost in costs.items()}

        # reached end of control horizon?
        terminated = self._is_done()
        truncated = self._is_done()

        info = {}
        return obs, costs, terminated, truncated, info

    def _is_done(self) -> bool:
        """
        Whether a system reached the end of the control horizon

        Returns:
            bool: True if end of horizon was reached

        """
        if self.continuous_control:
            done = False
        else:
            done = self.t == self.start_time + self.horizon * self.tau
        return done

    def observe(self) -> dict:
        """
        Get observations for all controllers within the system.

        Returns:
            dict: dictionary of {controller_id: controller_observation}

        """
        obs = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            ctrl_obs = OrderedDict()
            nodes = ctrl.get_nodes()
            nodes = [n for n in nodes if not isinstance(n, System)]
            for node in nodes:
                node_obs = node.observe(ctrl.obs_mask)
                if node_obs is not None:
                    ctrl_obs[node.id] = node_obs
            if "global" in ctrl.obs_mask.keys():
                ctrl_obs["global"] = self.global_obs(ctrl.obs_mask["global"])
            obs[ctrl_id] = ctrl_obs
        obs_info = {}
        return obs, obs_info

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
            output += f"{print_indentation}{line.id}: {line.src.id} -- {line.dst.id} \n"

        print(output)

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        return quicksum([n.cost_fcn(model, t) for n in self.nodes])

    def compute_cost(self) -> None:
        """
        Computes the cost based on the specified cost_fcn and stores the result in the systems' cost parameter.
        """
        for t in range(self.horizon):
            self.set_value(self.instance, "cost", value(self.cost_fcn(self.instance, t)), idx=t)

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

        self.id = self.CLASS_INDEX + "_" + src.id + "_" + dst.id


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
            name (str): Name of the node object.
            config (dict, optional): Configuration for defined model elements. Defaults to {}.
        """
        super().__init__(name, config)

        self.controller = None
        self.is_valid = False

        self.t = None  # current time
        self.tau = None
        self.horizon = None

        self.nodes = []

        self.data_providers = []

    def set_id(self, parent_identity: str = "", number: int = 0) -> None:
        """
        Generates and sets the node id.
        This is called by the parent entity, i.e., the next higher entity in the object tree.

        Args:
            parent_identity (str, optional): Id of the parent entity. Defaults to "".
            number (int, optional): Number assigned by the parent entity. Defaults to 0.
        """
        if self.id:
            raise EntityError(self, f"This node has already been added to a parent. Its current id is {self.id}")
        parent_number = re.findall(r"\d+", parent_identity.split(".")[-1])
        parent_number = parent_number[0] if len(parent_number) > 0 else ""

        own_id = self.CLASS_INDEX + str(parent_number) + str(number)
        # possible alternative: own_id = self.CLASS_INDEX + "_" + '%03x' % random.randrange(16**3)
        self.id = parent_identity + "." + own_id if parent_identity != "" else own_id

    def add_data_provider(self, data_provider: DataProvider) -> Component:
        """
        Adds a data provider to the component.
        It will be checked during validation if all model elements which require a data provider are covered.

        Args:
            data_provider (DataProvider): Data provider instance.

        Returns:
            Component: Component instance.
        """
        self.data_providers.append(data_provider)
        return self

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
        self.tau = kwargs["tau"]
        self.horizon = kwargs["horizon"]

        rsetattr(self.model, self.id, ConcreteModel())

        for node in self.nodes:
            node.add_to_model(self.model, tau=self.tau, horizon=self.horizon)

        self.model_elements = self._augment_model_elements(self._get_model_elements())
        self.model_elements = self._add_constraints(self.model_elements)

        self._check_config(self.config)

        for el in self.model_elements:
            self._add_model_element(el)

    def validate(self, horizon: int, tau: timedelta) -> None:
        """
        Validates if data providers have compatible configurations and if controllers have been defined appropriately.

        Args:
            horizon (int): Control horizon.
            tau (timedelta): Sample time.
        """
        # check if all required dataproviders are attached
        needed_from_dataprovider = [el.name for el in self.model_elements if el.type == ElementTypes.DATA]
        if needed_from_dataprovider:
            if not self.data_providers:
                raise EntityError(self, f"Data Providers for {needed_from_dataprovider} required.")
            sourced_params = np.concatenate([s.get_variables() for s in self.data_providers], axis=None)
            if not all(x in sourced_params for x in needed_from_dataprovider):
                raise EntityError(self, f"Data Providers for {needed_from_dataprovider} required.")

        # check if all dataproviders have an appropriate forecast horizon, data frequency
        for dp in self.data_providers:
            dp_horizon = int(dp.horizon / dp.frequency)
            if horizon != dp_horizon:
                raise EntityError(
                    self,
                    f"The Data Provider providing {dp.get_variables()} must implement a forecast horizon of"
                    f" {horizon} instead of {dp_horizon}",
                )
            if dp.frequency != tau:
                raise EntityError(
                    self,
                    f"The Data Provider providing {dp.get_variables()} must implement data frequency {tau} instead of"
                    f" {dp.frequency}",
                )

        # check if all states have corresponding initializer instances in the config
        states = [el for el in self.model_elements if el.type == ElementTypes.STATE]
        for s in states:
            if not isinstance(self.config[f"{s.name}_init"], ParamInitializer):
                raise EntityError(
                    self,
                    f"The initializer of state init parameter {s.name}_init must be of type"
                    f" {ParamInitializer.__name__}",
                )

        for node in self.nodes:
            node.validate(horizon, tau)

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
        if not self.id:
            raise EntityError(self, "Cannot add node. Node id is not defined.")

        self.nodes.append(node)
        self.nodes[-1].set_id(self.id, len(self.nodes) - 1)
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
        self._update_data(at_time)
        self._additional_updates()

        for node in self.nodes:
            node.update(at_time)

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Returns the node's cost as pyomo expression at time t.

        .. math::
            cost = \\sum_{i \\in children} cost_i

        Returns:
            Expression: Cost.
        """

        if self.nodes:
            return quicksum([n.cost_fcn(model, t) for n in self.nodes])
        else:
            return 0.0

    def compute_cost(self) -> None:
        """
        Computes the cost based on the specified cost_fcn and stores the result in the node's cost parameter.
        """
        for t in range(self.horizon):
            self.set_value(self.instance, "cost", value(self.cost_fcn(self.instance, t)), idx=t)

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

        new_model_elements += self._get_internal_power_balance_constraints()
        new_model_elements += self._get_additional_constraints()
        new_model_elements += self._get_dynamic_fcn()

        return model_elements + new_model_elements

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
        Updates states by moving one timestep "forward", i.e., state[t] <- state[t+1]
        """
        for el in self.model_elements:
            if el.type == ElementTypes.STATE:
                # one timestep forward, i.e. state[t] <- state[t+1]
                self.set_value(
                    self.instance, el.name, self.get_value(self.instance, el.name)[1], idx=0, fix_value=True
                )

    def _update_data(self, at_time: datetime):
        """
        Reads data providers.
        """
        obs_dict = {}
        for dp in self.data_providers:
            obs_dict.update(dp.observe(at_time))

        # update model
        for el in self.model_elements:
            if el.type == ElementTypes.DATA:
                self.set_value(self.instance, el.name, obs_dict[el.name])

    def _additional_updates(self) -> None:
        """
        Additional update actions can be defined here.
        This could for example be the "true" dynamics of the system, or maniputations of parameters.
        """


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

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        """
        Components do not have any internal balance constraint.

        Returns:
            List[ModelElement] = []
        """
        return []
