"""
Collection of pre-defined controller types.
"""
import warnings
from collections import OrderedDict
from copy import copy, deepcopy
from typing import Callable, List, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch as th
from pyomo.core import ConcreteModel, Objective, quicksum
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.utils import set_random_seed

from commonpower.core import Node, System
from commonpower.modelling import ControllableModelEntity, ElementTypes
from commonpower.utils.cp_exceptions import ControllerError, EntityError
from commonpower.utils.default_solver import get_default_solver


class BaseController:
    def __init__(
        self,
        name: str,
        obs_types: List[ElementTypes] = [ElementTypes.DATA, ElementTypes.STATE],
        global_obs_elements: List[Tuple[Union[Node, list]]] = None,
        cost_callback: Callable = None,
    ):
        """
        This is the base class for any controller type that will be implemented. It manages assignment of controllable
        entities to the controller and automatically deduces the action space from the bounds of the elements within
        these entities. The observation space of the controller is captured within the obs_maks and defaults to all
        model elements of type STATE or DATA. Additional elements can be passed through global_obs_elements. The most
        important functionality of the controller is to compute the control input, a function that has to be implemented
        by the subclasses.

        Args:
            name (str): name of the controller
            obs_types (List[ElementTypes]): types of model elements of the controlled entities that are included in \
            the observation of the controller.
            global_obs_elements (List[Tuple[Union[Node, list]]]): additional model elements (can also be from outside \
            the controlled entities) that should be observed.
            cost_callback (Callable): function used within the cost function of the controller to compute additional \
            cost terms.

        Returns:
            BaseController
        """
        self.name = name
        self.ctrl_type = None  # specified by subclasses

        self.nodes = []
        self.node_ids = []
        self.top_level_nodes = []

        self.history = {}

        self.obs_mask = {}
        self.obs_types = obs_types
        self.global_obs_elements = global_obs_elements

        self.input_space = None

        self.cost_callback = cost_callback

    def initialize(self):
        """
        Initial set-up of controller.
        """
        self._index_entities()
        self.set_obs_mask(self.obs_types, self.global_obs_elements)
        self.top_level_nodes = self.get_top_level_nodes()
        self.input_space = self.get_input_space()

    def reset_history(self) -> None:
        """
        Has to be implemented by subclasses.

        Returns:
            None

        """
        raise NotImplementedError

    def filter_history_for_time_period(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> dict:
        """
        Filters all element histories for a given time period

        Args:
            start (Union[str, pd.Timestamp]): beginning of the time period.
            If str, should be in format "2016-09-04 00:00:00".
            end (Union[str, pd.Timestamp]): end of the time period. If str, should be in format "2016-09-04 00:00:00".

        Returns:
            (dict): the filtered history.

        """
        filtered_history = {}
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        history_keys = [key for key in self.history.keys()]
        time_stamps = [t[0] for t in self.history[history_keys[0]]]

        start_index = [i for i in range(len(time_stamps)) if time_stamps[i] == start]
        end_index = [i for i in range(len(time_stamps)) if time_stamps[i] == end]
        for key, history in self.history.items():
            filtered_history[key] = [history[start_index[0] + t] for t in range(end_index[0] - start_index[0] + 1)]

        return filtered_history

    def compute_control_input(
        self, obs: OrderedDict = None, input_callback: Callable = None
    ) -> Tuple[OrderedDict, float]:
        """
        Has to be implemented by subclasses.

        Args:
            obs (OrderedDict): observation at current time point
            input_callback (Callable): only needed in training mode - retrieves action selected within training \
                algorithm

        Returns:
            Tuple: tuple containing
                - action (OrderedDict)
                - penalty for action adjustment performed by safety layer (float).

        """
        # This has to be implemented by the subclasses
        raise NotImplementedError

    def clip_to_bounds(self, control_input: dict) -> dict:
        """
        Clips the control inputs to their bounds to avoid numerical errors.

        Args:
            control_input (dict): dictionary of {control input ID: value of control input}

        Returns:
            dict: dictionary of clipped control inputs

        """
        clipped_action = copy(control_input)
        for node_id, node_inputs in control_input.items():
            for input_id, input_value in node_inputs.items():
                clipped_action[node_id][input_id] = np.clip(
                    input_value, self.input_space[node_id][input_id].low, self.input_space[node_id][input_id].high
                )

        return clipped_action

    def _denormalize_input(self, action: OrderedDict) -> OrderedDict:
        """
        Denormalize action to original input space
        Args:
            action: normalized action

        Returns:

        """
        scaled_action = deepcopy(action)
        for node_id, node_action in action.items():
            for el_id, el_action in node_action.items():
                action_low = self.input_space[node_id][el_id].low
                action_high = self.input_space[node_id][el_id].high

                new_action = (action[node_id][el_id] - (-1 * np.ones((len(action_high,))))) / 2 * np.ones(
                    (
                        len(
                            action_high,
                        )
                    )
                ) * (action_high - action_low) + action_low
                scaled_action[node_id][el_id] = new_action

        return scaled_action

    def _index_entities(self):
        """
        Called during init to record the all entity ids.
        We can do this only now because entity ids are assigned when they are added to the pyomo model.
        """
        for node in self.nodes:
            self.node_ids.append(node.id)

    def add_system(self, system: System):
        """
        When adding a system to a controller, the system tree is searched recursively and all controllable entities
        that do not yet have a controller are added to 'nodes'.

        Args:
            system (System): system to be added.

        Returns:
            BaseController: The current controller.

        """
        if system.controller is not None:
            warnings.warn(f"The system already has a global controller with the name {system.controller.get_id()}")

        system.register_controller(self)
        self.nodes.append(system)

        children = system.get_children()
        for child in children:
            if child.controller is None:
                child.register_controller(self)
                self.nodes.append(child)

        return self

    def add_entity(self, entity: ControllableModelEntity):
        """
        Add a controllable entity to the controller. Recursively searches the component tree of the entity and adds all
        individual controllable entities from that tree to 'nodes'.

        Args:
            entity: controllable entity to be added

        Returns:
            BaseController: The current controller

        """
        # first, check whether the node has any children. If it doesn't, it is a single component and we can just assign
        # the controller. If it does, we have to recursively go through all children and check whether they already have
        # a controller!
        if entity.controller is not None:
            warnings.warn(f"Node {entity.id} already has a controller")

        entity.register_controller(self)
        self.nodes.append(entity)

        children = entity.get_children()
        for child in children:
            if child.controller is not None:
                warnings.warn(f"Node {child.id} already has a controller!")
            child.register_controller(self)
            self.nodes.append(child)

        return self

    def get_nodes(self) -> List[ControllableModelEntity]:
        """
        Get controlled nodes.

        Returns:
            List[ControllableModelEntity]: all entities under control

        """
        return self.nodes

    def get_top_level_nodes(self) -> List[ControllableModelEntity]:
        """
        Retrieve the controlled entities at the highest level in the tree.

        Returns:
            List[ControllableModelEntity]: Highest-level entities under control.

        """

        def get_entity_level(entity_id: str) -> int:
            if entity_id.split(".") == [""]:
                return 0  # root node (sys)
            else:
                return len(entity_id.split("."))

        def get_top_level_node(entity_id: str, top_level: int) -> str:
            top_level_node = ".".join(entity_id.split(".")[:top_level])
            return top_level_node

        shortest_node_id = min([get_entity_level(nid) for nid in self.node_ids])
        top_level_nodes = [node for node in self.nodes if get_entity_level(node.id) == shortest_node_id]
        # check that the controlled subsystem does not have a disconnected structure
        if shortest_node_id > 0:
            required_top_level_nodes = [get_top_level_node(nid, shortest_node_id) for nid in self.node_ids]
            top_level_node_ids = [n.id for n in top_level_nodes]
            if not all([nid in top_level_node_ids for nid in required_top_level_nodes]):
                raise ControllerError(
                    self,
                    "Tree of controlled subsystem is disconnected. You have added model entities to the "
                    "controller that are not on the same level.",
                )
        return top_level_nodes

    def get_id(self) -> str:
        """
        Get ID of controller.

        Returns:
            str: controller name

        """
        return self.name

    def get_cost(self, sys_inst: ConcreteModel) -> float:
        """
        Compute control cost for one time step based on 1) cost resulting from solution of optimization problem in Pyomo
        model for the controllable entities assigned to this controller and 2) the cost callback to add additional
        terms.

        Args:
            sys_inst (ConcreteModel): current Pyomo model with solution from optimization

        Returns:
            float: control cost for one time step

        """
        # ToDo: Need to adjust if we ever have an action horizon > 1 time step
        cost_values = [n.get_value(sys_inst, "cost") for n in self.top_level_nodes]
        cost_values = [item for sublist in cost_values for item in sublist]
        ctrl_cost = sum(cost_values)

        if self.cost_callback:
            ctrl_cost += self.cost_callback(ctrl=self, sys_inst=sys_inst)
        return ctrl_cost

    def set_obs_mask(
        self,
        obs_types: List[ElementTypes] = [ElementTypes.DATA, ElementTypes.STATE],
        glob_obs_elements: List[Tuple[Union[Node, list]]] = None,
    ) -> dict:
        """
        Sets the elements observed by the controller.

        Args:
            obs_types (List[ElementTypes]): types of model elements that will be included from all entities controlled
                by this controller.
            glob_obs_elements List[Tuple[Union[Node, list]]]: additional model elements that have to be included
                in the observation (may be from outside the scope of the controller).

        Returns:
            dict: observed model element ids

        """
        for node in self.nodes:
            self.obs_mask[node.id] = [el.name for el in node.model_elements if el.type in obs_types]
        if glob_obs_elements:
            self.obs_mask["global"] = glob_obs_elements

    def get_input_space(self, normalize: bool = False) -> gym.spaces.Dict:
        """
        Derives action space of the controller from the list of its controlled entities.

        Args:
            normalize (bool): whether or not to normalize the action space

        Returns:
            gym.spaces.Dict: action space of each entity that has INPUT model elements

        """
        ctrl_input_space = {}
        for node in self.nodes:
            node_input_space = node.input_space(normalize=normalize)
            if node_input_space is not None:
                ctrl_input_space[node.id] = node_input_space
        ctrl_input_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in ctrl_input_space.items()})
        return ctrl_input_space

    def flatten_obs(self, obs: dict) -> np.ndarray:
        """
        Converts observation dictionary to a numpy array.

        Args:
            obs (dict): dictionary of observed element IDs and their values

        Returns:
            np.ndarray: numpy array of all the observations

        """
        flattened_obs = np.array([])
        for n_id, n_obs in obs.items():
            for el_obs in n_obs.values():
                flattened_obs = np.concatenate((flattened_obs, el_obs))
        return flattened_obs

    def act_array_to_dict(self, action: np.ndarray) -> OrderedDict:
        """
        Converts numpy array of actions to dictionary.

        Args:
            action (np.ndarray): numpy array of actions

        Returns:
            OrderedDict: dictionary of input element IDs and action value for all controlled entities

        """
        dummy_action = self.input_space.sample()
        act_count = 0
        # fill action dictionary with values
        for n_id, n_act in dummy_action.items():
            for el_id, el_act in n_act.items():
                num_act = el_act.shape[0]
                dummy_action[n_id][el_id] = action[act_count : act_count + num_act]
                act_count = act_count + num_act
        return dummy_action

    def detach(self):
        """
        Remove controller from all controlled entities

        Returns:
            None

        """
        for node in self.nodes:
            node.detach_controller()


class OptimalController(BaseController):
    def __init__(
        self,
        name: str,
        cost_callback: Callable = None,
        solver: OptSolver = get_default_solver(),
        control_input_trajectory_length: int = 1,
    ):
        """
        Optimal controller that solves a constrained optimization problem to find the control inputs which minimize
        the cost function of all its controlled entities while satisfying their constraints.

        Args:
            name (str): name of the controller
            cost_callback (Callable, optional): function used within the cost function of the controller
                to compute additional cost terms
            solver (OptSolver, optional): solver for optimization problem
            control_input_trajectory_length (int, optional): number of time steps the controller
                computes control inputs for

        Returns:
            OptimalController
        """
        super().__init__(name=name, cost_callback=cost_callback)
        self.ctrl_type = "oc"  # optimal control
        self.sys_inst = None
        self.model = None
        self.solver = solver

        self.control_input_trajectory_length = control_input_trajectory_length  # only one time step for optimal control

    def reset_history(self) -> None:
        """
        Delete history

        Returns:
            None

        """
        self.history = {}

    def compute_control_input(
        self, obs: OrderedDict = None, input_callback: Callable = None
    ) -> Tuple[OrderedDict, float]:
        """
        Main functionality of the controller: computes the control inputs which minimize the objective function of the
        controlled entities while satisfying their constraints.

        Args:
            obs (OrderedDict): not needed her
            input_callback (Callable): not needed here

        Returns:
            Tuple: tuple containing:
                - action (OrderedDict)
                - safety penalty (float) (not needed hear, only for RL controllers).

        """
        # get current system pyomo instance
        self.sys_inst = self.nodes[0].instance
        mdl = ConcreteModel()

        for node in self.top_level_nodes:
            if isinstance(node, System):
                mdl = self.sys_inst.clone()
            else:
                setattr(mdl, node.id.split(".")[-1], node.get_self_as_pyomo_block(self.sys_inst).clone())

        def obj_fcn_mpc(model):
            return quicksum(
                [n.cost_fcn(model, t) for t in range(len(self.sys_inst.t) - 1) for n in self.top_level_nodes]
            )

        # we want to delete existing objectives from the original system and define our own for the controller
        for objective in mdl.component_objects(pyo.Objective, descend_into=True):
            mdl.del_component(objective)
        mdl.control_obj1 = Objective(expr=obj_fcn_mpc)

        self.model = mdl

        results = self.solver.solve(self.model, warmstart=True)
        self.model.solutions.store_to(results)
        # catch infeasible solution
        if results.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            raise EntityError(self.model, "Cannot find an input satisfying all constraints")

        node_actions = {}
        for node in self.nodes:
            node_action = node.get_inputs(self.model)
            if node_action is not None:
                node_actions[node.id] = {
                    el_id: np.array(el_action[: self.control_input_trajectory_length])
                    for el_id, el_action in node_action.items()  # TODO: We can generalize this to multiple time steps
                }  # only get first input element to apply to system

        # clip actions to bounds to account for numerical errors
        node_actions = self.clip_to_bounds(node_actions)
        # return action as OrderedDict to make it compatible with Gym
        action = OrderedDict(node_actions)
        safety_penalty = 0.0  # no safety correction necessary for optimal control
        return action, safety_penalty


class RLBaseController(BaseController):
    def __init__(
        self,
        name: str,
        train: bool = True,
        device: str = "cpu",
        safety_layer=None,
        cost_callback: Callable = None,
        pretrained_policy_path: str = None,
    ):
        """
        Base class for reinforcement learning (RL) controllers. Requires a safety layer to ensure constraint
        satisfaction. For the RL controller, there are two different modes: training and deployment. In training mode,
        the action is obtained through a callback from the Gym environment. During deployment, the action is computed
        by propagating the observation through the trained neural network policy. Saving and loading this policy and
        computing the action in deployment mode depend on the RL algorithm and therefore have to be implemented in
        the respective subclasses.

        Args:
            name (str): name of the controller
            train (bool): whether the controller is in training mode
            device (str): whether to use 'cpu' or 'cuda' (GPU)
            safety_layer (BaseSafetyLayer): safety layer instance
            cost_callback (Callable): function used within the cost function of the controller to compute additional \
            cost terms
            pretrained_policy_path (str): directory with stored policy parameters of an existing policy

        Returns:
            RLBaseController

        """
        super().__init__(name=name, cost_callback=cost_callback)
        self.ctrl_type = "rl"  # Reinforcement Learning
        self.device = device
        self.train = train
        self.policy = None
        self.safety_layer = safety_layer
        self.load_path = pretrained_policy_path
        self.train_history = {}
        self.deployment_history = []
        self.denormalize_inputs = False

    def initialize(self):
        """
        Initial set-up of controller and safety layer
        """
        super().initialize()
        self.safety_layer.initialize(nodes=self.nodes, top_level_nodes=self.top_level_nodes)

    def reset_history(self):
        """
        Delete history

        Returns:
            None

        """
        if not self.train:
            self.deployment_history.append(self.history)
        for key in self.history.keys():
            self.history[key] = []

    def set_normalize_inputs(self, normalize_inputs: bool):
        """

        Args:
            normalize_inputs (bool): Whether actions sampled from RL policy are normalized. Needed during deployment.

        Returns:

        """
        self.denormalize_inputs = normalize_inputs

    def update_history(self, info_dict: dict):
        """
        Insert new data into training history.

        Args:
            info_dict (dict): dictionary of information that should be written into history

        Returns:
            None

        """
        time = self.top_level_nodes[0].t
        for key, value in info_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((time, value))

    def save(self, policy: BasePolicy, save_path: str = "./saved_models/test_model"):
        # has to be implemented by subclasses
        raise NotImplementedError

    def load(self, policy_class, env, config, policy_kwargs=None):
        # has to be implemented by subclasses
        raise NotImplementedError

    def compute_control_input(
        self, obs: OrderedDict = None, input_callback: Callable = None
    ) -> Tuple[OrderedDict, float]:
        """
        In training mode, the control input is computed within the training algorithm and passed to this controller
        through a callback. It is then verified by the safety layer and adjusted (minimally) in case it violates any
        constraints. In deployment mode, the control input is computed from the stored neural network policy and
        verified by the safety layer.

        Args:
            obs (OrderedDict): observation at current time point
            input_callback (Callable): only needed in training mode - retrieves action selected within training \
                algorithm

        Returns:
            Tuple: tuple containing
                - action (OrderedDict)
                - penalty for action adjustment performed by safety layer (float).

        """
        if self.train:
            if input_callback is None:
                raise ValueError("Need to provide an action callback in training mode!")
            action = input_callback(self.name)
            verified_action, action_corrected, safety_penalty = self.safety_layer.compute_safe_action(action)
            # clip actions to bounds to account for numerical errors
            verified_action = self.clip_to_bounds(verified_action)
            self.update_history({"safety_penalty": safety_penalty, "action_corrected": action_corrected})
        else:
            # ToDo: generalize this?
            obs = self.flatten_obs(obs)
            action = self.predict_action(obs)
            action = self.act_array_to_dict(action)
            if self.denormalize_inputs:
                action = self._denormalize_input(action)
            verified_action, action_corrected, safety_penalty = self.safety_layer.compute_safe_action(action)
            # clip actions to bounds to account for numerical errors
            verified_action = self.clip_to_bounds(verified_action)
            self.update_history({"safety_penalty": safety_penalty, "action_corrected": action_corrected})
        return verified_action, safety_penalty

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Actual forward pass of the current policy. Needs to be implemented by subclasses.

        Args:
            obs (np.ndarray): observation at current time step (has to be numpy array, not dictionary, since a \
            dictionary cannot be processed by the neural network.)
            deterministic (bool): Whether to use a deterministic action selection algorithm

        Returns:
            np.ndarray: control action

        """

        raise NotImplementedError

    def set_mode(self, mode: str):
        """
        Set mode to training (True) or deployment.

        Args:
            mode (str): 'train', 'test'

        Returns:
            None

        """
        if mode == "train":
            self.train = True
        else:
            self.train = False


class RLControllerSB3(RLBaseController):
    """
    Controller class for RL agents trained with algorithms from the StableBaselines repository
    (https://stable-baselines3.readthedocs.io/). Single-agent RL algorithms only!

    """

    def save(self, policy: BasePolicy, save_path: str = "./saved_models/test_model"):
        """
        Save neural network policy parameters and structure.

        Args:
            policy (BasePolicy): policy trained with algorithm from StableBaselines
            save_path (str): where to save the policy parameters

        Returns:
            None

        """
        # has to be implemented by subclasses
        self.policy = policy
        self.policy.save(save_path)

    def load(self, env, config: dict, policy_kwargs: dict = None):
        """
        Loading a pre-trained policy from a directory.

        Args:
            env (ControlEnv): The gym environment constructed from the power system the RL algorithm interacts with. \
            Required to construct the neural network policy because it determines the number of inputs (observations) \
            and outputs (actions) of the network.
            config (dict): Configuration for the StableBaselines policy class (also constructs training buffers etc., \
            which is why this also contains algorithm parameters).
            policy_kwargs (dict): Configuration of the actual neural networks of the policy (e.g., number of neurons \
            in the hidden layers of the actor and critic network of an ActorCriticPolicy). Depends on policy type. \
            Consult the StableBaselines documentation (https://stable-baselines3.readthedocs.io/en/master/) for more \
            information.

        Returns:
            None

        """
        # check that a path from which to load the policy has been instantiated
        if not self.load_path:
            raise ValueError(
                "No load path for pre-trained policy! Needs to be handed over in constructor (pretrained_policy_path)"
            )
        # has to be implemented by subclasses
        TrainAlg = config["algorithm"]
        self.policy = TrainAlg(
            env=env,
            policy=config["policy"],
            device=config["device"],
            n_steps=config["n_steps"],
            normalize_advantage=False,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            policy_kwargs=policy_kwargs,
        )
        self.policy = self.policy.load(self.load_path)
        # ugly hack to overwrite the seed in in self.policy.load (which will be done with the seed used during training)
        set_random_seed(seed=config["seed"])

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Compute the control action based on a given observation by propagating this observation through the policy
        network.

        Args:
            obs (np.ndarray): observation at current time step (has to be numpy array, not dictionary, since a \
            dictionary cannot be processed by the neural network.)
            deterministic (bool): Whether to use a deterministic action selection algorithm

        Returns:
            np.ndarray: control action
        """
        # actual forward pass of the current policy
        action, _ = self.policy.predict(obs, deterministic=deterministic)
        return action


class RLControllerMA(RLBaseController):
    """
    Controller class for RL agents trained with MAPPO algorithm from on-policy repository
    (https://github.com/marlbenchmark/on-policy/blob/main/README.md). Multi-agent RL algorithms only!

    """

    def __init__(
        self,
        name: str,
        train: bool = True,
        device: str = "cpu",
        safety_layer=None,
        cost_callback: Callable = None,
        pretrained_policy_path: str = None,
    ):
        super().__init__(
            name=name,
            cost_callback=cost_callback,
            train=train,
            device=device,
            safety_layer=safety_layer,
            pretrained_policy_path=pretrained_policy_path,
        )
        # flattened observation and action space (needed for loading policies)
        self.flattened_obs_space = None
        self.flattened_input_space = None
        self._last_rnn_state = None  # used for predictions if training recursive policies
        self.policy_kwargs = None

    def save(self, policy, save_path: str = "./saved_models/test_model"):
        """
        Save neural network policy parameters and structure.

        Args:
            policy: trained policy
            save_path (str): where to save the policy parameters

        Returns:
            None

        """
        th.save(policy.state_dict(), save_path)

    def load(self, env, config: dict, policy_kwargs: dict = None):
        """
        Loading a pre-trained policy from a directory.

        Args:
            env (ControlEnv): The gym environment constructed from the power system the RL algorithm interacts with. \
            Required to construct the neural network policy because it determines the number of inputs (observations) \
            and outputs (actions) of the network.
            config: Configuration for the policy class (also constructs training buffers etc., which is why this also \
            contains algorithm parameters).
            policy_kwargs: Not used here

        Returns:
            None

        """
        from commonpower.control.controller_utils import ArgsWrapper

        config = ArgsWrapper(config)
        self.policy_kwargs = config
        self._check_alg_config()
        share_observation_space = (
            env.share_observation_space[0] if self.policy_kwargs.use_centralized_V else self.flattened_obs_space
        )

        # policy network
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        po = Policy(
            self.policy_kwargs,
            self.flattened_obs_space,
            share_observation_space,
            self.flattened_input_space,
            device=self.device,
        )

        policy_actor_state_dict = th.load(self.load_path + "/actor_agent" + ".pt")
        po.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = th.load(self.load_path + "/critic_agent" + ".pt")
        po.critic.load_state_dict(policy_critic_state_dict)
        self.policy = po
        self._last_rnn_state = np.zeros((1, config.recurrent_N, config.hidden_size))

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Compute the control action based on a given observation by propagating this observation through the
        policy network.

        Args:
            obs (np.ndarray): observation at current time step (has to be numpy array, not dictionary, since a \
            dictionary cannot be processed by the neural network.)
            deterministic (bool): Whether to use a deterministic action selection algorithm

        Returns:
            np.ndarray: control action
        """
        # actual forward pass of the current policy
        obs = obs.reshape((1, -1))  # make compatible with on-policy-repository (rMAPPOPolicy)
        # we do not need masks as our agents always terminate at the same time
        dummy_mask = np.ones((1, 1), dtype=np.float32)
        # in case we use recurrent policies, we need to store the hidden state of the recurrent NN
        action, rnn_state = self.policy.act(
            obs, rnn_states_actor=self._last_rnn_state, masks=dummy_mask, deterministic=True
        )
        action = action.detach().cpu().numpy()
        action = action[0]
        self._last_rnn_state = rnn_state.detach().cpu().numpy()

        return action

    def _check_alg_config(self):
        """
        Sanity check for the algorithm configuration: If we use any variant of MAPPO,
        we want a shared observation space which means that use_centralized_V has to be true.
        If we use a recurrent policy (RMAPPO), the respective arguments have to be true.

        Returns:
            None
        """
        if self.policy_kwargs.algorithm_name == "rmappo":
            print("You are choosing to use RMAPPO, we set use_recurrent_policy to be True")
            self.policy_kwargs.use_recurrent_policy = True
            self.policy_kwargs.use_naive_recurrent_policy = False
            self.policy_kwargs.use_centralized_V = True
        elif self.policy_kwargs.algorithm_name == "mappo":
            print("You are choosing to use MAPPO, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
            self.policy_kwargs.use_recurrent_policy = False
            self.policy_kwargs.use_naive_recurrent_policy = False
            self.policy_kwargs.use_centralized_V = True
        elif self.policy_kwargs.algorithm_name == "ippo":
            print("You are choosing to use IPPO, we set use_centralized_V to be False")
            self.policy_kwargs.use_centralized_V = False
        else:
            raise NotImplementedError
