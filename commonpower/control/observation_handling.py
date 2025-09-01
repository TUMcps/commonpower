from collections import OrderedDict, deque
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from commonpower.modeling.base import ElementTypes


class Observer:
    def get_node_obs(self, node, obs_mask: Tuple[dict, int]) -> dict:
        """
        Get observations for one node within the system based on the model items within the observation mask.

        Args:
            node (Node): Node for which we want to get the observation
            obs_mask Tuple(dict, int): tuple with a) dictionary containing the IDs of model elements which should
            be observed, b) number of forecast steps that should be included in observation

        Returns:
            dict: dict of observed values as {element ID: value}

        """
        obs = OrderedDict()
        observed_model_elements, n_forecasts = obs_mask
        node_elements = getattr(node, "model_elements")
        node_id = getattr(node, "id")
        sys_inst = getattr(node, "instance")
        for el in node_elements:
            if el.name in observed_model_elements[node_id]:
                # for states, we only want to get the current value
                if el.type == ElementTypes.STATE:
                    obs[el.name] = np.array(node.get_value(sys_inst, el.name))[0].reshape((1,))
                else:
                    obs[el.name] = np.array(node.get_value(sys_inst, el.name)[0:n_forecasts])

        if len(obs) == 0:
            return None
        else:
            return obs

    def global_obs(self, obs_mask: Tuple[dict, int]) -> Dict:
        """
        Gets values of model elements in global_obs_mask which should be added to the observation of a controller

        Args:
            obs_mask (Tuple[dict, int]): tuple with a) dictionary containing the IDs of model elements which should
            be observed, b) number of forecast steps that should be included in observation

        Returns:
            dict: dictionary of {entity_id: entity_observation} of all model elements in obs_mask

        """
        obs = OrderedDict()
        observed_model_elements, n_forecasts = obs_mask
        global_obs_mask = observed_model_elements["global"]
        # rewrite global obs_mask to local obs_maks:
        local_obs_mask = {}
        nodes = [item[0] for item in global_obs_mask]
        obs_el = [item[1] for item in global_obs_mask]
        for count, node in enumerate(nodes):
            local_obs_mask[node.id] = obs_el[count]
        for node in nodes:
            node_obs = self.get_node_obs(node, (local_obs_mask, n_forecasts))
            obs[node.id] = node_obs
        return obs

    def observe(self, controllers: dict) -> dict:
        """
        Get observations for all controllers within the system.
        Args:
            controllers (dict): dictionary {ctrl_id: controller}

        Returns:
            dict: dictionary of {controller_id: controller_observation}

        """
        from commonpower.core import System

        obs = OrderedDict()
        for ctrl_id, ctrl in controllers.items():
            ctrl_obs = OrderedDict()
            nodes = ctrl.get_nodes()
            nodes = [n for n in nodes if not isinstance(n, System)]
            for node in nodes:
                node_obs = self.get_node_obs(node, ctrl.obs_mask)
                if node_obs is not None:
                    ctrl_obs[node.id] = node_obs
            if "global" in ctrl.obs_mask[0].keys():
                ctrl_obs["global"] = self.global_obs(ctrl.obs_mask)
            obs[ctrl_id] = ctrl_obs
        obs_info = {}
        return obs, obs_info


class ObservationHandler:
    def __init__(
        self,
        num_forecasts: int = 1,
        observation_types: List[ElementTypes] = [ElementTypes.DATA, ElementTypes.STATE],
        num_past_observations: int = 0,
        global_obs_elements: List[Tuple] = None,
    ):
        """
        Base class for handling observations of RL agents. Extracts information from the underlying controlled
        system and processes it for RL observations.

        Args:
            num_forecasts (int): The number of forecasted steps to include in the observations. Default is 0.
            observation_types (List[ElementTypes]): The types of elements to include in the observations.
                Default includes DATA and STATE.
            num_past_observations (int): The number of past observations to stack for the agent. Default is 0.
            global_obs_elements(List[Tuple[Union[Node, list]]]): additional model elements (can also be from
            outside the controlled entities) that should be observed.
        """
        self.obs_types = observation_types
        self.n_forecasts = num_forecasts
        self.obs_mask = ({}, 1)
        self.global_obs_elements = global_obs_elements

        self.n_past_obs = num_past_observations
        self.past_observations = deque(
            maxlen=self.n_past_obs + 1
        )  # we want to store the current obs and the n_past_obs past observations

    def reset(self) -> None:
        """
        Resets the ObservationHandler by clearing all stored past observations.

        Returns:
            None
        """
        self.past_observations.clear()

    def set_obs_mask(self, nodes_controller: List[None]) -> None:
        """
        Sets the observation mask for the elements observed by the RL controller.

        Args:
            nodes_controller (List[None]): A list of nodes representing the system components controlled by
            the RL agent.

        Returns:
            None
        """
        elements_obs_mask = {}
        for node in nodes_controller:
            elements_obs_mask[node.id] = [el.name for el in node.model_elements if el.type in self.obs_types]
        if self.global_obs_elements:
            elements_obs_mask["global"] = self.global_obs_elements
        self.obs_mask = (elements_obs_mask, self.n_forecasts)

    def get_obs_mask(self) -> Tuple[dict, int]:
        """
        Retrieves the current observation mask.

        Returns:
            Tuple[dict, int]: A tuple containing the observation mask (mapping of node IDs to observed elements)
                and the number of forecasted steps.
        """
        return self.obs_mask

    def get_ctrl_observation_space(self, nodes_controller: List[None]) -> gym.spaces.Dict:
        """
        Builds and retrieves the observation space for the RL controller based on the controlled nodes.

        Args:
            nodes_controller (List[None]): A list of nodes controlled by the RL agent. Each node provides its
                observation space using the `observation_space` method.

        Returns:
            gym.spaces.Dict: A dictionary representing the observation space for the controller, including stacking
                for past observations if configured.
        """
        ctrl_obs_space = {}
        for node in nodes_controller:
            node_obs_space = self.get_node_obs_space(node, self.obs_mask)
            if node_obs_space is not None:
                ctrl_obs_space[node.id] = node_obs_space
        if "global" in self.obs_mask[0].keys():
            ctrl_obs_space["global"] = self.global_observation_space()
        # transform to gymnasium spaces.Dict
        ctrl_obs_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in ctrl_obs_space.items()})
        return ctrl_obs_space

    def get_node_obs_space(self, node, obs_mask: Tuple[dict, int]):
        """
        Determines the observation space of an entity based on the observation mask by retrieving
        the bounds of the model elements listed in the mask

        Args:
            node (Node): Node for which we want to get the observation space
            obs_mask Tuple(dict, int): tuple with a) dictionary containing the IDs of model elements which should
            be observed, b) number of forecast steps that should be included in observation

        Returns:
            None/gym.spaces.Dict: None if the node has no elements that should be observed, else a dictionary as in
            {model element ID: box observation space}

        """
        # ToDo: check type of variables/data --> if they are binary, we cannot use box spaces?
        # for now all model elements with type DATA and STATE are observations
        observed_model_elements, n_forecasts = obs_mask
        node_elements = getattr(node, "model_elements")
        node_id = getattr(node, "id")
        sys_inst = getattr(node, "instance")
        obs = [e for e in node_elements if e.name in observed_model_elements[node_id]]
        lower = {}
        upper = {}
        for e in obs:
            pyomo_el = node.get_pyomo_element(e.name, sys_inst)
            # for states, we only want to observe the first element
            if e.type == ElementTypes.STATE:
                if e.bounds is not None:
                    lower[e.name] = e.bounds[0]
                    upper[e.name] = e.bounds[1]
                else:
                    lower[e.name] = -np.inf
                    upper[e.name] = np.inf
            else:
                if e.bounds is not None:
                    lower[e.name] = (
                        [e.bounds[0] for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else e.bounds[0]
                    )
                    upper[e.name] = (
                        [e.bounds[1] for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else e.bounds[1]
                    )
                else:
                    lower[e.name] = [-np.inf for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else -np.inf
                    upper[e.name] = [np.inf for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else np.inf
                # limit length of observation space to num of desired forecasts
                lower[e.name] = lower[e.name][0:n_forecasts]
                upper[e.name] = upper[e.name][0:n_forecasts]

        if lower:
            obs_space = gym.spaces.Dict(
                {
                    el.name: gym.spaces.Box(
                        low=np.array([lower[el.name]]).reshape((-1,)),
                        high=np.array([upper[el.name]]).reshape((-1,)),
                        dtype=np.float64,
                    )
                    for el in obs
                }
            )

            return obs_space
        else:
            return None

    def global_observation_space(self):
        """
        Translates a list of either model entities or strings (identifiers of model entities) to the respective
        observation space corresponding to the bounds of the observed quantities.

        Returns:
            gym.spaces.Dict: observation space for the quantities in the obs_mask

        """
        global_obs_space = {}
        observed_model_elements, n_forecasts = self.obs_mask
        global_obs_mask = observed_model_elements["global"]
        # rewrite global obs_mask to local obs_maks:
        local_obs_mask = {}
        nodes = [item[0] for item in global_obs_mask]
        obs_el = [item[1] for item in global_obs_mask]
        for count, node in enumerate(nodes):
            local_obs_mask[node.id] = obs_el[count]
        for node in nodes:
            node_obs_space = self.get_node_obs_space(node, (local_obs_mask, n_forecasts))
            global_obs_space[node.id] = node_obs_space
        global_obs_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in global_obs_space.items()})
        return global_obs_space

    def get_observation_space(self, nodes_controller: List[None]) -> gym.spaces.Dict:
        """
        Builds and retrieves the observation space for the RL controller based on the controlled nodes.

        Args:
            nodes_controller (List[None]): A list of nodes controlled by the RL agent. Each node provides its
                observation space using the `observation_space` method.

        Returns:
            gym.spaces.Dict: A dictionary representing the observation space for the controller, including stacking
                for past observations if configured.
        """
        ctrl_obs_space = self.get_ctrl_observation_space(nodes_controller=nodes_controller)
        # Extend if we have stacked observations
        if self.n_past_obs > 0:
            obs_time_indices = [-i for i in range(self.n_past_obs + 1)]
            stacked_ctrl_obs_space = gym.spaces.Dict(
                {str(obs_time_indices[idx]): ctrl_obs_space for idx in range(self.n_past_obs + 1)}
            )
            ctrl_obs_space = stacked_ctrl_obs_space
        return ctrl_obs_space

    def get_adjusted_obs(self, system_obs: dict) -> dict:
        """
        Adjusts the system observations by stacking past observations as needed.

        Args:
            system_obs (dict): A dictionary containing the current system observations.

        Returns:
            dict: A dictionary containing the adjusted observations, including stacked past observations if
                configured.
        """
        # save observation to stack
        ctrl_obs = system_obs
        if len(self.past_observations) == 0:
            # when we do not have past observations yet, we stack the current observation
            for n_stacks in range(self.n_past_obs):
                self.past_observations.appendleft(system_obs)
        self.past_observations.appendleft(system_obs)
        if self.n_past_obs > 0:
            obs_time_indices = [-i for i in range(self.n_past_obs + 1)]
            ctrl_obs = {str(obs_time_indices[idx]): obs for idx, obs in enumerate(self.past_observations)}
        return ctrl_obs
