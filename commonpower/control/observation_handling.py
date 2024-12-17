from collections import deque
from typing import List, Tuple

import gymnasium as gym

from commonpower.modeling.base import ElementTypes


class ObservationHandler:
    def __init__(
        self,
        num_forecasts: int = 0,
        observation_types: List[ElementTypes] = [ElementTypes.DATA, ElementTypes.STATE],
        num_past_observations: int = 0,
    ):
        """
        Base class for handling observations of RL agents. Extracts information from the underlying controlled
        system and processes it for RL observations.

        Args:
            num_forecasts (int): The number of forecasted steps to include in the observations. Default is 0.
            observation_types (List[ElementTypes]): The types of elements to include in the observations.
                Default includes DATA and STATE.
            num_past_observations (int): The number of past observations to stack for the agent. Default is 0.
        """
        self.obs_types = observation_types
        self.n_forecasts = num_forecasts
        self.obs_mask = ({}, 1)

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
        self.obs_mask = (elements_obs_mask, self.n_forecasts)

    def get_obs_mask(self) -> Tuple[dict, int]:
        """
        Retrieves the current observation mask.

        Returns:
            Tuple[dict, int]: A tuple containing the observation mask (mapping of node IDs to observed elements)
                and the number of forecasted steps.
        """
        return self.obs_mask

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
        ctrl_obs_space = {}
        for node in nodes_controller:
            node_obs_space = node.observation_space(self.obs_mask)
            if node_obs_space is not None:
                ctrl_obs_space[node.id] = node_obs_space
        # transform to gymnasium spaces.Dict
        ctrl_obs_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in ctrl_obs_space.items()})
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
