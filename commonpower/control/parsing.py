from collections import OrderedDict
from typing import Union

import gymnasium as gym
import numpy as np

from commonpower.control.environments import ControlEnv


class ParserFactory:
    def __init__(self, env: Union[gym.Wrapper, ControlEnv]):
        """
        Class for handling conversions of observations and actions during deployment of RL controllers.
        Args:
            env (Union[gym.Wrapper, ControlEnv]): environment (potentially wrapped)

        """
        self.env = env
        self.unwrapped_env = getattr(env, "unwrapped")
        self.env_obs_space = getattr(env, "observation_space")
        self.env_action_space = getattr(env, "action_space")
        if not self.env_obs_space.__class__ == self.env_action_space.__class__:
            raise TypeError("The type of the observation space does not match the type of the action space.")

    def get_parser(self):
        """
        Implements switching logic for deciding which parser to use.
        """
        if isinstance(self.env_obs_space, list):
            sample_obs = [env_obs_space.sample() for env_obs_space in self.env_obs_space]
            sample_act = [env_act_space.sample() for env_act_space in self.env_action_space]
        else:
            sample_obs = self.env_obs_space.sample()
            sample_act = self.env_action_space.sample()

        if isinstance(sample_obs, np.ndarray):
            # single-agent
            parser = ArrayParser(env=self.env, unwrapped_env=self.unwrapped_env)
        elif isinstance(sample_obs, list):
            # multi-agent (MAPPO)
            parser = ListParser(env=self.env, unwrapped_env=self.unwrapped_env, sample_action=sample_act)
        elif isinstance(sample_obs, (dict, OrderedDict)):
            parser = DictParser(env=self.env, unwrapped_env=self.unwrapped_env)
        else:
            raise NotImplementedError("The current space is not supported!")

        return parser


class BaseParser:
    def __init__(self, env: Union[gym.Wrapper, ControlEnv], unwrapped_env: ControlEnv):
        """
        Base class for parsers.
        Args:
            env (Union[gym.Wrapper, ControlEnv]): environment (potentially wrapped)
            unwrapped_env (ControlEnv): unwrapped environment
        """
        self.env = env
        self.unwrapped_env = unwrapped_env

    def parse_obs(self, original_obs: Union[np.ndarray, list, dict, OrderedDict]) -> OrderedDict:
        """
        Transforms the observation returned by the environment to the form {ctrl_id: np.ndarray} used in the
        DeploymentRunner _run() function.
        Args:
            original_obs (Union[np.ndarray, list, dict, OrderedDict]): observation from environment
        Returns:
            (OrderedDict): transformed observation as {ctrl_id: np.ndarray}
        """
        raise NotImplementedError

    def parse_action(self, original_action: OrderedDict) -> Union[np.ndarray, list, dict, OrderedDict]:
        """
        Transforms the action provided by the DeploymentRunner _run() function to the format required by the underlying
        environment
        Args:
            original_action (OrderedDict): action provided by DeploymentRunner
        Returns:
            (Union[np.ndarray, list, dict, OrderedDict]): transformed action
        """
        raise NotImplementedError


class ArrayParser(BaseParser):
    def parse_obs(self, original_obs: Union[np.ndarray, list, dict, OrderedDict]) -> OrderedDict:
        """
        Transforms the observation returned by the environment to the form {ctrl_id: np.ndarray} used in the
        DeploymentRunner _run() function.
        Args:
            original_obs (np.ndarray): observation from environment
        Returns:
            (OrderedDict): transformed observation as {ctrl_id: np.ndarray}
        """
        transformed_obs = OrderedDict()
        transformed_obs[getattr(self.env, "ctrl_id")] = original_obs
        return transformed_obs

    def parse_action(self, original_action: OrderedDict) -> Union[np.ndarray, list, dict, OrderedDict]:
        """
        Transforms the action provided by the DeploymentRunner _run() function to the format required by the underlying
        environment
        Args:
            original_action (OrderedDict): action provided by DeploymentRunner
        Returns:
            (np.ndarray): transformed action
        """
        ctrl_action = []
        ctrl_action_dict = original_action[next(iter(original_action))]
        for node_action in ctrl_action_dict.values():
            for el_action in node_action.values():
                ctrl_action.append(el_action)
        return np.array(ctrl_action).reshape((-1,))


class ListParser(BaseParser):
    def __init__(self, env, unwrapped_env, sample_action):
        super().__init__(env=env, unwrapped_env=unwrapped_env)
        self.sample_action = sample_action

    def parse_obs(self, original_obs: Union[np.ndarray, list, dict, OrderedDict]) -> OrderedDict:
        """
        Transforms the observation returned by the environment to the form {ctrl_id: np.ndarray} used in the
        DeploymentRunner _run() function.
        Args:
            original_obs (list): observation from environment
        Returns:
            (OrderedDict): transformed observation as {ctrl_id: np.ndarray}
        """
        transformed_obs = OrderedDict()
        controllers = getattr(self.unwrapped_env, "controllers")  # all RL controllers
        for ctrl_id in controllers.keys():
            ctrl_idx = list(controllers.keys()).index(ctrl_id)
            transformed_obs[ctrl_id] = original_obs[ctrl_idx]
        return transformed_obs

    def parse_action(self, original_action: OrderedDict) -> Union[np.ndarray, list, dict, OrderedDict]:
        """
        Transforms the action provided by the DeploymentRunner _run() function to the format required by the underlying
        environment
        Args:
            original_action (OrderedDict): action provided by DeploymentRunner
        Returns:
            (list): transformed action
        """
        all_agents_actions = np.zeros(np.array(self.sample_action).shape)
        for agent_idx, agent_action in enumerate(original_action.values()):
            action_idx = 0
            for node_action in agent_action.values():
                for element_action in node_action.values():
                    all_agents_actions[agent_idx][action_idx] = element_action[0]
        all_agents_actions_list = list(all_agents_actions)
        all_agents_actions_list = [agent_action.reshape((1, -1)) for agent_action in all_agents_actions_list]
        return all_agents_actions_list


class DictParser(BaseParser):
    def parse_obs(self, original_obs: Union[np.ndarray, list, dict, OrderedDict]) -> OrderedDict:
        """
        Transforms the observation returned by the environment to the form {ctrl_id: np.ndarray} used in the
        DeploymentRunner _run() function.
        Args:
            original_obs (OrderedDict): observation from environment
        Returns:
            (OrderedDict): transformed observation as {ctrl_id: np.ndarray}
        """
        return original_obs

    def parse_action(self, original_action: OrderedDict) -> Union[np.ndarray, list, dict, OrderedDict]:
        """
        Transforms the action provided by the DeploymentRunner _run() function to the format required by the underlying
        environment
        Args:
            original_action (OrderedDict): action provided by DeploymentRunner
        Returns:
            (Union[dict, OrderedDict]): transformed action
        """
        return original_action
