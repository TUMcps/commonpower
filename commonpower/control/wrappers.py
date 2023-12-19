"""
Wrappers to adjust API in environments.py to different RL training algorithms.
"""
from collections import deque
from typing import List, Tuple

import gymnasium as gym
import numpy as np


def ctrl_dict_to_list(input_dict: dict) -> list:
    """
    Transforms the orginal dict of the controller assignments to a list of lists.

    Args:
        input_dict (dict): dictionary {agent_id: value}

    Returns:
        list: list of entries within the dict
    """
    output_list = [value for value in input_dict.values()]

    return output_list


def list_to_ctrl_dict(input_list: list, original_keys: dict) -> dict:
    """
    Reverses the transform_to_ordered_dict_keys function.

    Args:
        input_list (list): list of control actions for each agent
        original_keys (dict): nested dictionary of original action keys for each agent
        as {agent_id: {node_id: list[element_ids]}}

    Returns:
        dict: original dictionary mapping {original_key: value}

    """
    output_dict = {}
    agent_count = 0
    for agent_id, agent_action_keys in original_keys.items():
        agent_input_count = 0
        agent_output_dict = {}
        for node_id, node_action_keys in agent_action_keys.items():
            num_node_inputs = len(node_action_keys)
            agent_output_dict[node_id] = {
                node_action_keys[i]: np.array([input_list[agent_count][0, i + agent_input_count]])
                for i in range(num_node_inputs)
            }
            agent_input_count = agent_input_count + num_node_inputs
        agent_count = agent_count + 1
        output_dict[agent_id] = agent_output_dict
    # output_dict = {original_keys[i]: value for i, value in enumerate(input_list)}
    return output_dict


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Wrapper to standardize ControlEnv to the API for single-agent RL training with any RL algorithm from the
        StableBaselines 3 repository.

        Args:
            env (ControlEnv): power system environment with multi-agent API

        Returns:
             SingleAgentWrapper

        """
        super().__init__(env)
        self.env = env
        if len(self.controllers) > 1:
            raise ValueError("SingleAgentWrapper cannot handle more than 1 agent")
        self.ctrl_id = list(self.env.controllers.keys())[0]
        # training history
        self.train_history = {}
        self.episode_history = deque(maxlen=100)

        # transform observation and action space from dictionary to box
        ctrl_obs_space = self.env.observation_space[self.ctrl_id]
        obs_low = np.array([])
        obs_high = np.array([])
        for n_id, n_obs_space in ctrl_obs_space.items():
            for el in n_obs_space.values():
                obs_low = np.concatenate((obs_low, el.low))
                obs_high = np.concatenate((obs_high, el.high))
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        ctrl_act_space = self.env.action_space[self.ctrl_id]
        act_low = np.array([])
        act_high = np.array([])
        for n_id, n_act_space in ctrl_act_space.items():
            for el in n_act_space.values():
                act_low = np.concatenate((act_low, el.low))
                act_high = np.concatenate((act_high, el.high))
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment

        Args:
            seed: seed for the random number generator
            options: not needed here

        Returns:
            None

        """
        obs, obs_info = self.env.reset(seed=seed, options=options)
        # unpack observation
        obs = self._unpack_obs(obs)
        return obs, obs_info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step function with the single-agent API (takes numpy array action and outputs numpy array observation)

        Args:
            action (np.ndarray): action selected by the RL policy

        Returns:
            Tuple: tuple containing:
                - single-agent observation (np.ndarray)
                - single-agent reward (float)
                - whether the environment is terminated (bool)
                - whether environment is truncated. In our case, the same as terminated (bool)
                - additional information (dict)

        """
        dummy_action = self.env.action_space.sample()
        act_count = 0
        # fill action dictionary with values
        for n_id, n_act in dummy_action[self.ctrl_id].items():
            for el_id, el_act in n_act.items():
                num_act = el_act.shape[0]
                dummy_action[self.ctrl_id][n_id][el_id] = action[act_count : act_count + num_act]
                act_count = act_count + num_act

        obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        reward = reward[self.ctrl_id]
        obs = self._unpack_obs(obs)
        if terminated:
            self.train_history = self.env.train_history[self.ctrl_id]
            self.episode_history = self.env.episode_history[self.ctrl_id]
        return obs, reward, terminated, truncated, info

    def _unpack_obs(self, obs: dict) -> np.ndarray:
        """
        Convert dictionary of {agent_id: observation_dict} to flattened observation array.

        Args:
            obs (dict): observation dictionary {agent_id: observation_dict}

        Returns:
            np.ndarray: flat array of observations

        """
        ctrl_obs = obs[self.ctrl_id]
        new_obs = np.array([])
        for n_id, n_obs in ctrl_obs.items():
            for el_obs in n_obs.values():
                new_obs = np.concatenate((new_obs, el_obs))
        return new_obs


class MultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Wrapper to standardize ControlEnv to the API for MAPPO/IPPO implementation of the on-policy repository
        (https://github.com/marlbenchmark/on-policy/tree/main/onpolicy). NOTE: We use our own fork of this repository,
        see the Readme file.

        Args:
            env (ControlEnv): power system environment with multi-agent API

        Returns:
             MultiAgentWrapper

        """
        super().__init__(env)
        self.env = env
        self.n_agents = len(self.controllers)
        # training history
        self.train_history = {}
        self.episode_history = {}
        # the MAPPO/IPPO implementation expects the action/observation space as a list of lists
        self.action_space, self.original_action_keys = self.act_space_dict_to_list(self.action_space)
        self.observation_space = self.obs_space_dict_to_list(self.observation_space)

        # The shared observation space is a list with as many entries as we have agents. Each entry contains a numpy
        # array with the stacked observation space of all agents for now
        # (even if there are redundant observations)
        # TODO: remove redundant observations
        total_n_obs = sum([len(agent_obs_space.low) for agent_obs_space in self.observation_space])
        share_low = np.empty(shape=(total_n_obs,))
        share_high = np.empty(shape=(total_n_obs,))
        n_obs = 0
        for agent_obs in self.observation_space:
            n_agent_obs = len(agent_obs.low)
            share_low[n_obs : n_obs + n_agent_obs] = agent_obs.low
            share_high[n_obs : n_obs + n_agent_obs] = agent_obs.high
            n_obs = n_obs + n_agent_obs
        self.share_observation_space = [gym.spaces.Box(low=share_low, high=share_high) for _ in range(self.n_agents)]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment

        Args:
            seed: seed for the random number generator
            options: not needed here

        Returns:
            None

        """
        obs, obs_info = self.env.reset(seed=seed, options=options)
        obs = self._unpack_obs(obs)
        obs = ctrl_dict_to_list(obs)

        # We do not return the obs_info here as it is more complicated to handle in the DummyVecEnv provided by the
        # on-policy repository
        return obs

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, dict]:
        """
        Advance the environment (in our case, the power system) by one step in time by applying control actions to
        discrete-time dynamics and updating data sources. Handled within the System class. The actions of the RL agent
        are selected within the RL training algorithm and are passed on to the power system using a callback. After the
        system update, a reward is computed which indicates how good the action selected by the algorithm was in the
        current state. This reward is passed to the training algorithm to gradually improve the policies of the RL
        agents.

        Args:
            actions (List[np.ndarray]): actions of RL agents (here as a list of numpy arrays)

        Returns:
            Tuple: tuple containing:
                - observations of all RL agents, here as a list of observations of each agent as numpy arrays (list).
                - rewards of all RL agents (list).
                - whether the episode has terminated (bool). We assume that all agents terminate an episode at the \
                same time, as we have a centralized time management. Always false for continuous control
                - same as above (bool), but the gymnasium API makes a difference between terminated and truncated, \
                which can be useful for other environments but is not needed in our case
                - additional information (dict)

        """
        # transform the actions from a list of numpy arrays to a  nested dictionary
        # {agent_id: {node_id: {element_id: action, ...}, ...}, ...} with the original keys from the ControlEnv
        action_dict = list_to_ctrl_dict(actions, self.original_action_keys)

        for ctrl in action_dict:
            dummy_action = self.controllers[ctrl].input_space.sample()
            act_count = 0
            # fill action dictionary with values
            for n_id, n_act in dummy_action.items():
                for el_id, el_act in n_act.items():
                    num_act = el_act.shape[0]
                    dummy_action[n_id][el_id] = actions[act_count : act_count + num_act]
                    act_count = act_count + num_act
        # step original ControlEnv with the transformed action_dict
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        # convert observation dictionary to list of observations
        obs = self._unpack_obs(obs)
        obs = ctrl_dict_to_list(obs)
        if terminated:
            self.train_history = ctrl_dict_to_list(self.env.train_history)
            self.episode_history = ctrl_dict_to_list(self.env.episode_history)
        rewards = ctrl_dict_to_list(rewards)
        return obs, rewards, terminated, truncated, info

    def _unpack_obs(self, obs: dict) -> np.ndarray:
        """
        Convert dictionary of {agent_id: observation_dict} to a dictonary of
        {agent_id: flattened observation arrays}.

        Args:
            obs (dict): observation dictionary {agent_id: observation_dict}

        Returns:
            np.ndarray: flat array of observations

        """

        # Get list of all controller ids
        ctrl_ids = list(self.env.controllers.keys())

        # Initialize an empty dictionary for new observations
        new_obs_dict = {}
        # Iterate over each controller id
        for ctrl_id in ctrl_ids:
            # Get observations for this controller
            ctrl_obs = obs[ctrl_id]
            # Initialize an empty array for this controller's new observations
            new_obs = np.array([])
            # Unpack the observation dictionary for this controller
            for n_id, n_obs in ctrl_obs.items():
                for el_obs in n_obs.values():
                    new_obs = np.concatenate((new_obs, el_obs))
            # Add this controller's new observations to the dictionary
            new_obs_dict[ctrl_id] = new_obs
        # print(f"new_obs_dict: {new_obs_dict}")
        return new_obs_dict

    def act_space_dict_to_list(self, action_space: dict) -> Tuple[List[gym.spaces.Box], dict]:
        """
        Transforms an action space in the form of a nested dictionary into a list of Box spaces for each agent.
        Returns the original keys to allow re-transformation

        Args:
            action_space (dict): nested dictionary of {agent_id: {node_id: {element_id: el_action_space}}}

        Returns:
            Tuple: tuple containing:
                - list of flattened agent action spaces (List[gym.spaces.Box])
                - dictionary with original actions keys from the action space received as an input (dict)

        """
        # dictionary of {node_ids: {action_keys}}
        action_keys = {}
        env_action_space = []
        for agent_id, agent_action_space in action_space.items():
            agent_action_keys = {}
            # lower and upper limits for Box spaces
            agent_lower = np.array([])
            agent_higher = np.array([])
            for node_id, node_action_space in agent_action_space.items():
                agent_action_keys[node_id] = list(node_action_space.keys())
                for element_action_space in node_action_space.values():
                    agent_lower = np.concatenate((agent_lower, element_action_space.low))
                    agent_higher = np.concatenate((agent_higher, element_action_space.high))
            action_keys[agent_id] = agent_action_keys
            flat_agent_action_space = gym.spaces.Box(low=agent_lower, high=agent_higher)
            self.controllers[agent_id].flattened_input_space = flat_agent_action_space
            env_action_space.append(flat_agent_action_space)

        return env_action_space, action_keys

    def obs_space_dict_to_list(self, observation_space: dict) -> List[gym.spaces.Box]:
        """
        Transforms the observation space in the form of a nested dictionary into a list of Box spaces for each agent

        Args:
            observation_space (dict): nested dictionary of {agent_id: {node_id: {element_id: el_obs_space}}}

        Returns:
            List[gym.spaces.Box]: list of flattened agent observation spaces

        """
        env_obs_space = []

        def recursive_items(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, (gym.spaces.Dict)):
                    yield from recursive_items(value)
                else:
                    yield (key, value)

        for agent_id, agent_obs_space in observation_space.items():
            lower = np.array([])
            higher = np.array([])
            for element_id, element_obs_space in recursive_items(agent_obs_space):
                # print(element_obs_space)
                lower = np.concatenate((lower, element_obs_space.low))
                higher = np.concatenate((higher, element_obs_space.high))
            flat_agent_obs_space = gym.spaces.Box(low=lower, high=higher)
            self.controllers[agent_id].flattened_obs_space = flat_agent_obs_space
            env_obs_space.append(flat_agent_obs_space)

        return env_obs_space
