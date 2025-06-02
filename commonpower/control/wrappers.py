"""
Wrappers to adjust API in environments.py to different RL training algorithms.
"""
from collections import OrderedDict, deque
from functools import partial
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from commonpower.control.environments import ControlEnv
from commonpower.control.parsing import ParserFactory
from commonpower.utils.tuple_db import RLTuple, TupleDB


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


def recursive_items(dictionary):
    """
    Recursive extraction of all values in a nested dictionary or gym.spaces.Dict
    """
    for key, value in dictionary.items():
        if isinstance(value, (gym.spaces.Dict, dict)):
            yield from recursive_items(value)
        else:
            yield (key, value)


def obs_ids_and_values(dictionary, parent_key=""):
    """
    Recursive extraction of all values in a nested dictionary or gym.spaces.Dict
    """
    identifiers = []
    all_values = []
    for key, value in dictionary.items():
        full_key = f"{parent_key}.{key}" if parent_key and not parent_key == "global" else key
        if isinstance(value, (gym.spaces.Dict, dict)):
            sub_identifiers, sub_values = obs_ids_and_values(value, full_key)
            identifiers.extend(sub_identifiers)
            all_values.extend(sub_values)
        else:
            identifiers.append(full_key)
            all_values.append(value)
    return identifiers, all_values


class WrapperStack:
    def __init__(self):
        self.wrappers = []

    def add(self, wrapper: gym.Wrapper, **kwargs):
        self.wrappers.append((wrapper, kwargs))
        return self

    def get_stack(self):
        def wrap_func(env: gym.Env, wrappers: list):
            for wrapper in wrappers:
                env = wrapper[0](env, **wrapper[1])
            return env

        return partial(wrap_func, wrappers=self.wrappers)


class DeploymentWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Wrapper to standardize the deployment within CommonPower. Mainly takes care of transforming actions and
        observations such that they match the underlying wrappers (e.g., SingleAgentWrapper, MultiAgentWrapper, ...).

        Args:
            env (gym.Environment): potentially wrapped ControlEnv

        Returns: DeploymentWrapper

        """
        super().__init__(env)
        self.env = env
        self.parser = ParserFactory(env).get_parser()

    def step(self, action: Union[OrderedDict, None]) -> Tuple[dict, dict, bool, bool, dict]:
        # convert the action to the format required by the underlying environment/ wrapper stack
        if action is not None:
            action = self.parser.parse_action(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.parser.parse_obs(obs)
        return obs, reward, done, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        obs, obs_info = self.env.reset()
        obs = self.parser.parse_obs(obs)
        return obs, obs_info


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
        if len(self.env.get_wrapper_attr("controllers")) > 1:
            raise ValueError("SingleAgentWrapper cannot handle more than 1 agent")
        self.ctrl_id = list(self.env.get_wrapper_attr("controllers").keys())[0]
        # training history
        self.train_history = {}
        self.episode_history = deque(maxlen=100)

        # transform observation and action space from dictionary to box
        ctrl_obs_space = self.env.observation_space[self.ctrl_id]
        obs_low = np.array([])
        obs_high = np.array([])

        for el_id, el_obs in recursive_items(ctrl_obs_space):
            obs_low = np.concatenate((obs_low, el_obs.low))
            obs_high = np.concatenate((obs_high, el_obs.high))

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
        if terminated or truncated:
            self.train_history = self.env.get_wrapper_attr("train_history")[self.ctrl_id]
            self.episode_history = self.env.get_wrapper_attr("episode_history")[self.ctrl_id]
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
        for el_id, el_obs in recursive_items(ctrl_obs):
            new_obs = np.concatenate((new_obs, el_obs))
        return new_obs


class RecordTransitionsWrapper(gym.Wrapper):
    def __init__(
        self,
        env: ControlEnv,
        scenario_id: str,
        run_config: dict,
        seed: int,
        tuple_db: TupleDB,
        buffer_size: int = 100,
        write_buffer_on_done: bool = True,
    ):
        """
        Wrapper for recording transition tuples (s,a,s',r) either to current disk or to a data base.
        NOTE: Currently only available for single-agent RL!

        Args:
            env (gym.Env): The gym environment to be wrapped.
            tuple_db (TupleDB): The database for storing the transition tuples.
            buffer_size (int, optional): The maximum size of the tuple buffer. Defaults to 100.
            write_buffer_on_done (bool, optional): Whether to always write out the buffer on a done state.
                Defaults to True.
        """
        super().__init__(env)

        if len(env.get_wrapper_attr("controllers")) > 1:
            raise ValueError("RecordTransitionsWrapper cannot handle more than 1 agent")

        self.tuple_db = tuple_db
        self.buffer_size = buffer_size
        self.write_buffer_on_done = write_buffer_on_done

        self.tuple_buffer: List[RLTuple] = []

        # next obs structure ensures that (s, a, r, s') are collected in the correct order
        # necessary for d3rlpy library
        self.current_obs = None

        self.tuple_db.create_run(scenario_id, run_config, seed)

    def step(self, action):

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # TODO: support raw information in the future as well
        assert isinstance(self.current_obs, np.ndarray) and isinstance(
            action, np.ndarray
        ), "observation and action can only be numpy arrays for now"

        current_tuple = RLTuple(
            observation=self.current_obs,
            action=action,
            reward=reward,
            terminal=terminated,
            timeout=truncated,
        )

        self.tuple_buffer.append(current_tuple)

        if len(self.tuple_buffer) >= self.buffer_size or ((terminated or truncated) and self.write_buffer_on_done):
            self.tuple_db.record_tuples(self.tuple_buffer)
            self.tuple_buffer = []

        self.current_obs = next_obs

        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.current_obs = obs

        return obs, info


class MultiAgentWrapper(gym.Wrapper):
    def __init__(self, env: ControlEnv, remove_redundant_obs: bool = True):
        """
        Wrapper to standardize ControlEnv to the API for MAPPO/IPPO implementation of the on-policy repository
        (https://github.com/marlbenchmark/on-policy/tree/main/onpolicy). NOTE: We use our own fork of this repository,
        see the Readme file.

        Args:
            env (ControlEnv): power system environment with multi-agent API
            remove_redundant_obs (bool): whether to remove redundant observations in the shared observation space


        Returns:
             MultiAgentWrapper

        """
        super().__init__(env)
        self.env = env
        self.n_agents = len(self.get_wrapper_attr("controllers"))
        self.last_shared_obs = None
        self.remove_redundant_obs = remove_redundant_obs
        # training history
        self.train_history = {}
        self.episode_history = {}
        # the MAPPO/IPPO implementation expects the action/observation space as a list of lists
        self.action_space, self.original_action_keys = self.act_space_dict_to_list(self.action_space)
        self.observation_space, shared_obs_space = self.obs_space_dict_to_list(self.observation_space)
        # The shared observation space is a list with as many entries as we have agents.
        self.unwrapped.share_observation_space = [shared_obs_space for _ in range(self.n_agents)]

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
        self.last_shared_obs = self._get_shared_obs(obs)
        obs = self._unpack_obs(obs)
        obs = ctrl_dict_to_list(obs)
        return obs, obs_info

    def step(self, action: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, dict]:
        """
        Advance the environment (in our case, the power system) by one step in time by applying control actions to
        discrete-time dynamics and updating data sources. Handled within the System class. The actions of the RL agent
        are selected within the RL training algorithm and are passed on to the power system using a callback. After the
        system update, a reward is computed which indicates how good the action selected by the algorithm was in the
        current state. This reward is passed to the training algorithm to gradually improve the policies of the RL
        agents.

        Args:
            action (List[np.ndarray]): actions of RL agents (here as a list of numpy arrays)

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
        action_dict = list_to_ctrl_dict(action, self.original_action_keys)

        for ctrl in action_dict:
            dummy_action = self.env.get_wrapper_attr("controllers")[ctrl].input_space.sample()
            act_count = 0
            # fill action dictionary with values
            for n_id, n_act in dummy_action.items():
                for el_id, el_act in n_act.items():
                    num_act = el_act.shape[0]
                    dummy_action[n_id][el_id] = action[act_count : act_count + num_act]
                    act_count = act_count + num_act
        # step original ControlEnv with the transformed action_dict
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        # store shared obs
        self.last_shared_obs = self._get_shared_obs(obs)
        # convert observation dictionary to list of observations
        obs = self._unpack_obs(obs)
        obs = ctrl_dict_to_list(obs)
        if terminated or truncated:
            self.train_history = ctrl_dict_to_list(self.env.get_wrapper_attr("train_history"))
            self.episode_history = ctrl_dict_to_list(self.env.get_wrapper_attr("episode_history"))
        rewards = ctrl_dict_to_list(rewards)
        return obs, rewards, terminated, truncated, info

    def _get_shared_obs(self, obs: dict) -> np.ndarray:
        # get shared obs
        all_obs_ids = []
        all_obs_values = []
        for ctrl_id in self.env.get_wrapper_attr("controllers").keys():
            obs_ids, obs_values = obs_ids_and_values(obs[ctrl_id])
            all_obs_ids.extend(obs_ids)
            all_obs_values.extend(obs_values)
        # remove duplicate entries
        if self.remove_redundant_obs:
            all_obs = {obs_id: obs_value for obs_id, obs_value in zip(all_obs_ids, all_obs_values)}
            shared_obs = np.concatenate([obs for obs in all_obs.values()])
        else:
            shared_obs = np.concatenate(all_obs_values)
        return shared_obs

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
        ctrl_ids = list(self.env.get_wrapper_attr("controllers").keys())

        # Initialize an empty dictionary for new observations
        new_obs_dict = {}
        # Iterate over each controller id
        for ctrl_id in ctrl_ids:
            # Get observations for this controller
            ctrl_obs = obs[ctrl_id]
            # Initialize an empty array for this controller's new observations
            new_obs = np.array([])
            # Unpack the observation dictionary for this controller
            for el_id, el_obs in recursive_items(ctrl_obs):
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
            self.get_wrapper_attr("controllers")[agent_id].flattened_input_space = flat_agent_action_space
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
        all_obs_spaces = []
        all_obs_keys = []

        for agent_id, agent_obs_space in observation_space.items():
            element_ids, obs_spaces = obs_ids_and_values(agent_obs_space)
            lower = [obs_space.low for obs_space in obs_spaces]
            higher = [obs_space.high for obs_space in obs_spaces]
            all_obs_keys = all_obs_keys + element_ids
            all_obs_spaces = all_obs_spaces + obs_spaces
            flat_agent_obs_space = gym.spaces.Box(low=np.concatenate(lower), high=np.concatenate(higher))
            self.env.get_wrapper_attr("controllers")[agent_id].flattened_obs_space = flat_agent_obs_space
            env_obs_space.append(flat_agent_obs_space)

        # remove duplicate observations for shared observation space
        if self.remove_redundant_obs:
            shared_obs_spaces = {obs_id: obs_value for obs_id, obs_value in zip(all_obs_keys, all_obs_spaces)}
            shared_obs_spaces = [obs_space for obs_space in shared_obs_spaces.values()]
        else:
            shared_obs_spaces = all_obs_spaces
        shared_obs_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low for obs_space in shared_obs_spaces]),
            high=np.concatenate([obs_space.high for obs_space in shared_obs_spaces]),
        )
        return env_obs_space, shared_obs_space
