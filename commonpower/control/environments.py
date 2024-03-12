"""
Base API (based on gymnasium API) between controlled system and RL training algorithms.
"""
from collections import OrderedDict, deque
from copy import copy, deepcopy
from datetime import datetime
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from commonpower.modelling import ControllableModelEntity
from commonpower.utils.cp_exceptions import ControllerError


class ControlEnv(gym.Env):
    def __init__(
        self,
        system: ControllableModelEntity,
        continuous_control: bool = False,
        fixed_start: datetime = None,
        normalize_action_space: bool = True,
    ):
        """
        Class that provides the interface between our power system and any reinforcement learning algorithm. Based on
        the OpenAI Gym API (which is now maintained as 'gymnasium', see https://gymnasium.farama.org/). Manages all
        RL controllers within the power system.

        Args:
            system (ControllableModelEntity): power system including Pyomo model with all constraints
            continuous_control (bool): whether to use an infinite control horizon
            fixed_start (datetime): if None, we will train from multiple random start times.
                Otherwise, we will always train from the same start time.
            normalize_action_space (bool): whether to normalize the action space to [-1,1]


        Returns:
            ControlEnv

        """
        from commonpower.control.controllers import RLBaseController

        self.controllers = system.get_controllers(ctrl_types=[RLBaseController])
        self.sys = system
        self.current_action = None
        self.train_history = {}
        self.episode_history = {agent_id: deque(maxlen=100) for agent_id in self.controllers.keys()}
        self.normalize_actions = normalize_action_space

        # ToDo: shared observation space?
        self.observation_space = self._get_observation_space()
        if self.normalize_actions:
            self.action_space, self.original_action_space = self._get_normalized_action_space()
        else:
            self.action_space = self._get_action_space()

        # whether to just continuously step through the year or not
        self.continuous_control = continuous_control
        # whether or not to train on a fixed day
        self.fixed_start = fixed_start

        self.n_steps = 0

    def step(self, action: OrderedDict) -> Tuple[dict, dict, bool, bool, dict]:
        """
        Advance the environment (in our case, the power system) by one step in time by applying control actions to
        discrete-time dynamics and updating data sources. Handled within the System class. The actions of the RL agent
        are selected within the RL training algorithm and are passed on to the power system using a callback. After the
        system update, a reward is computed which indicates how good the action selected by the algorithm was in the
        current state. This reward is passed to the training algorithm to gradually improve the policies of the RL
        agents.

        Args:
            action (OrderedDict): actions of RL agents (here as a dictionary of agent IDs and their respective actions)

        Returns:
            Tuple: tuple containing:
                - observations of all RL agents (dict), here as a dictionary of agent IDs and their respective \
                observations
                - rewards of all RL agents (dict)
                - whether the episode has terminated (bool). We assume that all agents terminate an episode at the \
                same time, as we have a centralized time management. Always false for continuous control
                - same as above but the gymnasium API makes a difference between terminated and truncated, which can \
                be useful for other environments but is not needed in our case
                - additional information (dict)

        """
        # expects a list of actions or a single action (numpy array) as an input
        if len(self.controllers) == 1 and isinstance(action, list):
            raise ControllerError(self.controllers[0], "One agent but multiple actions")
        if len(self.controllers) > 1 and isinstance(action, float):
            raise ControllerError(self.controllers[0], "Multiple agents but only one action")

        # store action
        if self.normalize_actions:
            action = self._denormalize_action(action)
            for ctrl in self.controllers.values():
                ctrl.set_normalize_inputs(True)

        self.current_action = action

        obs, costs, terminated, truncated, info = self.sys.step(rl_action_callback=self.rl_action_callback)
        # extract only the info for the RL controllers
        obs = {agent: agent_obs for agent, agent_obs in obs.items() if agent in self.controllers.keys()}
        # rewards are negative costs
        rewards = {agent: -agent_cost for agent, agent_cost in costs.items() if agent in self.controllers.keys()}
        # update history with reward-penalty
        for agent_id, agent in self.controllers.items():
            agent.update_history({"reward_without_penalty": rewards[agent_id] + agent.history["safety_penalty"][-1][1]})
        # get train history at end of episode:
        if terminated:
            self.train_history = {agent_id: copy(agent.history) for agent_id, agent in self.controllers.items()}
            for agent_id in self.controllers.keys():
                self.episode_history[agent_id].append(
                    {
                        "mean_penalty": np.mean([t[1] for t in self.train_history[agent_id]["safety_penalty"]]),
                        "rew_without_penalty": np.sum(
                            [t[1] for t in self.train_history[agent_id]["reward_without_penalty"]]
                        ),
                        "n_corrections": np.sum([t[1] for t in self.train_history[agent_id]["action_corrected"]]),
                    }
                )

        return obs, rewards, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Reset the power system to the beginning of an episode (which spans 24 hours).

        Args:
            seed:  The seed that is used to initialize the environment’s PRNG (np_random). If the environment does not
                already have a PRNG and seed=None (the default option) is passed, a seed will be chosen from some \
                source of entropy (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and \
                seed=None is passed, the PRNG will not be reset. If you pass an integer, the PRNG will be reset even \
                if it already exists. Usually, you want to pass an integer right after the environment has been \
                initialized and then never again. This should be taken care of by calling super().reset(seed=seed) in \
                the first line of this function. (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
            options: not needed

        Returns:
            Tuple: tuple containing
                - observations of all RL agents after reset (dict)
                - additional information for observations (dict)

        """
        # call reset() of gymnasium Env class to ensure that we only reset ONCE right after initialization
        # and then never again
        super().reset(seed=seed)

        self.n_steps = 0
        reset_time = self.sys.sample_start_date(self.fixed_start)
        self.sys.reset(reset_time)
        obs, obs_info = self.sys.observe()
        # extract only the info for the RL controllers
        obs = {agent: agent_obs for agent, agent_obs in obs.items() if agent in self.controllers.keys()}
        return obs, obs_info

    def rl_action_callback(self, ctrl_id: str):
        """
        Passes current action selected by training algorithm to the compute_control_input() function of the
        BaseController class.

        Args:
            ctrl_id(str): ID of the controller for which to retrieve the action

        Returns:
            dict: actions for all controlled entities assigned to this controller

        """
        return self.current_action[ctrl_id]

    def _get_observation_space(self) -> gym.spaces.Dict:
        """
        Retrieve observation space from list of RL controllers and their observation masks.

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their observation spaces

        """
        # ToDo: What happens in case we don't have a box space?
        obs_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            ctrl_obs_space = {}
            nodes = ctrl.get_nodes()
            for node in nodes:
                node_obs_space = node.observation_space(ctrl.obs_mask)
                if node_obs_space is not None:
                    ctrl_obs_space[node.id] = node_obs_space
            if "global" in ctrl.obs_mask.keys():
                ctrl_obs_space["global"] = self.sys.global_observation_space(ctrl.obs_mask["global"])
            ctrl_obs_space = gym.spaces.Dict({node_id: node_space for node_id, node_space in ctrl_obs_space.items()})
            obs_spaces[ctrl_id] = ctrl_obs_space

        obs_spaces = gym.spaces.Dict(obs_spaces)
        return obs_spaces

    def _get_action_space(self) -> gym.spaces.Dict:
        """
        Retrieve action space from RL controllers

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their action spaces

        """
        # ToDo: What happens in case we don't have a box space?
        act_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            act_spaces[ctrl_id] = ctrl.get_input_space(normalize=False)
        act_spaces = gym.spaces.Dict(act_spaces)
        return act_spaces

    def _get_normalized_action_space(self) -> gym.spaces.Dict:
        """
        Normalize all actions to [-1,1]

        Returns:
            gym.spaces.Dict: dictionary of agent IDs and their action spaces

        """
        # ToDo: What happens in case we don't have a box space?
        act_spaces = OrderedDict()
        norm_act_spaces = OrderedDict()
        for ctrl_id, ctrl in self.controllers.items():
            act_spaces[ctrl_id] = ctrl.get_input_space(normalize=False)
            norm_act_spaces[ctrl_id] = ctrl.get_input_space(normalize=True)
        norm_act_spaces = gym.spaces.Dict(norm_act_spaces)
        act_spaces = gym.spaces.Dict(act_spaces)
        return norm_act_spaces, act_spaces

    def _denormalize_action(self, action: OrderedDict) -> OrderedDict:
        """
        Denormalize action to original input space.

        Args:
            action (OrderedDict): normalized action

        Returns:

        """
        scaled_action = deepcopy(action)
        for ctrl_id, ctrl_action in action.items():
            for node_id, node_action in ctrl_action.items():
                for el_id, el_action in node_action.items():
                    action_low = self.original_action_space[ctrl_id][node_id][el_id].low
                    action_high = self.original_action_space[ctrl_id][node_id][el_id].high

                    new_action = (action[ctrl_id][node_id][el_id] - (-1 * np.ones((len(action_high,))))) / 2 * np.ones(
                        (
                            len(
                                action_high,
                            )
                        )
                    ) * (action_high - action_low) + action_low
                    scaled_action[ctrl_id][node_id][el_id] = new_action

        return scaled_action
