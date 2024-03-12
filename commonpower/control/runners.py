"""
Runners to manage training/deployment in systems with both RL and non-RL controllers.
"""
from __future__ import annotations

import os
import random
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from itertools import chain
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import wandb
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.utils import safe_mean
from tqdm import tqdm

from commonpower.control.controller_utils import ArgsWrapper, t2n
from commonpower.control.controllers import OptimalController, RLBaseController
from commonpower.control.environments import ControlEnv
from commonpower.control.logging.loggers import BaseLogger, TensorboardLogger
from commonpower.core import System
from commonpower.modelling import ModelHistory
from commonpower.utils.cp_exceptions import InstanceError
from commonpower.utils.default_solver import get_default_solver
from commonpower.utils.helpers import to_datetime


class BaseRunner:
    def __init__(
        self,
        sys: System,
        global_controller: OptimalController = OptimalController("global"),
        forecast_horizon: timedelta = timedelta(hours=24),
        control_horizon: timedelta = timedelta(hours=24),
        dt: timedelta = timedelta(minutes=60),
        continuous_control: bool = False,
        history: ModelHistory = None,
        solver: OptSolver = get_default_solver(),
        seed: int = None,
        normalize_actions: bool = True,
    ):
        """
        Base class for any runner for power system control with one or multiple agents. Initializes the system and its
        controllers. Can be used for training one or multiple reinforcement learning (RL) agents or for deploying
        agents. Subclasses mainly have to implement the 'run()' method.

        Args:
            sys (System): power system to be controlled
            global_controller (OptimalController): instance of controller taking over control of all nodes
                that have not yet been assigned a controller. Mostly used to balance the system using a market node
                or a generator. Defaults to OptimalController("global").
            forecast_horizon (timedelta): amount of time that the controller looks into the future
            control_horizon (timedelta): amount of time to run before the system is reset if continuous_control=False
            dt (timedelta): control time interval
            continuous_control (bool): whether to use an infinite control horizon
            history (ModelHistory): logger
            solver (OptSolver): solver for optimization problem
            seed (int): seed for the global random number generator of numpy (we use np.random.seed(seed) instead
            of instantiating our own generator)
            normalize_actions (bool): whether or not to normalize the action space

        Returns:
            BaseRunner

        """
        self.sys = sys
        self.controllers = None
        self.rl_controllers = None
        self.model_inst = None
        self.env = None
        self.solver = solver
        # time handling
        self.start_time = None

        self.forecast_horizon = forecast_horizon
        self.control_horizon = control_horizon
        self.dt = dt
        self.continuous_control = continuous_control
        # dummy controller to balance system
        self.global_controller = global_controller.add_system(self.sys)
        # model history for logging
        self.history = history
        # seed for global random number generator
        self.seed = seed
        # normalizing actions of controllers (just matters for RL controllers)
        self.normalize_actions = normalize_actions

    def run(self, n_steps: int = 24, fixed_start: datetime = None) -> None:
        """
        Simulates the scenario for a given number of time steps.

        Args:
            n_steps (int): number of steps to run
            fixed_start (datetime): whether to run from a fixed given start timestamp
        """

        self.fixed_start = to_datetime(fixed_start)

        self._run(n_steps)

    def _run(n_steps: int = 24) -> None:
        raise NotImplementedError

    def prepare_run(self):
        """
        Prepare the training or deployment by initializing the system and its controllers.
        Assigns a global controller that takes over control of all entities which require inputs and
        have not been assigned a controller by the system's set-up.

        Returns: None

        """
        # initialize system
        self.sys.initialize(
            forecast_horizon=self.forecast_horizon,
            control_horizon=self.control_horizon,
            tau=self.dt,
            continuous_control=self.continuous_control,
            solver=self.solver,
        )
        # test whether system set-up is feasible
        self.system_feasible()
        # get controllers
        self.controllers = self.sys.get_controllers()
        self.rl_controllers = self.sys.get_controllers(ctrl_types=[RLBaseController])
        # seeding the global random number generator of the random & numpy module (will mainly be used for initializers)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.start_time is None:
            self.start_time = self.sys.sample_start_date(fixed_start=self.fixed_start)
        self.sys.reset(self.start_time)
        self.model_inst = self.sys.instance

    def finish_run(self):
        """
        Terminates run.

        Returns:
            None

        """
        # remove dummy controller from system
        self.sys.controllers.pop(self.global_controller.name, None)
        self.global_controller.detach()

    def set_start_time(self, start_time: datetime):
        """
        Set start time from external.

        Args:
            start_time (datetime): date and hour at which to reset the system before starting a run.

        Returns:
            None

        """
        self.start_time = start_time

    def system_feasible(self, n_checks: int = 1):
        """
        Check whether the current system set-up is feasible.

        Args:
            n_checks (int): number of feasibility checks to run

        Returns:
            None

        """
        for i in range(n_checks):
            start_time = self.sys.sample_start_date(fixed_start=None)
            self.sys.reset(start_time)
            inst = self.sys.instance
            results = self.solver.solve(inst)
            if results.solver.termination_condition in [
                TerminationCondition.infeasible,
                TerminationCondition.unbounded,
                TerminationCondition.infeasibleOrUnbounded,
            ]:
                raise InstanceError(
                    inst, "Solving the model is infeasible or unbounded, please consider your system set-up"
                )


class BaseTrainer(BaseRunner):
    def __init__(
        self,
        sys: System,
        global_controller: OptimalController = OptimalController("global"),
        wrapper: gym.Wrapper = None,
        forecast_horizon: timedelta = timedelta(hours=24),
        control_horizon: timedelta = timedelta(hours=24),
        dt: timedelta = timedelta(minutes=60),
        continuous_control: bool = False,
        history: ModelHistory = None,
        solver: OptSolver = get_default_solver(),
        save_path: str = "./saved_models/test_model",
        seed: int = None,
        normalize_actions: bool = True,
    ):
        """
        Base class for any runner used for training one or multiple reinforcement learning (RL) agents.

        Args:
            sys (System): power system to be controlled
            global_controller (OptimalController): instance of controller taking over control of all nodes
                that have not yet been assigned a controller.
                Mostly used to balance the system using a market node or a generator.
                Defaults to OptimalController("global").
            wrapper (gym.Wrapper): wrapper for the environment that handles the RL agents during training
                (used for example for single-agent RL control).
            forecast_horizon (timedelta): amount of time that the controller looks into the future
            control_horizon (timedelta): amount of time to run before the system is reset if continuous_control=False
            dt (timedelta): control time interval
            continuous_control (bool): whether to use an infinite control horizon
            history (ModelHistory): logger
            solver (OptSolver): solver for optimization problem
            save_path (str): local path to folder in which the trained policy will be stored (as .zip file)
                after the training is finished
            seed (int): seed for the global random number generator of numpy (we use np.random.seed(seed) instead
            of instantiating our own generator)
            normalize_actions (bool): whether or not to normalize the action space

        Returns:
            BaseTrainer

        """
        super().__init__(
            sys=sys,
            global_controller=global_controller,
            forecast_horizon=forecast_horizon,
            control_horizon=control_horizon,
            dt=dt,
            continuous_control=continuous_control,
            history=history,
            solver=solver,
            seed=seed,
            normalize_actions=normalize_actions,
        )
        # environment wrapper function
        self.wrapper = wrapper
        # model save path
        self.save_path = save_path

    def prepare_run(self):
        """
        In addition to the preparation in BaseRunner, we also instantiate an environment function as an API for the RL
        training.

        Returns: None

        """
        super().prepare_run()
        # create environment function according to gymnasium API
        if len(list(self.sys.get_controllers(ctrl_types=[RLBaseController]))) >= 1:
            self.env = self.sys.create_env_func(
                self.wrapper, self.fixed_start, normalize_actions=self.normalize_actions
            )


class SingleAgentTrainer(BaseTrainer):
    def __init__(
        self,
        sys: System,
        alg_config: dict,
        global_controller: OptimalController = OptimalController("global"),
        policy: BasePolicy = None,
        wrapper: gym.Wrapper = None,
        logger: BaseLogger = None,
        forecast_horizon: timedelta = timedelta(hours=24),
        control_horizon: timedelta = timedelta(hours=24),
        dt: timedelta = timedelta(minutes=60),
        continuous_control: bool = False,
        history: ModelHistory = None,
        solver: OptSolver = get_default_solver(),
        save_path: str = "./saved_models/test_model",
        seed: int = None,
        normalize_actions: bool = True,
    ):
        """
        Runner for training a single RL agent (with algorithms from the StableBaselines 3 repository).

        Args:
            sys (System): power system to be controlled
            global_controller (OptimalController): instance of controller taking over control of all nodes
                that have not yet been assigned a controller. Mostly used to balance the system using
                a market node or a generator. Defaults to OptimalController("global").
            alg_config (dict): configuration for the RL algorithm and policy to be trained
            policy (BasePolicy): policy instance (can be handed over to be retrained)
            wrapper (gym.Wrapper): wrapper for the environment that handles the RL agents during training
                (used for example for single-agent RL control).
            logger (BaseLogger): object for handling training logs
            forecast_horizon (timedelta): amount of time that the controller looks into the future
            control_horizon (timedelta): amount of time to run before the system is reset if continuous_control=False
            dt (timedelta): control time interval
            continuous_control (bool): whether to use an infinite control horizon
            history (ModelHistory): logger
            solver (OptSolver): solver for optimization problem
            save_path (str): local path to folder in which the trained policy will be stored (as .zip file)
                after the training is finished
            seed (int): seed for the global random number generator of numpy (we use np.random.seed(seed) instead
            of instantiating our own generator)
            normalize_actions (bool): whether or not to normalize the action space

        Returns:
            SingleAgentTrainer

        """
        super().__init__(
            sys=sys,
            global_controller=global_controller,
            wrapper=wrapper,
            forecast_horizon=forecast_horizon,
            control_horizon=control_horizon,
            dt=dt,
            continuous_control=continuous_control,
            history=history,
            solver=solver,
            save_path=save_path,
            seed=seed,
            normalize_actions=normalize_actions,
        )
        self.alg_config = alg_config
        self.policy = policy
        if logger is None:
            warnings.warn("No logger specified. Writing tensorboard log files to 'default_log/' directory.")
            self.logger = TensorboardLogger(log_dir="./default_log/")
        else:
            self.logger = logger

    def _run(self, n_steps: int = 24):
        """
        Runs the single-agent RL training algorithm for a given number of time steps and saves the trained policy.

        Returns:
            None

        """
        self.prepare_run()
        training_steps = self.alg_config["total_steps"]
        self.policy.learn(total_timesteps=training_steps, callback=self.logger.log_function())
        # store reference to model in controller
        for ctrl in self.sys.get_controllers(ctrl_types=[RLBaseController]).values():
            ctrl.save(self.policy, save_path=self.save_path)

        self.finish_run()

    def prepare_run(self):
        """
        Prepare the training by initializing the system and its controllers. Assigns a global controller that
        takes over control of all entities which require inputs and have not been assigned a controller by the system's
        set-up. Sets an initial policy if no pre-trained policy was handed over at instantiation.

        Returns:
            None

        """
        super().prepare_run()
        TrainAlg = self.alg_config["algorithm"]
        if not self.policy:
            self.policy = TrainAlg(
                env=self.env,
                policy=self.alg_config["policy"],
                learning_rate=self.alg_config["learning_rate"],
                device=self.alg_config["device"],
                n_steps=self.alg_config["n_steps"],
                batch_size=self.alg_config["batch_size"],
                tensorboard_log=self.logger.get_log_dir(),
                seed=self.seed,
                verbose=2,
            )

    def finish_run(self):
        super().finish_run()
        self.logger.finish_logging()


class DeploymentRunner(BaseRunner):
    def __init__(
        self,
        sys: System,
        global_controller: OptimalController = OptimalController("global"),
        alg_config: dict = None,
        wrapper: gym.Wrapper = None,
        forecast_horizon: timedelta = timedelta(hours=24),
        control_horizon: timedelta = timedelta(hours=24),
        dt: timedelta = timedelta(minutes=60),
        continuous_control: bool = False,
        history: ModelHistory = None,
        solver: OptSolver = get_default_solver(),
        seed: int = None,
        normalize_actions: bool = True,
    ):
        """
        Runner for the deployment of multiple heterogeneous controllers (RL, optimal control).

        Args:
            sys (System): power system to be controlled
            global_controller (OptimalController): instance of controller taking over control of all nodes
                that have not yet been assigned a controller. Mostly used to balance the system using
                a market node or a generator. Defaults to OptimalController("global").
            alg_config (dict): configuration for the RL algorithm and policy to be trained
            wrapper (gym.Wrapper): wrapper for the environment that handles the RL agents during training
                (used for example for single-agent RL control).
            forecast_horizon (timedelta): amount of time that the controller looks into the future
            control_horizon (timedelta): amount of time to run before the system is reset if continuous_control=False
            dt (timedelta): control time interval
            continuous_control (bool): whether to use an infinite control horizon
            history (ModelHistory): logging
            solver (OptSolver): solver for optimization problem
            seed (int): seed for the global random number generator of numpy (we use np.random.seed(seed) instead
            of instantiating our own generator)
            normalize_actions (bool): whether or not to normalize the action space

        Returns:
            DeploymentRunner

        """
        super().__init__(
            sys=sys,
            global_controller=global_controller,
            forecast_horizon=forecast_horizon,
            control_horizon=control_horizon,
            dt=dt,
            continuous_control=continuous_control,
            history=history,
            solver=solver,
            seed=seed,
            normalize_actions=normalize_actions,
        )
        self.alg_config = alg_config
        self.wrapper = wrapper

    def _run(self, n_steps: int = 24):
        """
        Runs the deployment of multiple heterogeneous controllers for a given number of time steps.

        Args:
            n_steps (int): number of time steps to run the system for

        Returns:
            None

        """
        self.prepare_run()
        # run
        obs, _ = self.sys.observe()

        for step in tqdm(range(n_steps)):
            obs, reward, terminated, _, info = self.sys.step(obs=obs, history=self.history)

            if terminated:
                self.sys.reset(self.sys.sample_start_date(fixed_start=self.fixed_start))

        self.finish_run()

    def prepare_run(self):
        """
        Prepare the deployment by initializing the system and its controllers. Assigns a global controller that
        takes over control of all entities which require inputs and have not been assigned a controller by the system's
        set-up. Sets the operation mode of all RL controllers within the system to 'deployment'.

        Returns:

        """
        super().prepare_run()
        if self.rl_controllers:
            self.env = self.sys.create_env_func(
                self.wrapper, self.fixed_start, normalize_actions=self.normalize_actions
            )
            # set train flag of RL runners to False
            for rl_ctrl in self.rl_controllers.values():
                rl_ctrl.set_mode("deploy")
                # load RL policies
                self.alg_config["seed"] = self.seed  # need to hand over the seed to re-load the policy
                if not rl_ctrl.policy:
                    rl_ctrl.load(env=self.env, config=self.alg_config)


class MAPPOTrainer(BaseTrainer):
    def __init__(
        self,
        sys: System,
        alg_config: dict,
        global_controller: OptimalController = OptimalController("global"),
        wrapper: gym.Wrapper = None,
        logger: BaseLogger = None,
        forecast_horizon: timedelta = timedelta(hours=24),
        control_horizon: timedelta = timedelta(hours=24),
        dt: timedelta = timedelta(minutes=60),
        continuous_control: bool = False,
        history: ModelHistory = None,
        solver: OptSolver = get_default_solver(),
        save_path: str = "./saved_models/test_model",
        seed: int = None,
        normalize_actions: bool = True,
    ):
        """
        Runner for training multiple heterogeneous agents with MAPPO/IPPO from the on-policy repository
        (https://github.com/marlbenchmark/on-policy/tree/main/onpolicy). Based on our BaseTrainer and our logging
        framework as well as the BaseRunner from the on-policy repository
        Args:
            sys (System): power system to be controlled
            global_controller (OptimalController): instance of controller taking over control of all nodes
                that have not yet been assigned a controller. Mostly used to balance the system using
                a market node or a generator. Defaults to OptimalController("global").
            alg_config (dict): configuration for the RL algorithm and policy to be trained
            wrapper (gym.Wrapper): wrapper for the environment that handles the RL agents during training
                (used for example for single-agent RL control).
            logger (BaseLogger): object for handling training logs
            forecast_horizon (timedelta): amount of time that the controller looks into the future
            control_horizon (timedelta): amount of time to run before the system is reset if continuous_control=False
            dt (timedelta): control time interval
            continuous_control (bool): whether to use an infinite control horizon
            history (ModelHistory): logging
            solver (OptSolver): solver for optimization problem
            save_path (str): local path to folder in which the trained policy will be stored (as .zip file)
                after the training is finished
            seed (int): seed for the global random number generator of numpy (we use np.random.seed(seed) instead
            of instantiating our own generator)
            normalize_actions (bool): whether or not to normalize the action space

        """
        super().__init__(
            sys=sys,
            global_controller=global_controller,
            wrapper=wrapper,
            forecast_horizon=forecast_horizon,
            control_horizon=control_horizon,
            dt=dt,
            continuous_control=continuous_control,
            history=history,
            solver=solver,
            save_path=save_path,
            seed=seed,
            normalize_actions=normalize_actions,
        )

        # logging
        self.logger = logger
        self.callback = logger.get_callback()
        self.log_function = logger.get_log_function()

        all_args = alg_config
        self.all_args = ArgsWrapper(all_args)
        # set device
        self._set_device()
        # check other arguments according to algorithm:
        self._check_alg_config()
        # parse arguments
        self._parse_alg_config()

        # IPPO/MAPPO-specific attributes
        self.num_agents = None
        # MAPPO enables training with multiple envs in parallel, which we currently do not use,
        # but still keep to preserve compatibility
        self.envs = None
        self.eval_envs = None  # MAPPO enables evaluating training regulary on separate environments - also not used atm
        self.policy = []  # list of policies for each agent
        self.trainer = []  # list of training algorithm objects for each agent
        self.buffer = []  # list of buffers for each agent
        self.ep_info_buffer = []  # list of buffers for each agent which contain infos for each episode

    def prepare_run(self):
        # import these here such that no errors will be thrown in case someone does not have the on-policy repo
        # installed
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
        from onpolicy.utils.separated_buffer import SeparatedReplayBuffer

        """
        Prepare the training or deployment by initializing the system and its controllers.
        Assigns a global controller that takes over control of all entities which require inputs and
        have not been assigned a controller by the system's set-up.
        Takes care of initializing the training environment such that it is compatible with MAPPO/IPPO algorithm

        Returns:
            None

        """
        # initialize system
        self.sys.initialize(
            forecast_horizon=self.forecast_horizon,
            control_horizon=self.control_horizon,
            tau=self.dt,
            continuous_control=self.continuous_control,
        )
        # test whether system set-up is feasible
        self.system_feasible()
        # get controllers
        self.controllers = self.sys.get_controllers()
        self.rl_controllers = self.sys.get_controllers(ctrl_types=[RLBaseController])
        self.num_agents = len(self.rl_controllers)
        # make directories for saving the trained models
        for agent_id in range(self.num_agents):
            os.makedirs(self.save_path + "/agent" + str(agent_id), exist_ok=True)
        # system reset
        if self.start_time is None:
            self.start_time = self.sys.sample_start_date(fixed_start=self.fixed_start)
        self.sys.reset(self.start_time)
        self.model_inst = self.sys.instance
        # seeding the global random number generator of the torch module (mainly used for weight initialization)
        # and random & numpy module (will mainly be used for initializers)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # create environment, including vectorization wrappers

        def make_vec_env(all_args, system):
            def get_env_fn():
                def init_env():
                    env = ControlEnv(system=system)
                    if self.wrapper:
                        env = self.wrapper(env)
                    return env

                return init_env

            if all_args.n_rollout_threads == 1:
                return DummyVecEnv([get_env_fn()])
            else:
                return SubprocVecEnv([get_env_fn() for i in range(all_args.n_rollout_threads)])

        self.envs = make_vec_env(all_args=self.all_args, system=self.sys)
        # used to set self._seeds, which will then be used at reset to actually seed the envs
        self.envs.seed(self.seed)
        self.eval_envs = make_vec_env(all_args=self.all_args, system=self.sys)
        self.eval_envs.seed(self.seed)

        # set up the policies (actor and critic networks) of all agents
        for agent_id in range(self.num_agents):
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)

        # set up training algorithm instances and replay buffers
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )
            ep_info_bu = deque(maxlen=100)
            self.buffer.append(bu)
            self.ep_info_buffer.append(ep_info_bu)
            self.trainer.append(tr)

        self.warmup()

        # initialize callback
        self.callback.init_callback(runner=self)

    def _run(self, n_steps: int = 24):
        """
        Runs the multi-agent RL training algorithm (MAPPO or IPPO) for a given number of time steps
        and saves the trained policies.

        Returns:
            None

        """
        self.prepare_run()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print(f"Total episodes: {episodes}")

        # start logging
        self.callback.on_training_start(locals(), globals())
        # in each episode: collect rollouts and update policies based on them
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # logging for episode
            self.callback.on_rollout_start()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Observe reward and next obs
                # The on-policy training algorithms are based on the deprecated OpenAI gym environment API.
                # The step() function of the Gymnasium API returns additional information,
                # the "truncated" variable. We don't need it here, which is why we filter it out in the DummyVecEnv
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                # logging
                self.callback.update_locals(locals())
                num_timesteps = step * self.n_rollout_threads
                continue_training = self.callback.on_step(num_timesteps)
                # option to abort early, not used at the moment
                if not continue_training:
                    break

            # compute return and update network
            self.compute()
            # update policies based on data that is currently in the buffer
            train_infos = self.train()
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()

                for agent_id in range(self.num_agents):
                    # idv_rews = []
                    # ToDo: How to get individual rewards? Do we need this?
                    # for info in infos:
                    #     for count, info in enumerate(infos):
                    #         if 'individual_reward' in infos[count][agent_id].keys():
                    #             idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                    # train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                    self.ep_info_buffer[agent_id].append(
                        np.mean(self.buffer[agent_id].rewards) * int(self.forecast_horizon / self.dt)
                    )
                    train_infos[agent_id].update({"average_episode_rewards": safe_mean(self.ep_info_buffer[agent_id])})

                self.log_train(train_infos, total_num_steps, start, end)

                # invoke callback to log additional information
                self.callback.on_rollout_end()
                # this makes sure the logged information is written to the different output formats
                # (e.g., stdout, tensorboard)
                self.log_function.dump(total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        # save the models
        self.save()
        # end logging
        self.callback.on_training_end()
        self.finish_run()

    def warmup(self):
        """
        Pre-training preparations specific to MAPPO/IPPO

        Returns:
            None

        """
        # Seeding is considered within DummyVecEnv/SubProcVecEnv through seeds initialized previously
        # by calling self.envs.seed(seed=self.seed)
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].share_obs[0] = share_obs[0].copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[np.array, List[np.array], List[np.array], np.array, np.array, List[np.array]]:
        """
        Obtain actions for the current step based on current policies, observations, shared observations, and hidden
        states. The masks are not necessary in our case, because all agents terminate at the same time.

        Args:
            step (int): The current step within the episode

        Returns:
            Tuple: tuple containing:
                - values (np.array)
                - actions (List[np.array])
                - action probabilities, logarithmic (List[np.array])
                - hidden states of recurrent NN actor. Only needed for recurrent policies (np.array)
                - hidden states of recurrent NN critic. Only needed for recurrent policies (np.array)
                - environment actions ? not sure, adapted from on-policy BaseRunner (List[np.array])

        """
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )

            values.append(t2n(value))
            action = t2n(action)

            actions.append(action)
            action_log_probs.append(t2n(action_log_prob))
            rnn_states.append(t2n(rnn_state))
            rnn_states_critic.append(t2n(rnn_state_critic))

        actions_env = actions

        values = np.array(values).transpose(1, 0, 2)
        # actions = np.array(actions).transpose(1, 0, 2)
        # action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(
        self,
        data: Tuple[
            np.array,
            np.array,
            np.array,
            np.array,
            np.array,
            List[np.array],
            List[np.array],
            np.array,
            np.array,
        ],
    ) -> None:
        """
        Write information collected during rollout to buffers (one per agent) in the appropriate format (the
        "SeparatedReplayBuffer" from the on-policy repository logs some information we do not require, like masks for
        terminated agents).

        Args:
            data (Tuple): data collected during rollout which should be inserted into buffers

        Returns:
            None

        """
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        # Our ControlEnv and MAWrapper return one terminated/dones variable per environment and not one per agent,
        # so we reset all rnn_states if this "dones" is true
        if dones:
            for i, rnn_state in enumerate(rnn_states):
                rnn_states[i] = np.zeros((1, self.recurrent_N, self.hidden_size), dtype=np.float32)
                rnn_states[i] = np.zeros((1, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # Problem: dones in our env only says if the env is terminated or not, not if a specific agent is done
        # original: masks[np.array([dones]) is True] = \
        #               np.zeros(((np.array([dones]) is True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[agent_id],
                action_log_probs[agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps: int):
        """
        Evaluates current policies on separate eval environment (not used atm).

        Args:
            total_num_steps (int): Current training progress

        Returns:
            None

        """
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Observe reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones is True] = np.zeros(
                ((eval_dones is True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones is True] = np.zeros(((eval_dones is True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        # not used
        pass

    def compute(self):
        """
        Compute returns based on next value (will be needed for loss)

        Returns:
            None

        """
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self) -> List[dict]:
        """
        Perform updates of actor and critic parameters for each agent

        Returns:
            List[dict]: list of training metrics dictionary (one list entry per agent)

        """
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        """
        Save the actor and critic parameters for each agent

        Returns:
            None

        """
        for agent_id, agent in enumerate(self.rl_controllers.values()):
            policy_actor = self.trainer[agent_id].policy.actor
            actor_save_path = str(self.save_path) + "/agent" + str(agent_id) + "/actor_agent" + ".pt"
            agent.save(policy_actor, actor_save_path)
            policy_critic = self.trainer[agent_id].policy.critic
            critic_save_path = str(self.save_path) + "/agent" + str(agent_id) + "/critic_agent" + ".pt"
            agent.save(policy_critic, critic_save_path)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnorm = self.trainer[agent_id].value_normalizer
                vnorm_save_path = str(self.save_path) + "/agent" + str(agent_id) + "/vnrom_agent" + ".pt"
                agent.save(policy_vnorm, vnorm_save_path)

    def log_train(self, train_infos: List[dict], total_num_steps: int, start: time.time = None, end: time.time = None):
        """

        Args:
            train_infos (List[dict]): training metrics for each agent
            total_num_steps (int): current training progress
            start (time.time): start time of training episode
            end (time.time): end time of training episode

        Returns:
            None

        """
        self.log_function.record("time/total_timesteps", total_num_steps)
        if start:
            self.log_function.record("time/fps", int(total_num_steps / (end - start)))
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if isinstance(v, torch.Tensor):
                    self.log_function.record(agent_k, v.item())
                else:
                    self.log_function.record(agent_k, v)

    def finish_run(self):
        """
        Finish run, mostly needed for deleting global controller and terminating Weights&Biases logger

        Returns:
            None

        """
        super().finish_run()
        self.logger.finish_logging()

    def _set_device(self):
        """
        Set computing device according to algorithm configuration

        Returns:
            None

        """
        if self.all_args.cuda and torch.backends.cuda.is_available():
            print("choose to use gpu...")
            self.device = torch.device("cuda")
            torch.set_num_threads(self.all_args.n_training_threads)
            if self.all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            self.device = torch.device("cpu")
            torch.set_num_threads(self.all_args.n_training_threads)

    def _check_alg_config(self):
        """
        Sanity check for the algorithm configuration: If we use any variant of MAPPO, we want a shared observation space
        which means that use_centralized_V has to be true. If we use a recurrent policy (RMAPPO), the respective
        arguments have to be true.

        Returns:
            None
        """
        # we do not support training with vectorized environments atm
        if self.all_args.n_rollout_threads > 1:
            raise ValueError(
                "Parameter 'n_rollout_threads' has to equal '1' as we do not yet support training with "
                "multiple environments simultaneously!"
            )
        if self.all_args.algorithm_name == "rmappo":
            print("You are choosing to use RMAPPO, we set use_recurrent_policy to be True")
            self.all_args.use_recurrent_policy = True
            self.all_args.use_naive_recurrent_policy = False
            self.all_args.use_centralized_V = True
        elif self.all_args.algorithm_name == "mappo":
            print("You are choosing to use MAPPO, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
            self.all_args.use_recurrent_policy = False
            self.all_args.use_naive_recurrent_policy = False
            self.all_args.use_centralized_V = True
        elif self.all_args.algorithm_name == "ippo":
            print("You are choosing to use IPPO, we set use_centralized_V to be False")
            self.all_args.use_centralized_V = False
        else:
            raise NotImplementedError

    def _parse_alg_config(self):
        """
        Write algorithm configuration to class attributes

        Returns:
            None

        """
        # parameters
        self.algorithm_name = self.all_args.algorithm_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        if wandb.run:
            self.save_path = str(self.callback.model_save_path)
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
