"""
Collection of callbacks for logging.
"""
import os
from typing import Any, Dict, Literal, Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import safe_mean
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.lib import telemetry as wb_telemetry


class SafetyCallback(BaseCallback):
    """
    Class for logging additional safety information for single-agent stable-baselines3 algorithms
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        At the end of one training episode, we want to log some information about the safety shield.

        Returns:
            None

        """
        eps_history = self.training_env.get_attr("episode_history")[0]
        # ToDo: have to adjust for training with multiple vectorized envs!
        mean_episode_penalty = safe_mean([ep_info["mean_penalty"] for ep_info in eps_history])
        mean_n_corrections = safe_mean([ep_info["n_corrections"] for ep_info in eps_history])
        mean_episode_rew_without_pen = safe_mean([ep_info["rew_without_penalty"] for ep_info in eps_history])
        self.logger.record("safety/ep_penalty_mean", mean_episode_penalty)
        self.logger.record("safety/ep_corrections_mean", mean_n_corrections)
        self.logger.record("rollout/ep_rew_without_pen_mean", mean_episode_rew_without_pen)


class WandBSafetyCallback(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
    ):
        """
        Callback for logging safety information during training of a single agent.

        Args:
            verbose (int): The verbosity of sb3 output
            model_save_path (str): Path to the folder where the model will be saved, The default value is `None` so the
                model is not logged
            model_save_freq (int): Frequency to save the model
            gradient_save_freq (int): Frequency to log gradient. The default value is 0 so the gradients are not logged.
        """
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq)

    def _on_rollout_end(self) -> None:
        """
        At the end of one training episode, we want to log some information about the safety shield.

        Returns:
            None

        """
        eps_history = self.training_env.get_attr("episode_history")[0]
        # ToDo: have to adjust for training with multiple vectorized envs!
        mean_episode_penalty = safe_mean([ep_info["mean_penalty"] for ep_info in eps_history])
        mean_n_corrections = safe_mean([ep_info["n_corrections"] for ep_info in eps_history])
        mean_episode_rew_without_pen = safe_mean([ep_info["rew_without_penalty"] for ep_info in eps_history])
        self.logger.record("safety/ep_penalty_mean", mean_episode_penalty)
        self.logger.record("safety/ep_corrections_mean", mean_n_corrections)
        self.logger.record("rollout/ep_rew_without_pen_mean", mean_episode_rew_without_pen)


class MARLBaseCallback:
    # The RL runner
    # Type hint as string to avoid circular import
    runner: "runners.BaseRunner"
    logger: Logger

    def __init__(self, verbose: int = 0):
        """
        Base class for a multi-agent callback.
        Adapted from stable-baselines3 BaseCallback
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

        Args:
            verbose (int): Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__()
        # An alias for self.runner.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, ShareVecEnv, None]
        self.num_agents = None
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, runner: "runners.BaseRunner") -> None:
        """
        Initialize the callback by saving references to the
        RL runner and the training environment for convenience.
        """
        self.runner = runner
        self.training_env = runner.envs
        self.num_agents = runner.num_agents
        self.logger = runner.logger.get_log_function()
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any], num_timesteps: int = 0) -> None:
        """
        Any operations the callback has to perform before the training starts

        Args:
            locals_ (Dict[str, Any]): local variables
            globals_ (Dict[str, Any]): global variables
            num_timesteps (int): current training progress

        Returns:
            None

        """
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.update_num_timesteps(num_timesteps)
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def on_step(self, num_timesteps: int) -> bool:
        """
        This method will be called by the runner after each call to ``env.step()``.

        Args:
            num_timesteps (int): Number of environments * number of steps per env

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.update_num_timesteps(num_timesteps)

        return self._on_step()

    def _on_step(self) -> bool:
        """
        Internal operation that should be performed in each step

        Returns:
            (bool): If the callback returns False, training is aborted early.
        """
        return True

    def on_training_end(self) -> None:
        """
        Any operations the callback has to perform after the training is finished

        Returns:
            None

        """
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        """
        Any operations the callback has to perform at the end of one training episode

        Returns:
            None

        """
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        """
        At the end of one training episode, we want to log some information about the safety shield.

        Returns:
            None

        """
        # ToDo: have to adjust for training with multiple vectorized envs!
        eps_history = self.training_env.get_attr("episode_history")[0]
        mean_penalties = [
            safe_mean([ep_info["mean_penalty"] for ep_info in eps_history[i]]) for i in range(self.num_agents)
        ]
        n_action_corrections = [
            safe_mean([ep_info["n_corrections"] for ep_info in eps_history[i]]) for i in range(self.num_agents)
        ]

        mean_episode_rew_without_pen = [
            safe_mean([ep_info["rew_without_penalty"] for ep_info in eps_history[i]]) for i in range(self.num_agents)
        ]
        for agent_id in range(self.num_agents):
            agent_k = "agent%i/" % agent_id
            self.logger.record(agent_k + "ep_penalty_mean", mean_penalties[agent_id])
            self.logger.record(agent_k + "ep_corrections_mean", n_action_corrections[agent_id])
            self.logger.record(agent_k + "ep_rew_without_pen_mean", mean_episode_rew_without_pen[agent_id])

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        Args:
            (Dict[str, Any]): the local variables during rollout collection

        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        Args:
            (Dict[str, Any]): the local variables during rollout collection

        """

    def update_num_timesteps(self, num_timesteps: int = 0) -> None:
        """

        Args:
            num_timesteps(int): number of environments * number of time steps (training progress)

        Returns:
            None

        """
        self.num_timesteps = num_timesteps


class MARLWandBCallback(MARLBaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        """
        Callback for logging experiments to Weights and Biases.

        Log MAPPO experiments to Weights and Biases
            - Added model tracking and uploading
            - Added complete hyperparameters recording
            - Note that `wandb.init(...)` must be called before the WandbCallback can be used.

        Args:
            verbose (int): The verbosity of output
            model_save_path (Optional[str]): Path to the folder where the model will be saved, The default value is \
            `None` so the model is not logged
            model_save_freq (int): Frequency to save the model
            log (Optional[Literal["gradients", "parameters", "all"]]) : What to log. One of "gradients", "parameters", \
            or "all".

        """
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn("`log` must be one of `None`, 'gradients', 'parameters', or 'all', falling back to 'all'")
            log = "all"
        self.log = log
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

    def _init_callback(self) -> None:
        pass
        # d = {}
        # algorithm_config = dict(self.runner.all_args)
        # if "algo" not in d:
        #     d["algo"] = type(self.runner).__name__
        # for key in algorithm_config:
        #     if key in wandb.config:
        #         continue
        #     if type(algorithm_config[key]) in [float, int, str]:
        #         d[key] = algorithm_config[key]
        #     else:
        #         d[key] = str(algorithm_config[key])
        #
        # wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        """
        Internal operation that should be performed in each step. Here we want to save the model from time to time.

        Returns:
            (bool): If the callback returns False, training is aborted early.

        """
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def _on_training_end(self) -> None:
        """
        We want to save the model at the end of the training.

        Returns:
            None

        """
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        """
        Call the runner to save the actor and critic parameters of each agent

        Returns:
            None

        """
        self.runner.save()
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            self.logger.info(f"Saving model checkpoint to {self.model_save_path}")
