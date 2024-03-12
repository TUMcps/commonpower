"""
Collection of loggers for controller performance.
"""
from typing import Callable

import wandb
from stable_baselines3.common.logger import Logger, make_output_format

from commonpower.control.logging.callbacks import (
    BaseCallback,
    MARLBaseCallback,
    MARLWandBCallback,
    SafetyCallback,
    WandBSafetyCallback,
)


class BaseLogger:
    def __init__(self, log_dir: str):
        """
        Base class for logging metrics during RL training.

        Args:
            log_dir (str): relative path to logging directory
        """
        self.log_dir = log_dir

    def get_log_dir(self) -> str:
        return self.log_dir

    def log_function(self) -> Callable:
        raise NotImplementedError

    def finish_logging(self) -> None:
        raise NotImplementedError


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: str, callback: BaseCallback = SafetyCallback):
        """
        Class for using tensorboard logging in single-agent stable-baselines3 algorithms.

        Args:
            log_dir (str): relative path to logging directory
            callback (BaseCallback, optional): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
        """
        super().__init__(log_dir=log_dir)
        self.callback = callback

    def log_function(self) -> BaseCallback:
        """
        Hands over the callback so it can be used by the stable-baselines3 internal logging.

        Returns:
            BaseCallback: callback which is used during training to log additional information

        """
        return self.callback()

    def finish_logging(self) -> None:
        pass


class WandBLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        entity_name: str,
        project_name: str = None,
        callback: BaseCallback = WandBSafetyCallback,
        model_save_freq: int = 100,
        verbose: int = 2,
        alg_config: dict = None,
    ):
        """
        Class for using Weights&Biases (wandb) logging in single-agent stable-baselines3 algorithms.

        Args:
            log_dir (str): relative path to logging directory
            entity_name (str): name of the wandb entity to which the runs will be logged
            project_name (str, optional): name of the wandb project to which the runs will be logged
            callback (BaseCallback, optional): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
            model_save_freq (int, optional): after how many episodes the current model should be logged
            verbose (int, optional): output verbosity
            alg_config (dict, optional): dictionary of algorithm hyperparameters. Can be used to filter runs in wandb
                API
        """

        super().__init__(log_dir=log_dir)
        self.entity_name = entity_name
        self.project_name = project_name
        self.callback = callback
        self.alg_config = alg_config
        self.model_save_freq = model_save_freq
        self.verbose = verbose

        self.run = wandb.init(
            project=self.project_name, entity=self.entity_name, config=self.alg_config, sync_tensorboard=True
        )
        self.model_save_path = self.log_dir + f"models/{self.run.id}"
        self.log_dir = self.log_dir + f"runs/{self.run.id}"

    def log_function(self) -> BaseCallback:
        """
        Hands over the callback so it can be used by the stable-baselines3 internal logging.

        Returns:
            BaseCallback: callback which is used during training to log additional information

        """
        return self.callback(
            model_save_path=self.model_save_path, model_save_freq=self.model_save_freq, verbose=self.verbose
        )

    def finish_logging(self) -> None:
        """
        Terminates the W&B run.

        Returns:
            None

        """
        wandb.finish()


class MARLTensorboardLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        callback: MARLBaseCallback = MARLBaseCallback,
        format_strings: list = ["stdout", "tensorboard"],
    ):
        """
        Class for using tensorboard logging in multi-agent IPPO/MAPPO algorithms from the on-policy repository
        (https://github.com/marlbenchmark/on-policy/blob/main/README.md).

        Args:
            log_dir (str): relative path to logging directory
            callback (MARLBaseCallback): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
            format_strings (list): list of output formats for the SB3 logger
        """
        super().__init__(log_dir=log_dir)
        self.callback = callback
        log_suffix = ""
        output_formats = [make_output_format(f, self.log_dir, log_suffix) for f in format_strings]
        self.log_function = Logger(folder=self.log_dir, output_formats=output_formats)

    def get_callback(self) -> MARLBaseCallback:
        """
        Hands over the callback.

        Returns:
            MARLBaseCallback: callback which is used during training to log additional information

        """
        return self.callback()

    def get_log_function(self) -> Callable:
        """
        Hands over the logger we get from stable-baselines3

        Returns:
            Callable: Logger

        """
        return self.log_function

    def finish_logging(self) -> None:
        pass


class MARLWandBLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        entity_name: str,
        project_name: str = None,
        callback: BaseCallback = MARLWandBCallback,
        format_strings: list = ["stdout", "tensorboard"],
        model_save_freq: int = 100,
        verbose: int = 2,
        alg_config: dict = None,
    ):
        """
        Class for using Weights&Biases (wandb) logging in single-agent stable-baselines3 algorithms
        Args:
            log_dir (str): relative path to logging directory
            entity_name (str): name of the wandb entity to which the runs will be logged
            project_name (str, optional): name of the wandb project to which the runs will be logged
            callback (BaseCallback, optional): object that implements actual logging during training -
                by defining a customized callback, additional information can be logged
                (apart from standard metrics like mean_eps_reward)
            format_strings (list): list of output formats for the SB3 logger
            model_save_freq (int, optional): after how many episodes the current model should be logged
            verbose (int, optional): output verbosity
            alg_config (dict, optional): dictionary of algorithm hyperparameters.
                Can be used to filter runs in wandb API
        """
        super().__init__(log_dir=log_dir)
        self.entity_name = entity_name
        self.project_name = project_name
        self.callback = callback
        self.alg_config = alg_config

        # init WandB
        self.model_save_freq = model_save_freq
        self.verbose = verbose

        self.run = wandb.init(
            project=self.project_name, entity=self.entity_name, config=self.alg_config, sync_tensorboard=True
        )
        self.model_save_path = self.log_dir + f"models/{self.run.id}"
        self.log_dir = self.log_dir + f"runs/{self.run.id}"

        # init logger (importer from SB3)
        log_suffix = ""
        output_formats = [make_output_format(f, self.log_dir, log_suffix) for f in format_strings]
        self.log_function = Logger(folder=self.log_dir, output_formats=output_formats)

    def get_callback(self) -> MARLBaseCallback:
        """
        Hands over the callback.

        Returns:
            MARLBaseCallback: callback which is used during training to log additional information

        """
        return self.callback(
            model_save_path=self.model_save_path, model_save_freq=self.model_save_freq, verbose=self.verbose
        )

    def get_log_function(self) -> Callable:
        """
        Hands over the logger we get from stable-baselines3

        Returns:
            Callable: Logger

        """
        return self.log_function

    def finish_logging(self) -> None:
        """
        Terminates the W&B run.

        Returns:
            None

        """
        wandb.finish()
