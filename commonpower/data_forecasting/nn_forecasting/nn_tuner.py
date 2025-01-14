"""
Tuning for NNForecasters.
"""
import os
import shutil
from typing import Callable

import torch
import torch.nn.functional as F
from ray import train as ray_train
from ray import tune as ray_tune
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from commonpower.data_forecasting.data_sources import DataSource
from commonpower.data_forecasting.nn_forecasting.config import ParameterSpace, TrainConfig
from commonpower.data_forecasting.nn_forecasting.data_splitting import DatasetSplit, SimpleFractionalSplit
from commonpower.data_forecasting.nn_forecasting.dataset_wrappers import DatasetWrapper, NStepAhead
from commonpower.data_forecasting.nn_forecasting.eval_metrics import EvalMetric, MeanAccuracy
from commonpower.data_forecasting.nn_forecasting.models import NNModule
from commonpower.data_forecasting.nn_forecasting.nn_forecasting import NNForecaster
from commonpower.utils import guaranteed_path


class NNTrainer(ray_tune.Trainable):
    """
    This class wraps the training of NNForecasters.
    """

    def _train_one_epoch(
        self,
    ) -> dict[str, float]:
        self.forecaster.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forecaster.model(data)
            loss = self.train_loss_fcn(output, target)
            loss.backward()
            self.optimizer.step()

        return {'train_loss': loss.item()}

    def test(self, model: NNModule, val_loader: DataLoader, eval_metrics: list[EvalMetric]) -> dict[str, float]:
        model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                targets.append(target)
                outputs.append(output)

        return {m.name: m(torch.cat(targets), torch.cat(outputs)) for m in eval_metrics}

    def setup(
        self,
        config: dict,
        forecaster: NNForecaster,
        data_source: DataSource,
        dataset_wrapper_class: DatasetWrapper.__class__ = NStepAhead,
        dataset_split_class: DatasetSplit.__class__ = SimpleFractionalSplit,
        optimizer: Optimizer.__class__ = Adam,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: LRScheduler.__class__ = None,
        train_loss_fcn: Callable = F.mse_loss,
        eval_metrics: list[EvalMetric] = [MeanAccuracy()],
        **kwargs,
    ):

        self.train_loss_fcn = train_loss_fcn
        self.eval_metrics = eval_metrics
        self.device = device

        param_space = ParameterSpace(**config)

        self.torch_model_kwargs = param_space.model

        self.forecaster = forecaster.setup(data_source, param_space)

        self.train_loader, self.val_loader = self.forecaster.get_train_val_loaders(
            data_source,
            param_space,
            dataset_wrapper_class,
            dataset_split_class,
        )

        self.forecaster.model.to(device)

        self.optimizer = optimizer(self.forecaster.model.parameters(), **param_space.optimizer)
        if scheduler:
            self.scheduler = scheduler(optimizer)
        else:
            self.scheduler = None

    def step(self) -> dict[str, float]:
        metric: dict = self._train_one_epoch()

        metric.update(self.test(self.forecaster.model, self.val_loader, self.eval_metrics))

        if self.scheduler:
            self.scheduler.step()

        return metric

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        checkpoint_path = guaranteed_path(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        torch.save(
            {
                'model_kwargs': self.torch_model_kwargs,
                'model_state_dict': self.forecaster.model.state_dict(),
                'feature_transformer_state_dict': self.forecaster.feature_transform.state_dict(),
                'target_transformer_state_dict': self.forecaster.target_transform.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_dir


def tune(
    train_config: TrainConfig,
    tune_config: ray_tune.TuneConfig,
    tune_run_config: ray_train.RunConfig,
    tuner_result_dir: str,
    tuner_resources: dict[str, int] = {'cpu': 8, 'gpu': 1},
) -> tuple[NNForecaster, str]:
    """
    Tunes the hyperparameters of an NNForecaster.

    Args:
        train_config (TrainConfig): The training configuration.
        tune_config (ray_tune.TuneConfig): The tuning configuration.
        tune_run_config (ray_train.RunConfig): The run configuration.
        tuner_result_dir (str): The directory to store the results of the tuner.

    Returns:
        NNForecaster: The best-performing NNForecaster instance.
        str: The directory containing the best-performing model checkpoint.
    """
    tuner = ray_tune.Tuner(
        trainable=ray_tune.with_resources(
            ray_tune.with_parameters(NNTrainer, **dict(train_config)),
            resources=tuner_resources,
        ),
        param_space=train_config.parameter_space.model_dump(),
        tune_config=tune_config,
        run_config=tune_run_config,
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    # retrive best model and store checkpoint centrally
    experiment_name = best_result.path.split("/")[-2]
    tuner_final_result_dir = guaranteed_path(os.path.join(tuner_result_dir, experiment_name))

    shutil.copytree(best_result.checkpoint.path, tuner_final_result_dir)

    print('Best config is:', best_result.config)
    print('Best model is stored in:', tuner_final_result_dir)

    return (
        train_config.forecaster.with_model(train_config.forecaster.model_class.from_checkpoint(tuner_final_result_dir)),
        tuner_final_result_dir,
    )
