from typing import Callable

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from commonpower.data_forecasting.data_sources import DataSource
from commonpower.data_forecasting.nn_forecasting.data_splitting import DatasetSplit, SimpleFractionalSplit
from commonpower.data_forecasting.nn_forecasting.dataset_wrappers import DatasetWrapper, NStepAhead
from commonpower.data_forecasting.nn_forecasting.eval_metrics import EvalMetric, MeanAbsoluteError
from commonpower.data_forecasting.nn_forecasting.nn_forecasting import NNForecaster


class FlexibleModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParameterSpace(BaseModel):
    model: dict = {}
    optimizer: dict = {}
    dataset_wrapper: dict = {}
    dataset_split: dict = {}
    data_loader: dict = {}


class TrainConfig(FlexibleModel):
    forecaster: NNForecaster
    data_source: DataSource
    dataset_wrapper_class: DatasetWrapper.__class__ = NStepAhead
    dataset_split_class: DatasetSplit.__class__ = SimpleFractionalSplit
    optimizer: Optimizer.__class__ = Adam
    parameter_space: ParameterSpace = ParameterSpace()
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler: LRScheduler.__class__ | None = None
    train_loss_fcn: Callable = F.mse_loss
    eval_metrics: list[EvalMetric] = [MeanAbsoluteError()]


default_tune_config = tune.TuneConfig(
    metric=MeanAbsoluteError().name,
    mode='min',
    scheduler=AsyncHyperBandScheduler(),
    num_samples=2,
    max_concurrent_trials=8,
    trial_dirname_creator=lambda trial: f'{trial.trial_id}',
)


default_tune_run_config = train.RunConfig(
    stop=CombinedStopper(
        MaximumIterationStopper(max_iter=1e4),
        TrialPlateauStopper(metric=MeanAbsoluteError().name, std=0.01, num_results=5, grace_period=5),
    ),
    callbacks=[WandbLoggerCallback(project='nn_conformance', mode='offline')],
    checkpoint_config=train.CheckpointConfig(
        num_to_keep=1, checkpoint_score_attribute=MeanAbsoluteError().name, checkpoint_score_order='min'
    ),
)
