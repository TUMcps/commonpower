import unittest
import os
import shutil

from pathlib import Path
from functools import partial

from commonpower.core import System, Bus
from commonpower.models.components import *
from commonpower.models.busses import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import RLControllerSB3, OptimalController
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from stable_baselines3 import PPO
from commonpower.data_forecasting.forecasters import *
from commonpower.utils.param_initialization import *
from commonpower.data_forecasting.data_sources import CSVDataSource
from commonpower.data_forecasting.base import DataProvider
from commonpower.modelling import ModelHistory
from commonpower.control.logging.loggers import TensorboardLogger
from commonpower.control.logging.callbacks import *


class TestControl(unittest.TestCase):
    def setUp(self):
        os.makedirs("./tests/artifacts/", exist_ok=True)

    def tearDown(self):
        shutil.rmtree("./tests/artifacts/")

    def test_rl_control(self):
        horizon = timedelta(hours=24)
        frequency = timedelta(minutes=60)

        data_path = Path(__file__).parent / "data" / "1-LV-rural2--1-sw"
        data_path = data_path.resolve()

        ds1 = CSVDataSource(
            data_path / "LoadProfile.csv",
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
            auto_drop=True,
            resample=frequency,
        )

        ds2 = CSVDataSource(
            data_path / "LoadProfile.csv",
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis"},
            auto_drop=True,
            resample=frequency,
        )

        ds3 = CSVDataSource(
            data_path / "RESProfile.csv",
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "PV3": "p"},
            auto_drop=True,
            resample=frequency,
        ).apply_to_column("p", lambda x: -x)

        dp1 = DataProvider(ds1, LookBackForecaster(frequency=frequency, horizon=horizon))
        dp2 = DataProvider(ds2, LookBackForecaster(frequency=frequency, horizon=horizon))
        dp3 = DataProvider(ds3, LookBackForecaster(frequency=frequency, horizon=horizon))

        n1 = Bus("GenBus", {"p": [-50, 50], "q": [-50, 50], "v": [0.95, 1.05], "d": [-15, 15]})

        r1 = RenewableGen("PV1").add_data_provider(dp3)

        d1 = Load("TestLoad1").add_data_provider(dp1)

        e1 = ESS(
            "TestESS1",
            {
                "rho": 0.1,
                "etac": 0.95,
                "etad": 0.95,
                "etas": 0.99,
                "cap": 6.0,
                "p": [-3, 3],
                "q": [0, 0],
                "soc": [0.2 * 6, 0.8 * 6],
                "soc_init": RangeInitializer(0.2 * 6, 0.8 * 6),
            },
        )

        m1 = TradingBus("Trading1").add_data_provider(dp2)

        sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)
        n1.add_node(d1).add_node(e1).add_node(r1)

        agent1 = RLControllerSB3(
            name="agent1",
            safety_layer=ActionProjectionSafetyLayer(penalty_factor=10.0),
        )

        # set up configuration for the PPO algorithm
        alg_config = {}
        alg_config["total_steps"] = 1
        alg_config["algorithm"] = PPO
        alg_config["policy"] = "MlpPolicy"
        alg_config["learning_rate"] = 0.0008
        alg_config["device"] = "cpu"
        alg_config["n_steps"] = int(horizon.total_seconds() // 3600)
        alg_config["batch_size"] = 12

        # set up logger
        log_dir = "./tests/artifacts/test_run/"
        logger = TensorboardLogger(log_dir=log_dir)

        # specify the path where the model should be saved
        model_path = "./tests/artifacts/saved_models/my_model"
        train_seed = 1

        runner = SingleAgentTrainer(
            sys=sys,
            global_controller=agent1,
            wrapper=SingleAgentWrapper,
            alg_config=alg_config,
            forecast_horizon=horizon,
            control_horizon=horizon,
            logger=logger,
            save_path=model_path,
            seed=train_seed,
            normalize_actions=True,
        )
        runner.run(fixed_start="27.11.2016")

        # Just for demonstration purposes, we show here how to load a pre-trained policy
        # However, in the present case this would not be necessary, since "agent1" has saved the policy after training

        # First, we need to create a new agent and pass the pretrained_policy_path from which to load the neural network
        # params.
        agent2 = RLControllerSB3(
            name="pretrained_agent",
            safety_layer=ActionProjectionSafetyLayer(penalty_factor=10.0),
            pretrained_policy_path=model_path,
        )

        # The deployment runner has to be instantiated with the same arguments used during training
        # The runner will automatically recognize that it has to load the policy for agent2
        eval_seed = 5
        rl_model_history = ModelHistory([sys])
        rl_deployer = DeploymentRunner(
            sys=sys,
            global_controller=agent2,
            alg_config=alg_config,
            wrapper=SingleAgentWrapper,
            forecast_horizon=horizon,
            control_horizon=horizon,
            history=rl_model_history,
            seed=eval_seed,
        )
        # Finally, we can simulate the system with the trained controller for the given day
        rl_deployer.run(n_steps=1, fixed_start="27.11.2016")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
