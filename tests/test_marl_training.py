from commonpower.control.logging.loggers import *
from pathlib import Path
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.core import System
from commonpower.models.components import *
from commonpower.models.busses import *
from commonpower.models.powerflow import *
from commonpower.data_forecasting import *
from commonpower.utils.param_initialization import *
from commonpower.control.controllers import RLControllerMA, OptimalController
from commonpower.control.logging.callbacks import *
from commonpower.control.wrappers import MultiAgentWrapper
from commonpower.control.runners import MAPPOTrainer, DeploymentRunner
from commonpower.modelling import ModelHistory
import unittest
import shutil


class TestControl(unittest.TestCase):
    def setUp(self):
        os.makedirs("./tests/artifacts/", exist_ok=True)

    def tearDown(self):
        shutil.rmtree("./tests/artifacts/")

    def test_marl_training(self):
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
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
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

        dp1 = DataProvider(
            ds1, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon)
        )
        dp2 = DataProvider(
            ds2, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon)
        )
        dp3 = DataProvider(
            ds3, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon)
        )

        n1 = RTPricedBusLinear(
            "MultiFamilyHouse",
            {
                "p": [-50, 50],
                "q": [-50, 50],
                "v": [0.95, 1.05],
                "d": [-15, 15]
            }
        ).add_data_provider(dp2)

        n2 = RTPricedBusLinear(
            "MultiFamilyHouse_2",
            {
                "p": [-50, 50],
                "q": [-50, 50],
                "v": [0.95, 1.05],
                "d": [-15, 15]
            }
        ).add_data_provider(dp2)

        # components
        # energy storage sytem
        capacity = 3  # kWh
        e1 = ESSLinear(
            "ESS1",
            {
                "rho": 0.1,
                "p": [-1.5, 1.5],
                "q": [0, 0],
                "soc": [0.2 * capacity, 0.8 * capacity],
                "soc_init": RangeInitializer(0.2 * capacity, 0.8 * capacity),
            },
        )

        capacity_2 = 6
        e2 = ESSLinear(
            "ESS1",
            {
                "rho": 0.1,
                "p": [-3, 3],
                "q": [0, 0],
                "soc": [0.2 * capacity_2, 0.8 * capacity_2],
                "soc_init": RangeInitializer(0.2 * capacity_2, 0.8 * capacity_2),
            },
        )

        # photovoltaic with generation data
        r1 = RenewableGen("PV1").add_data_provider(dp3)

        # static load with data source
        d1 = Load("Load1").add_data_provider(dp1)
        d2 = Load("Load1").add_data_provider(dp1)

        # external grid
        n999 = ExternalGrid("ExternalGrid")

        # we first have to add the nodes to the system
        # and then add components to the node in order to obtain a tree-like structure
        sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(n2).add_node(n999)

        # add components to nodes
        n1.add_node(d1).add_node(e1).add_node(r1)
        n2.add_node(d2).add_node(e2)

        # show system structure:
        sys.pprint()

        # algorithm configuration
        all_args_dict = {
            "algorithm_name": "mappo",
            "seed": 1,
            "cuda": False,
            "cuda_deterministic": True,
            "n_training_threads": 1,
            "n_rollout_threads": 1,
            "n_eval_rollout_threads": 1,
            "num_env_steps": 24,
            "episode_length": 24,
            "share_policy": True,
            "use_centralized_V": True,
            "hidden_size": 64,
            "layer_N": 1,
            "use_ReLU": True,
            "use_popart": False,
            "use_valuenorm": True,
            "use_feature_normalization": True,
            "use_orthogonal": True,
            "gain": 0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy": True,
            "recurrent_N": 1,
            "data_chunk_length": 10,
            "lr": 0.0005,
            "critic_lr": 0.0005,
            "opti_eps": 1e-05,
            "weight_decay": 0,
            "ppo_epoch": 15,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "num_mini_batch": 1,
            "entropy_coef": 0.01,
            "value_loss_coef": 1,
            "use_max_grad_norm": True,
            "max_grad_norm": 10.0,
            "use_gae": True,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "use_proper_time_limits": False,
            "use_huber_loss": True,
            "use_value_active_masks": True,
            "use_policy_active_masks": True,
            "huber_delta": 10.0,
            "use_linear_lr_decay": False,
            "log_interval": 1,
            "use_eval": False,
            "eval_interval": 25,
            "eval_episodes": 32,
            "ifi": 0.1,
            # args from Commonpower
            "safety_penalty": 2.0,
        }

        # add controllers
        for i in range(len(sys.nodes) - 1):
            # will also add a controller to households which do not have inputs (e.g., households with only a Load component),
            # but these are disregarded when the system is initialized
            print("test")
            _ = RLControllerMA(
                name=str.join("agent", str(i)),
                safety_layer=ActionProjectionSafetyLayer(penalty_factor=all_args_dict["safety_penalty"]),
            ).add_entity(sys.nodes[i])

        # set up logger
        logger = MARLTensorboardLogger(log_dir="./tests/artifacts/test_run/", callback=MARLBaseCallback)
        # set up trainer
        runner = MAPPOTrainer(
            sys=sys,
            global_controller=OptimalController("global"),
            wrapper=MultiAgentWrapper,
            alg_config=all_args_dict,
            seed=5,
            logger=logger,
        )
        # run training
        runner.run(fixed_start="27.11.2016")

        # deployment
        # load pre-trained policies
        load_path = "./saved_models/test_model"  # default location
        trained_agent_1 = RLControllerMA(
            name="trained_mappo_agent_1",
            safety_layer=ActionProjectionSafetyLayer(penalty_factor=all_args_dict["safety_penalty"]),
            pretrained_policy_path=load_path + "/agent0",
        ).add_entity(sys.nodes[0])
        trained_agent_2 = RLControllerMA(
            name="trained_mappo_agent_2",
            safety_layer=ActionProjectionSafetyLayer(penalty_factor=all_args_dict["safety_penalty"]),
            pretrained_policy_path=load_path + "/agent1",
        ).add_entity(sys.nodes[1])

        sys_history_mappo = ModelHistory([sys])

        runner = DeploymentRunner(
            sys=sys,
            global_controller=OptimalController("global"),
            alg_config=all_args_dict,
            wrapper=MultiAgentWrapper,
            history=sys_history_mappo,
            seed=1,
        )
        runner.run(n_steps=2, fixed_start="27.11.2016")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
