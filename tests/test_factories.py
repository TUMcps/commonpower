import unittest

from pathlib import Path
from datetime import timedelta
from numpy.random import uniform, choice
from copy import deepcopy

from commonpower.control.controllers import OptimalController
from commonpower.control.runners import DeploymentRunner
from commonpower.core import System, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.data_forecasting import *
from commonpower.utils.param_initialization import RangeInitializer
from commonpower.extensions.factories import Factory, Sampler


class TestFactories(unittest.TestCase):
    def test_factory(self):
        horizon = timedelta(hours=24)
        frequency = timedelta(minutes=60)

        data_path = Path(__file__).parent / "data" / "1-LV-rural2--1-sw" / "LoadProfile.csv"
        data_path = data_path.resolve()

        ds1 = CSVDataSource(
            data_path,
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "p", "G1-B_qload": "q"},
            auto_drop=True,
            resample=frequency,
        )

        ds2 = CSVDataSource(
            data_path,
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
            auto_drop=True,
            resample=frequency,
        )

        ds3 = CSVDataSource(
            data_path,
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "p_pot", "G1-C_pload": "q_pot"},
            auto_drop=True,
            resample=frequency,
        ).apply_to_column("p_pot", lambda x: -x)

        ds4 = deepcopy(ds3)

        dp1 = DataProvider(ds1, LookBackForecaster(frequency=frequency, horizon=horizon))
        dp2 = DataProvider(ds2, ConstantForecaster(frequency=frequency, horizon=horizon))
        dp3 = DataProvider(
            ds3, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon)
        )
        dp4 = DataProvider(ds3, LookBackForecaster(frequency=frequency, horizon=horizon))

        sys = System(power_flow_model=PowerBalanceModel())

        factory = Factory()

        factory.set_bus_template(
            Bus,
            meta_config={
                "p": Sampler(uniform, low=[-15, 12], high=[-12, 15]),
                "q": Sampler(uniform, low=[-15, 12], high=[-12, 15]),
                "v": Sampler(uniform, low=[0.95, 1.05], high=[0.95, 1.05]),
                "d": Sampler(uniform, low=[-15, 15], high=[-15, 15]),
            },
        )

        factory.add_component_template(Load, 1.0, data_providers=[Sampler(choice, a=[dp1])])

        factory.add_component_template(
            RenewableGenCurtail,
            0.5,
            meta_config={
                "p": Sampler(uniform, low=[-7, 0], high=[-5, 0]),
                "q": Sampler(uniform, low=[0, 0], high=[0, 0]),
            },
            data_providers=[Sampler(choice, a=[dp3, dp4])],
        )

        factory.add_component_template(
            ESSLinear,
            0.5,
            meta_config={
                "rho": 0.1,
                "etac": 0.95,
                "etad": 0.95,
                "etas": 0.99,
                "p": Sampler(uniform, low=[-5, 3], high=[-3, 5]),
                "q": Sampler(uniform, low=[0, 0], high=[0, 0]),
                "soc": Sampler(uniform, low=[0, 5], high=[0.5, 6]),
                "soc_init": [
                    RangeInitializer,
                    {"lb": Sampler(uniform, low=1, high=2), "ub": Sampler(uniform, low=3, high=4)},
                ],
            },
        )

        n0 = Bus("Street1")
        n1 = TradingBusLinear("Trading1").add_data_provider(dp2)
        
        sys.add_node(n0).add_node(n1)

        factory.generate_households(n0, 100)

        runner = DeploymentRunner(sys)
        runner.run(n_steps=1)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
