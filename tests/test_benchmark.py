from datetime import timedelta
import os
import shutil
import unittest
from pathlib import Path

from commonpower.benchmark.benchmark import Benchmark
from commonpower.data_forecasting.base import DataProvider
from commonpower.data_forecasting.data_sources import CSVDataSource, ConstantDataSource
from commonpower.data_forecasting.forecasters import PersistenceForecaster
from commonpower.modeling.history import ModelHistory
from commonpower.core import System
from commonpower.control.controllers import OptimalController
from commonpower.control.runners import DeploymentRunner
from commonpower.benchmark.storage import BenchmarkStorageFactory
from commonpower.models.buses import ExternalGrid, RTPricedBus
from commonpower.models.components import ESSLinear, Load, RenewableGen
from commonpower.models.powerflow import PowerBalanceModel
from commonpower.modeling.param_initialization import RangeInitializer

@unittest.skip("Does not work on all platforms (TODO)")
class TestBenchmark(unittest.TestCase):
    """
    Unit test class for the Benchmark module.

    This class contains test cases for storing and loading benchmark objects,
    as well as checking for duplicate storage in MongoDB.
    """

    @classmethod
    def setUpClass(cls):
        cls.storage_path = Path(__file__).parent / "artifacts"
        cls.storage_path.mkdir(exist_ok=True)

        # Environment variables
        """ os.environ["MONGO_HOST"] = "localhost"
        os.environ["MONGO_PORT"] = "27017"
        os.environ["MONGO_DB_NAME"] = "commonpower"
        os.environ["MONGO_COLLECTION_NAME"] = "benchmark_tests" """

    @classmethod
    def tearDownClass(cls):
        # Delete temporary storage path
        shutil.rmtree(cls.storage_path)

        # Delete environment variables
        """ del os.environ["MONGO_HOST"]
        del os.environ["MONGO_PORT"]
        del os.environ["MONGO_DB_NAME"]
        del os.environ["MONGO_COLLECTION_NAME"] """

    def set_up_scenario(self):
        """
        Set up the scenario for the benchmark test.

        Returns:
            sys (System): The system object representing the power system.
            oc_deployer (DeploymentRunner): The deployment runner for the optimal controller.
            test_day (str): The test day for evaluation.
        """
        # Set up scenario
        current_path = Path(__file__).parent
        data_path = current_path / "data" / "benchmark_data"
        data_path = data_path.resolve()

        date_format = "%Y-%m-%d %H:%M:00"

        # Data source (ds) for active power(p) of a household
        p_load_ds = CSVDataSource(
            data_path / 'ICLR_load.csv', datetime_format=date_format, resample=timedelta(minutes=60)
        )

        # We neglect reactive power (q) during this tutorial
        q_load_ds = ConstantDataSource(
            {"q": 0.0}, date_range=p_load_ds.get_date_range(), frequency=timedelta(minutes=60)
        )

        # Data source for electricity prices
        buying_price_ds = CSVDataSource(
            data_path / 'ICLR_prices.csv', datetime_format=date_format, resample=timedelta(minutes=60)
        )

        # Data source for selling prices of electricity
        selling_price_ds = ConstantDataSource(
            {"psis": 0.1197}, date_range=buying_price_ds.get_date_range(), frequency=timedelta(minutes=60)
        )

        # Data source for PV generation
        pv_ds = CSVDataSource(
            data_path / 'ICLR_pv.csv', datetime_format=date_format, resample=timedelta(minutes=60)
        ).apply_to_column("p", lambda x: -x)

        forecast_frequency = timedelta(minutes=60)
        forecast_length = 12  # [time steps] --> since frequency is one hour, this is 12h
        forecast_horizon = timedelta(hours=forecast_length)

        forecaster = PersistenceForecaster(
            frequency=forecast_frequency, horizon=forecast_horizon, look_back=timedelta(hours=24)
        )

        load_p_dp = DataProvider(p_load_ds, forecaster)  # [kW]
        load_q_dp = DataProvider(q_load_ds, forecaster)  # [kVA]
        price_dp = DataProvider(buying_price_ds, forecaster)  # [€]
        selling_price_dp = DataProvider(selling_price_ds, forecaster)  # [€]
        pv_dp = DataProvider(pv_ds, forecaster)  # [kW]

        n1 = RTPricedBus("MultiFamilyHouse", {'p': (-50, 50), 'q': (-50, 50), 'v': (0.95, 1.05), 'd': (-15, 15)})
        n1.add_data_provider(price_dp).add_data_provider(selling_price_dp)

        m1 = ExternalGrid("ExternalGrid")
        # photovoltaic with generation data
        r1 = RenewableGen("PV1").add_data_provider(pv_dp)

        # static load with data source
        d1 = Load("Load1").add_data_provider(load_p_dp).add_data_provider(load_q_dp)

        capacity = 5  # kWh
        e1 = ESSLinear(
            "ESS1",
            {
                'p': (-1.5, 1.5),  # active power limits
                'q': (0, 0),  # reactive power limits
                'soc': (0.1 * capacity, 0.9 * capacity),  # soc limits
                "soc_init": RangeInitializer(
                    0.2 * capacity, 0.8 * capacity
                ),  # initial soc at the start of simulation; will be sampled from [1, 4]
            },
        )

        # add components to the household
        n1.add_node(d1).add_node(r1).add_node(e1)

        # create the system and add top-level busses
        sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)

        # Add controller
        opt_controller = OptimalController("opt_ctrl")
        opt_controller.add_entity(n1)

        # Create deployment runner
        oc_history = ModelHistory([sys])
        test_day = "20.11.2016"
        eval_seed = 5
        oc_deployer = DeploymentRunner(sys=sys, horizon=forecast_horizon, history=oc_history, seed=eval_seed)

        return sys, oc_deployer, test_day

    def test_file_store_and_load(self):
        """
        Test case for storing and loading a benchmark object from a file.
        This test case ensures that the benchmark object can be successfully stored and loaded from a file,
        and that the loaded benchmark retains the same properties as the original benchmark.
        """
        sys, oc_deployer, test_day = self.set_up_scenario()

        # Create and store a benchmark object
        storage = BenchmarkStorageFactory.get_storage("file")
        benchmark = Benchmark("BenchmarkTest", sys, storage, oc_deployer, n_steps=24, fixed_start=test_day)
        benchmark.store(self.storage_path)

        # Check if the file exists
        file_path = self.storage_path / "BenchmarkTest.json"
        self.assertTrue(file_path.exists())

        # Check if the file is not empty
        self.assertTrue(file_path.stat().st_size > 0)

        # Load the benchmark object from the file
        loaded_benchmark: Benchmark = Benchmark.load(file_path, storage)

        # Assert that the loaded benchmark is of the same type as the original benchmark
        self.assertIsInstance(loaded_benchmark, Benchmark)

        # Assert that the loaded benchmark has the same name as the original benchmark
        self.assertEqual(loaded_benchmark.benchmark_name, benchmark.benchmark_name)

        loaded_benchmark.run()

    @unittest.skip("MongoDB storage is not supported in CI")
    def test_mongo_store_and_load(self):
        """
        Test case for storing and loading a benchmark object using MongoDB storage.
        This test ensures that the benchmark object can be successfully stored, loaded, and executed using MongoDB storage.
        """
        sys, oc_deployer, test_day = self.set_up_scenario()

        benchmark_name = "BenchmarkTest"
        # Create and store a benchmark object
        storage = BenchmarkStorageFactory.get_storage("mongo")
        benchmark = Benchmark(benchmark_name, sys, storage, oc_deployer, n_steps=24, fixed_start=test_day)
        benchmark.store(Path.cwd())

        # Load the benchmark object from the database
        loaded_benchmark: Benchmark = Benchmark.load(benchmark_name, storage)

        self.assertIsInstance(loaded_benchmark, Benchmark)
        self.assertEqual(loaded_benchmark.scenario_hash, benchmark.scenario_hash)

        # Delete the benchmark from the database after loading it
        storage.delete(benchmark.scenario_hash)

        # Run the loaded benchmark
        loaded_benchmark.run()
    
    @unittest.skip("MongoDB storage is not supported in CI")
    def test_mongo_duplicate(self):
        """
        Test case for checking duplicate storage of a benchmark in MongoDB.

        This test case verifies that attempting to store a benchmark with the same scenario hash
        in MongoDB raises a ValueError, indicating that the benchmark already exists in the database.
        """
        sys, oc_deployer, test_day = self.set_up_scenario()

        benchmark_name = "DuplicationTest"
        # Create and store a benchmark object
        storage = BenchmarkStorageFactory.get_storage("mongo")
        benchmark = Benchmark(benchmark_name, sys, storage, oc_deployer, n_steps=24, fixed_start=test_day)
        benchmark.store(Path.cwd())

        self.assertRaises(ValueError, benchmark.store, Path.cwd())

        # Delete the benchmark from the database after loading it
        storage.delete(benchmark.scenario_hash)


if __name__ == "__main__":
    unittest.main()
