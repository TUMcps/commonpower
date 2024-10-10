"""
Module for everything related to benchmarking.
"""

import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import jsonpickle
import pkg_resources

from commonpower.benchmark.storage import BenchmarkStorage
from commonpower.control.runners import BaseRunner, BaseTrainer
from commonpower.core import System


def _get_library_version(package_name: str) -> str:
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound as e:
        raise ImportError(f"Package {package_name} not found. Please install it using pip.") from e


class Benchmark:
    """
    Benchmark class for creating reusable benchmark scenarios.

    Args:
        name (str): The name of the benchmark scenario.
        system_node (System): The system node object representing the system configuration.
        storage (BenchmarkStorage): The storage object used to store the benchmark scenario.
        deployment_runner (BaseRunner): The deployment runner object used to run the scenario.
        trainer (Optional[BaseTrainer]): The trainer object used for training (optional).
        n_steps (int): The number of steps to run the benchmark scenario (default: 24).
        fixed_start (datetime): The fixed start time for the scenario (default: None).
        global_seed (Optional[int]): The global seed for random number generation (optional).

    Methods:
        store(storage_path: Path): Write the benchmark scenario to storage.
        load(path: Path, storage: BenchmarkStorage) -> Benchmark: Load an existing benchmark scenario from storage.
        run(): Run the benchmark scenario with the predefined configuration.

    """

    def __init__(
        self,
        name: str,
        system_node: System,
        storage: BenchmarkStorage,
        deployment_runner: BaseRunner,
        trainer: Optional[BaseTrainer] = None,
        n_steps: int = 24,
        fixed_start: datetime = None,
        global_seed: Optional[int] = None,
    ) -> None:
        self.benchmark_name = name
        self.n_steps = n_steps
        self.fixed_start = fixed_start
        self.global_seed = global_seed

        self.commonpower_version = _get_library_version("commonpower")

        self.system_node = system_node
        self.storage = storage
        self.deployment_runner = deployment_runner
        self.trainer = trainer

        self.scenario_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        # Compute a hash for the current benchmark scenario
        return hashlib.sha256(
            pickle.dumps(
                [
                    self.system_node,
                    self.deployment_runner,
                    self.trainer,
                    self.n_steps,
                    self.fixed_start,
                    self.global_seed,
                    self.commonpower_version,
                ]
            )
        ).hexdigest()

    def store(self, storage_path: Path) -> None:
        """
        Write the benchmark scenario to storage using the previously specified storage provider.

        Args:
            storage_path (Path): The path to the storage location. This depends on the storage type.\
                For file storage, this is the file path.\
                For MongoDB storage, this is not required.

        Raises:
            ValueError: If the scenario was modified after creation.

        """
        data_source_folder = storage_path / "data_sources"
        # Check if the data source folder exists
        if not data_source_folder.exists():
            data_source_folder.mkdir(parents=True)
        # Save all data sources to storage
        self.storage.handle_data_sources(self.system_node, data_source_folder)

        # Serialize the benchmark scenario
        serialized_system_scenario = jsonpickle.encode(self, indent=4, warn=True)

        file_name = self.benchmark_name.replace(" ", "_") + ".json"
        # Check hash in case the scenario was modified after benchmark creation
        if self.scenario_hash != self._compute_hash():
            raise ValueError("Scenario was modified after creation. Please create a new benchmark.")

        self.storage.save(file_name, serialized_system_scenario, storage_path)

    @classmethod
    def load(cls, uri: Path, storage: BenchmarkStorage) -> "Benchmark":
        """
        Load an existing benchmark scenario from storage.
        This creates a new instance of the Benchmark class which can be used to run the scenario.

        Args:
            uri (Path): The uri to the stored benchmark scenario. This depends on the storage type.\
                For file storage, this is the file path.\
                For MongoDB storage, this is the hash of the benchmark scenario.
            storage (BenchmarkStorage): The storage object used to load the benchmark scenario.

        Returns:
            Benchmark: The loaded benchmark scenario.

        """
        result = storage.load(uri)
        result = jsonpickle.decode(result)

        if result.scenario_hash != result._compute_hash():
            print("Warning: The benchmark's stored hash does not match with the scenario anymore!")

        return result

    def run(self) -> None:
        """
        Run the benchmark scenario with the predefined configuration.
        """
        if self.trainer is not None:
            self.trainer.run(fixed_start=self.fixed_start)

        self.deployment_runner.run(n_steps=self.n_steps, fixed_start=self.fixed_start)

    def __getstate__(self) -> object:
        # Remove storage object from the state, since it cannot be pickled when using a DB
        obj = self.__dict__.copy()
        obj.pop("storage")
        return obj


class TimedeltaHandler(jsonpickle.handlers.BaseHandler):
    """
    A custom handler for serializing and deserializing\
        timedelta objects to and from JSON.

    This handler is used by the jsonpickle library to handle serialization and deserialization
    of timedelta objects.

    Attributes:
        None

    Methods:
        flatten(obj, data): Flattens the timedelta object into a dictionary representation.
        restore(obj): Restores the object from a dictionary representation.

    Usage:
        This handler should be registered with the jsonpickle library to enable serialization
        and deserialization of timedelta objects.

    """

    def flatten(self, obj, data: dict):
        return data | {"days": obj.days, "seconds": obj.seconds, "microseconds": obj.microseconds}

    def restore(self, obj):
        return timedelta(days=obj["days"], seconds=obj["seconds"], microseconds=obj["microseconds"])


TimedeltaHandler.handles(timedelta)
