"""
Module for everything related to banchmark storage.
"""

import hashlib
import json
import os
from pathlib import Path

import gridfs
import pandas as pd
from pymongo import MongoClient, errors

from commonpower.core import System
from commonpower.data_forecasting.data_sources import CSVDataSource, PandasDataSource


def _find_dict_path(key, var, path=[]):
    """
    Recursively searches for a given key in a nested dictionary or list and returns the value along with the path to it.

    Args:
        key (any): The key to search for.
        var (dict or list): The dictionary or list to search in.
        path (list, optional): The current path to the current element being searched. Defaults to an empty list.

    Yields:
        tuple: A tuple containing the value found and the path to it.

    """
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v, path
            if isinstance(v, dict):
                for result in _find_dict_path(key, v, path + [k]):
                    yield result
            elif isinstance(v, list):
                for i, d in enumerate(v):
                    for result in _find_dict_path(key, d, path + [k, i]):
                        yield result


class BenchmarkStorage:
    """
    The base class for benchmark storage.

    This class provides an interface for saving and loading benchmark scenarios.
    Subclasses should implement the `save`, `load`, and `handle_data_sources` methods.

    Attributes:
        None
    """

    def __init__(self):
        raise NotImplementedError

    def save(self, benchmark_identifier: str, serialized_scenario: str, uri: str):
        """
        Saves the serialized scenario to the specified URI.

        Args:
            benchmark_identifier (str): The identifier of the benchmark.
            serialized_scenario (str): The serialized scenario to be saved.
            uri (str): The URI where the serialized scenario should be saved.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def load(self, uri: str):
        """
        Loads data from the specified URI.

        Args:
            uri (str): The URI of the data to be loaded.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    def handle_data_sources(self, system_node: System, uri) -> None:
        """
        Handles the data sources for the benchmark scenario.
        This method should be overridden in a subclass to handle the data sources appropriately.

        Args:
            system_node (System): The system node of the benchmark scenario.
            uri (str): The URI of the data sources.

        Returns:
            None
        """
        raise NotImplementedError


class BenchmarkFileStorage(BenchmarkStorage):
    """
    A class that provides methods to save and load benchmark scenarios to/from files.

    Attributes:
        None

    Methods:
        save(benchmark_identifier: str, serialized_scenario: str, uri: Path) -> None:
            Saves the serialized scenario to a file at the specified URI.

        load(uri: Path) -> str:
            Loads a serialized scenario from the file at the specified URI and returns it as a string.

        handle_data_sources(system_node: System, uri: Path) -> None:
            Prepare all data sources for the benchmark scenario.
    """

    def __init__(self):
        pass

    def save(self, benchmark_identifier: str, serialized_scenario: str, uri: Path) -> Path:
        """
        Saves the serialized scenario to a file at the specified filepath (URI).

        Args:
            benchmark_identifier (str): The desired filename for the benchmark.
            serialized_scenario (str): The serialized scenario to be saved, as returned by jsonpickle.
            uri (Path): The URI specifying the location where the file should be saved.

        Returns:
            file_path (Path): The path to the saved file.
        """
        file_path = uri / benchmark_identifier

        with open(file_path, "w", encoding="UTF-8") as f:
            f.write(serialized_scenario)

        return file_path

    def load(self, uri: Path) -> str:
        """
        Loads a serialized scenario from the file at the specified URI and returns it as a string.

        Args:
            uri (Path): The URI specifying the location of the file to be loaded.

        Returns:
            str: The loaded serialized scenario.

        Raises:
            FileNotFoundError: If the data folder does not exist at the expected location.
            ValueError: If the specified benchmark file is not a JSON file.
        """
        # Load the serialized scenario from file
        data_folder = uri.parent / "data_sources"
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder {data_folder} does not exist at expected location.")
        if Path(uri).suffix != ".json":
            raise ValueError(f"File {uri} is not a JSON file.")
        with open(uri, "r", encoding="UTF-8") as f:
            benchmark_json = json.loads(f.read())

        # Patch data sources to point to the correct file path
        dict_paths = list(_find_dict_path("data_source_hash", benchmark_json))
        for data_hash, path in dict_paths:
            d_trav = benchmark_json
            for i, _ in enumerate(path):
                d_trav = d_trav[path[i]]
            data_path = data_folder / f"data_source_{data_hash}.csv"
            d_trav["resampled_data_path"] = data_path.absolute().as_posix()

        return json.dumps(benchmark_json)

    def handle_data_sources(self, system_node: System, uri: Path) -> None:
        """
        Store all data sources used in the system node as CSV files in the specified data folder.

        Args:
            system_node (System): The system node containing the data sources.
            uri (Path): The URI specifying the location where the data sources should be stored.

        Returns:
            None
        """
        # Use set to avoid duplicates
        data_providers = set(system_node.get_all_data_providers())
        for provider in data_providers:
            data_source = provider.data
            # We only have to prepare the Pandas and CSV data sources
            if isinstance(data_source, (PandasDataSource, CSVDataSource)):
                # Store the data source as a CSV file so they can be loaded later
                df = data_source.data
                # Compute hash of the data source first, so we have a unique identifier
                df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
                df.to_csv(uri / f"data_source_{df_hash}.csv")


class BenchmarkMongoStorage(BenchmarkStorage):
    """
    A class that provides storage functionality for benchmarks using MongoDB.
    MongoDB connection information (host, port, username, password) must be provided as environment variables.
    Specifically the following environment variables must be set:
    - MONGO_HOST: The MongoDB host.
    - MONGO_PORT: The MongoDB port.
    - MONGO_USERNAME: The MongoDB username.
    - MONGO_PASSWORD: The MongoDB password.

    Optionally, the following environment variables can be set:
    - MONGO_DB_NAME: The name of the MongoDB database. Default is "commonpower".
    - MONGO_COLLECTION_NAME: The name of the MongoDB collection. Default is "benchmarks".

    Methods:
        save(benchmark_identifier: str, serialized_scenario: str, uri: str) -> None:
            Saves a benchmark to the MongoDB collection.
        load(uri: str) -> str:
            Loads a benchmark from the MongoDB collection.
        handle_data_sources(system_node: System, uri) -> None:
            Handles data sources for a system node and stores them in MongoDB.
        delete(benchmark_hash: str) -> None:
            Deletes a benchmark from the MongoDB collection.
    """

    def __init__(self):
        # Try to get host and port from environment variables
        self.host = os.getenv("MONGO_HOST")
        self.port = os.getenv("MONGO_PORT")
        self.username = os.getenv("MONGO_USERNAME")
        self.password = os.getenv("MONGO_PASSWORD")
        if not self.host or not self.port or not self.username or not self.password:
            raise ValueError(
                """
                MongoDB host, port, username and password must be specified in environment variables
                 (MONGO_HOST, MONGO_PORT, MONGO_USERNAME, MONGO_PASSWORD)
                """
            )
        # Try converting port to int
        try:
            self.port = int(self.port)
        except ValueError as e:
            raise ValueError("MongoDB port must be an integer") from e

        self.db_name = os.getenv("MONGO_DB_NAME", "commonpower")
        self.collection_name = os.getenv("MONGO_COLLECTION_NAME", "benchmarks")

        try:
            client = MongoClient(
                self.host,
                self.port,
                username=self.username,
                password=self.password,
                authSource="admin",
                authMechanism="SCRAM-SHA-256",
            )
        except errors.ConnectionError as e:
            raise ValueError("Failed to connect to MongoDB") from e
        db = client[self.db_name]
        self.collection = db[self.collection_name]

        self.fs = gridfs.GridFS(db)

    def save(self, benchmark_identifier: str, serialized_scenario: str, uri: str) -> None:
        """
        Saves a benchmark to the MongoDB collection.

        Args:
            benchmark_identifier (str): Not used in this implementation but kept for compatibility with the base class.
            serialized_scenario (str): The serialized scenario data returned by jsonpickle.
            uri (str): Not used in this implementation but kept for compatibility with the base class.

        Raises:
            ValueError: If a benchmark with the same hash already exists in MongoDB.
        """
        benchmark_document = json.loads(serialized_scenario)
        # Check if benchmark with the same hash already exists
        existing_doc = self.collection.find_one(
            {"py/state.scenario_hash": benchmark_document["py/state"]["scenario_hash"]}
        )
        if existing_doc is not None:
            raise ValueError(
                f"Benchmark with hash {benchmark_document['py/state']['scenario_hash']} already exists in MongoDB"
            )

        # Set benchmark id to scenario hash
        benchmark_document["_id"] = benchmark_document["py/state"]["scenario_hash"]
        self.collection.insert_one(benchmark_document)

    def load(self, uri: str) -> str:
        """
        Loads a benchmark from the MongoDB collection.

        Args:
            uri (str): The scenario hash of the benchmark to load.

        Returns:
            str: The serialized benchmark document.

        Raises:
            FileNotFoundError: If the benchmark with the given hash is not found\
                  in MongoDB or required data sources are missing.
        """
        doc = self.collection.find_one({"py/state.scenario_hash": uri})

        if doc is None:
            raise FileNotFoundError(f"Benchmark with hash {uri} not found in MongoDB")

        # Download data sources from GridFS in the following
        data_folder = Path("/tmp/data_sources")
        if not data_folder.exists():
            data_folder.mkdir()
        # Iterate over all occurrences of the data_source_hash key in the document
        dict_paths = list(_find_dict_path("data_source_hash", doc))
        for data_hash, path in dict_paths:
            # Find corresponding data source in MongoDB
            d_source = self.fs.find_one({"hash": data_hash})
            data_path = data_folder / f"data_source_{data_hash}.csv"
            if d_source is None:
                raise FileNotFoundError(f"Data source with hash {data_hash} for benchmark {doc} not found in MongoDB")
            # Write data source to temporary file
            with open(data_path, "wb") as f:
                f.write(d_source.read())

            # Update the document to point to the correct file path
            d_trav = doc
            for i, _ in enumerate(path):
                d_trav = d_trav[path[i]]
            d_trav["resampled_data_path"] = data_path.absolute().as_posix()

        doc.pop("_id")
        return json.dumps(doc)

    def handle_data_sources(self, system_node: System, uri=None) -> None:
        """
        Finds all data sources used in the system node and stores them in MongoDB with GridFS.

        Args:
            system_node (System): The system node object.
            uri: Not used in this implementation but kept for compatibility with the base class.

        Raises:
            FileNotFoundError: If a data source for the benchmark is not found in MongoDB.
        """
        # Use set to avoid duplicates
        data_providers = set(system_node.get_all_data_providers())
        for provider in data_providers:
            data_source = provider.data
            # We only have to prepare the Pandas and CSV data sources
            if isinstance(data_source, (PandasDataSource, CSVDataSource)):
                df = data_source.data
                # Compute hash of the data source first, so we have a unique identifier
                df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
                # Write the data source to MongoDB using GridFS
                with self.fs.new_file(filename=f"data_source_{df_hash}.csv", hash=df_hash, encoding="utf-8") as f:
                    f.write(df.to_csv(encoding="utf-8"))

    def delete(self, benchmark_hash: str) -> None:
        """
        Deletes a benchmark from the MongoDB collection.

        Args:
            benchmark_hash (str): The hash of the benchmark.

        Raises:
            FileNotFoundError: If the benchmark with the given hash is not found in MongoDB.
        """
        res = self.collection.delete_one({"py/state.scenario_hash": benchmark_hash})
        if res.deleted_count == 0:
            raise FileNotFoundError(f"Benchmark with hash {benchmark_hash} not found in MongoDB")


class BenchmarkStorageFactory:
    """
    Factory class for creating benchmark storage objects based on the storage type.
    Currently supports two storage types: "file" and "mongo".

    Methods:
        get_storage: Returns a benchmark storage object based on the storage type.

    Usage:
        factory = BenchmarkStorageFactory()
        storage = factory.get_storage("file")
    """

    @staticmethod
    def get_storage(storage_type: str):
        """
        Returns an instance of a storage class based on the given storage type.

        Parameters:
            storage_type (str): The type of storage to use. Valid options are "file" and "mongo".

        Returns:
            An instance of a storage class based on the given storage type.

        Raises:
            ValueError: If an invalid storage type is provided.
        """
        if storage_type == "file":
            return BenchmarkFileStorage()
        elif storage_type == "mongo":
            return BenchmarkMongoStorage()
        else:
            raise ValueError("Invalid storage type")
