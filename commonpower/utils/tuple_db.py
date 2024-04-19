import hashlib
import json
from typing import List, Union

from pydantic import BaseModel


class RLTuple(BaseModel):
    observation: Union[list, None]
    action: list
    reward: float
    terminal: bool
    timeout: bool


class TupleItem(BaseModel):
    run_id: str
    tuple: RLTuple


class RecordedRun(BaseModel):
    run_id: str
    scenario_id: str
    config: dict


class TupleDB:
    """
    Base class for tuple database.
    """

    def __init__(self) -> None:

        self.current_run_id: str = None

    def create_run(self, scenario_id: str, config: dict, seed: int):
        self.current_run_id = hashlib.sha256(
            (json.dumps(config, sort_keys=True) + scenario_id + str(seed)).encode()
        ).hexdigest()

        self._create_run(run=RecordedRun(run_id=self.current_run_id, scenario_id=scenario_id, config=config))

    def record_tuples(self, tuples: List[RLTuple]):
        self._record_tuples(tuples=[TupleItem(run_id=self.current_run_id, tuple=t) for t in tuples])

    def list_runs(self) -> List[RecordedRun]:
        raise NotImplementedError()

    def get_tuples(self, filters: dict = {}) -> List[TupleItem]:
        raise NotImplementedError()

    def _create_run(self, run: RecordedRun):
        raise NotImplementedError()

    def _record_tuples(self, tuples: List[TupleItem]):
        raise NotImplementedError()


class MongoTupleDB(TupleDB):
    def __init__(self, db_url: str = "mongodb://localhost:27017/", db_name: str = "tuple_db"):
        from pymongo import MongoClient

        self.db = MongoClient(db_url)[db_name]

    def list_runs(self) -> List[RecordedRun]:
        runs = self.db.runs.find()
        return [RecordedRun(**run) for run in runs]

    def get_tuples(self, filters: dict = {}) -> List[TupleItem]:
        tuples = self.db.tuples.find(filters)
        return [TupleItem(**t) for t in tuples]

    def _create_run(self, run: RecordedRun):
        self.db.runs.insert_one(run.model_dump())

    def _record_tuples(self, tuples: List[TupleItem]):
        self.db.tuples.insert_many([t.model_dump() for t in tuples])


class LocalFileTupleDB(TupleDB):
    """
    Here we store everything in local files.
    They are written and read line by line.
    """

    def __init__(self, base_db_name: str = "."):
        self.db = base_db_name
        self.runs_file = base_db_name + "_runs.txt"
        self.tuples_file = base_db_name + "_tuples.txt"

    def list_runs(self) -> List[RecordedRun]:
        with open(self.runs_file, 'r') as file:
            runs = file.readlines()
        return [RecordedRun(**json.loads(run)) for run in runs]

    def get_tuples(self, filters: dict = {}) -> List[TupleItem]:
        with open(self.tuples_file, 'r') as file:
            tuples = file.readlines()
        return [TupleItem(**json.loads(t)) for t in tuples]

    def _create_run(self, run: RecordedRun):
        with open(self.runs_file, 'a') as file:
            file.write(json.dumps(run.model_dump()) + '\n')

    def _record_tuples(self, tuples: List[TupleItem]):
        with open(self.tuples_file, 'a') as file:
            for t in tuples:
                file.write(json.dumps(t.model_dump()) + '\n')
