import functools
import os
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427


def rsetattr(obj, attr, val):
    """
    Recursive version of setattr() capable of setting an attribute of a nested subobject.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Recursive version of getattr() capable of getting an attribute of a nested subobject.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    """
    Recursive version of hasattr() capable of checking an attribute of a nested subobject.
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def to_datetime(
    *args: Union[List[str], str, List[datetime], datetime]
) -> Union[List[str], str, List[datetime], datetime]:
    """
    Converts the given arguments to datetime objects via pandas.
    """
    # dts = [datetime.strptime(s, self.datetime_format) if isinstance(s, str) else s for s in args]
    dts = [pd.to_datetime(s, dayfirst=True) if isinstance(s, str) else s for s in args]
    return dts if len(args) > 1 else dts[0]


def get_adjusted_cost(hist, entity):
    from commonpower.core import System

    if isinstance(entity, System):
        costs = hist.filter_for_entities(entity, False).filter_for_element_names("cost").history
        output = [c[1]["cost"][0] for c in costs]
        terminal_cost = np.sum(costs[-1][1]["cost"])
    else:
        costs = hist.filter_for_entities(entity, True).filter_for_element_names("cost").history
        cost_id = ".".join([entity.id, 'cost'])
        output = [c[1][cost_id][0] for c in costs]
        terminal_cost = np.sum(costs[-1][1][cost_id])
    output[-1] = terminal_cost
    return output


def guaranteed_path(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return path
