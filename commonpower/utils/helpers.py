import functools
from datetime import datetime
from typing import List, Union

import pandas as pd
from pyomo.core import ConcreteModel

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


def model_root(model: ConcreteModel) -> ConcreteModel:
    """
    Returns the root model of the given model by recursively calling model.parent_block().

    Args:
        model (ConcreteModel): Model.

    Returns:
        ConcreteModel: Root model.
    """

    def get_root(model: ConcreteModel) -> ConcreteModel:
        root = model
        parent = model.parent_block()
        if parent is not None:
            root = get_root(parent)
        return root

    return get_root(model)
