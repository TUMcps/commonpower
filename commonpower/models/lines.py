"""
Collection of line models.
"""
from __future__ import annotations

import pyomo.environ as pyo

from commonpower.core import Line
from commonpower.modelling import ElementTypes as et
from commonpower.modelling import ModelElement


class BasicLine(Line):
    """
    Basic line model.
    Defines model elements current, active power, conductance, and susceptance.
    """

    CLASS_INDEX = "lb"

    @classmethod
    def _get_model_elements(cls) -> list[ModelElement]:
        model_elements = [
            ModelElement("I", et.VAR, "current"),
            ModelElement("p", et.VAR, "active power"),
            ModelElement("G", et.CONSTANT, "conductance", pyo.NonNegativeReals),
            ModelElement("B", et.CONSTANT, "susceptance", pyo.NonNegativeReals),
        ]
        return model_elements
