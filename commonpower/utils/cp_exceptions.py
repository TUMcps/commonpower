from __future__ import annotations

from typing import TYPE_CHECKING

from pyomo.core import ConcreteModel

if TYPE_CHECKING:
    from commonpower.control.controllers import BaseController
    from commonpower.modelling import ModelEntity


class EntityError(Exception):
    def __init__(self, entity: ModelEntity, message: str):
        """
        Exception which gives info of which entity raised it.

        Args:
            entity (ModelEntity): Raising entity.
            message (str): Error message.
        """
        self.entity = entity
        self.message = message

    def __str__(self):
        return f"Error on Node {self.entity.name}: {self.message}"


class ControllerError(Exception):
    def __init__(self, controller: BaseController, message: str):
        """
        Exception which gives info of which controller raised it.

        Args:
            controller (BaseController): Raising controller.
            message (str): Error message.
        """
        self.controller = controller
        self.message = message

    def __str__(self):
        return f"Error in Agent {self.controller.name}: {self.message}"


class InstanceError(Exception):
    def __init__(self, instance: ConcreteModel, message: str):
        """
        Exception which gives info of which pyomo model instance raised it.

        Args:
            instance (ConcreteModel): Rasining model instance.
            message (str): Error message.
        """
        self.instance = instance
        self.message = message

    def __str__(self):
        return f"Error on ConcreteModel instance {self.instance.name}: {self.message}"
