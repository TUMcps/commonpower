"""
Collection of bus models.
"""
from __future__ import annotations

from typing import List

import pyomo.environ as pyo
from pyomo.core import ConcreteModel, Expression, quicksum

from commonpower.core import Bus, Node
from commonpower.modelling import ElementTypes as et
from commonpower.modelling import MIPExpressionBuilder, ModelElement
from commonpower.utils.cp_exceptions import EntityError


class OptSelfSufficiencyNode(Bus):
    """
    Class for creating a household that optimizes its self-sufficiency, i.e., aims at importing as little power from the
    grid as possible (we do not currently consider the reactive power).
    """

    def cost_fcn(self, model: ConcreteModel, t: int) -> Expression:
        """
        Defines a cost function that contains the costs from the household components (self.nodes) plus the total power
        of the household to maximize self-sufficiency. Since the total power p is positive if the household has to
        import power from the grid, we minimize p.
        """
        if self.nodes:
            grid_import_cost = self.get_pyomo_element("p", model)
            return quicksum([n.cost_fcn(model, t) for n in self.nodes]) + grid_import_cost[t]
        else:
            raise EntityError(self, "Cannot define self-sufficiency cost function for an entity without components")


class RTPricedBus(Bus):
    """
    Bus which can directly trade its energy in real-time in stand-alone mode.
    It can also be child of a StructureNode (e.g., energy community, P2P market).
    In that case, the parent structure determines the cost of the bus.

    .. runblock:: pycon

        >>> from commonpower.models.busses import RTPricedBus
        >>> RTPricedBus.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = super()._get_model_elements()

        # additionally define buying and selling price
        model_elements += [
            ModelElement("psib", et.DATA, "buying price", pyo.Reals),
            ModelElement("psis", et.DATA, "selling price", pyo.Reals),
        ]

        return model_elements

    def __init__(self, name: str, config: dict = {}) -> None:
        super().__init__(name, config)
        self.stand_alone = True  # indicates if the bus is child of a StructureNode (energy community, P2P market)

    def set_as_structure_member(self) -> None:
        """
        Sets a flag indicating that the bus is a member of some structure (e.g., energy community, P2P market).
        """
        self.stand_alone = False

    def set_as_stand_alone(self) -> None:
        """
        Sets a flag indicating that the bus is stand-alone.
        """
        self.stand_alone = True

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Sets a binary buying indicator. \\

        .. math::
            p_{eb} = \\left\\{
            \\begin{array}{ll}
            1 & p \\geq 0 \\\\
            0 & \\, \\textrm{otherwise} \\\\
            \\end{array}
            \\right.
        """
        model_elements = super()._get_additional_constraints()  # fetch internal power balance constraints

        if self.stand_alone is True:
            mb = MIPExpressionBuilder(self)

            mb.from_geq("p", 0, "p_eb", is_new=True)

            return model_elements + mb.model_elements
        else:
            return model_elements

    def cost_fcn(self, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = \\sum_{i \\in components} cost_i + p * psib * p_{eb} + p * psis * (1 - p_{eb})
        """
        if self.nodes:
            if self.stand_alone is True:
                return (
                    quicksum([n.cost_fcn(model, t) for n in self.nodes])
                    + (
                        self.get_pyomo_element("p", model)[t]
                        * (1 - self.get_pyomo_element("p_eb", model)[t])
                        * self.get_pyomo_element("psis", model)[t]
                        * self.tau
                    )
                    + (
                        self.get_pyomo_element("p", model)[t]
                        * self.get_pyomo_element("p_eb", model)[t]
                        * self.get_pyomo_element("psib", model)[t]
                        * self.tau
                    )
                )
            else:
                return quicksum([n.cost_fcn(model, t) for n in self.nodes])
        else:
            return 0.0


class RTPricedBusLinear(Bus):
    """
    RTPricedBus which assumes selling and buying prices are identical.

    .. runblock:: pycon

        >>> from commonpower.models.busses import RTPricedBusLinear
        >>> RTPricedBusLinear.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = super()._get_model_elements()

        model_elements += [ModelElement("psi", et.DATA, "market price", pyo.Reals)]

        return model_elements

    def cost_fcn(self, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = \\sum_{i \\in components} cost_i + p * psi
        """
        if self.nodes:
            return quicksum([n.cost_fcn(model, t) for n in self.nodes]) + (
                self.get_pyomo_element("p", model)[t] * self.get_pyomo_element("psi", model)[t] * self.tau
            )
        else:
            return 0.0


class TradingBus(Bus):
    """
    Bus which trades energy with an external market.

    .. runblock:: pycon

        >>> from commonpower.models.busses import TradingBus
        >>> TradingBus.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power", bounds=(-1e6, 1e6)),
            ModelElement("q", et.VAR, "reactive power", bounds=(-1e6, 1e6)),
            ModelElement("v", et.VAR, "voltage magnitude", bounds=(0.9, 1.1)),
            ModelElement("d", et.VAR, "voltage angle", bounds=(-15, 15)),
            ModelElement("psib", et.DATA, "buying price", pyo.Reals),
            ModelElement("psis", et.DATA, "selling price", pyo.Reals),
        ]

        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Sets a binary selling indicator. \\

        .. math::
            p_{es} = \\left\\{
            \\begin{array}{ll}
            1 & p \\geq 0 \\\\
            0 & \\, \\textrm{otherwise} \\\\
            \\end{array}
            \\right.
        """
        mb = MIPExpressionBuilder(self)

        mb.from_geq("p", 0, "p_es", is_new=True)

        return mb.model_elements

    def cost_fcn(self, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = -p * psis * p_{es} -p * psib * (1 - p_{es})
        """
        return -(
            self.get_pyomo_element("p", model)[t]
            * (1 - self.get_pyomo_element("p_es", model)[t])
            * self.get_pyomo_element("psib", model)[t]
            * self.tau
        ) - (
            self.get_pyomo_element("p", model)[t]
            * self.get_pyomo_element("p_es", model)[t]
            * self.get_pyomo_element("psis", model)[t]
            * self.tau
        )

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        # overwrites super(), because otherwise p=0 would be enforced
        return []

    def add_node(self, node: Node) -> Node:
        raise NotImplementedError("Trading busses cannot have sub-nodes")


class TradingBusLinear(Bus):
    """
    TradingBus which assumes selling and buying prices are identical.

    .. runblock:: pycon

        >>> from commonpower.models.busses import TradingBusLinear
        >>> TradingBusLinear.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power", bounds=(-1e6, 1e6)),
            ModelElement("q", et.VAR, "reactive power", bounds=(-1e6, 1e6)),
            ModelElement("v", et.VAR, "voltage magnitude", bounds=(0.9, 1.1)),
            ModelElement("d", et.VAR, "voltage angle", bounds=(-15, 15)),
            ModelElement("psi", et.DATA, "market price", pyo.Reals),
        ]

        return model_elements

    def cost_fcn(self, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = -p * psi
        """
        return -(self.get_pyomo_element("p", model)[t] * self.get_pyomo_element("psi", model)[t] * self.tau)

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        # overwrites super(), because otherwise p=0 would be enforced
        return []

    def add_node(self, node: Node) -> Node:
        raise NotImplementedError("Trading busses cannot have sub-nodes")


class ExternalGrid(Bus):
    """
    Bus with a connection to an external grid.
    Does not have costs or dynamics, merely relevant for power flow calculations.

    .. runblock:: pycon

        >>> from commonpower.models.components import ExternalGrid
        >>> ExternalGrid.info()

    """

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        # overwrites super(), because otherwise p=0 would be enforced
        return []

    def add_node(self, node: Node) -> Node:
        raise NotImplementedError("External grid nodes cannot have sub-nodes")
