"""
Collection of bus models.
"""
from __future__ import annotations

from typing import List

import pyomo.environ as pyo
from pyomo.core import ConcreteModel, Expression, quicksum

from commonpower.core import Bus, Node, StructureNode
from commonpower.modeling.base import ElementTypes as et
from commonpower.modeling.base import ModelElement
from commonpower.modeling.mip_builder import MIPExpressionBuilder
from commonpower.modeling.robust_cost import CostScenario
from commonpower.utils.cp_exceptions import EntityError


class OptSelfSufficiencyNode(Bus):
    """
    Class for creating a household that optimizes its self-sufficiency, i.e., aims at importing as little power from the
    grid as possible (we do not currently consider the reactive power).
    """

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        Defines a cost function that contains the costs from the household components (self.nodes) plus the total power
        of the household to maximize self-sufficiency. Since the total power p is positive if the household has to
        import power from the grid, we minimize p.
        """
        if self.nodes:
            grid_import_cost = scenario(self, "p", model)
            return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes]) + grid_import_cost[t]
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

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = \\sum_{i \\in components} cost_i + p * psib * p_{eb} + p * psis * (1 - p_{eb})
        """
        if self.nodes:
            if self.stand_alone is True:
                return (
                    quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])
                    + (
                        scenario(self, "p", model)[t]
                        * (1 - scenario(self, "p_eb", model)[t])
                        * scenario(self, "psis", model)[t]
                        * self.tau
                    )
                    + (
                        scenario(self, "p", model)[t]
                        * scenario(self, "p_eb", model)[t]
                        * scenario(self, "psib", model)[t]
                        * self.tau
                    )
                )
            else:
                return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])
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

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = \\sum_{i \\in components} cost_i + p * psi
        """
        if self.nodes:
            if self.stand_alone:
                return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes]) + (
                    scenario(self, "p", model)[t] * scenario(self, "psi", model)[t] * self.tau
                )
            else:
                return quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])
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

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = -p * psis * p_{es} -p * psib * (1 - p_{es})
        """
        return -(
            scenario(self, "p", model)[t]
            * (1 - scenario(self, "p_es", model)[t])
            * scenario(self, "psib", model)[t]
            * self.tau
        ) - (
            scenario(self, "p", model)[t]
            * scenario(self, "p_es", model)[t]
            * scenario(self, "psis", model)[t]
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

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = -p * psi
        """
        return -(scenario(self, "p", model)[t] * scenario(self, "psi", model)[t] * self.tau)

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        # overwrites super(), because otherwise p=0 would be enforced
        return []

    def add_node(self, node: Node) -> Node:
        raise NotImplementedError("Trading busses cannot have sub-nodes")


class CarbonAwareTradingBus(TradingBus):
    """
    Carbon Aware Trading Bus.

    .. runblock:: pycon

        >>> from commonpower.models.busses import CarbonAwareTradingBus
        >>> CarbonAwareTradingBus.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = super()._get_model_elements()

        model_elements += [
            ModelElement("ci", et.DATA, "carbon intensity", pyo.NonNegativeReals),
            ModelElement("a", et.CONSTANT, "cost parameter a", pyo.NonNegativeReals),
            ModelElement("b", et.CONSTANT, "cost parameter b", domain=pyo.NonNegativeIntegers, bounds=(1, 2)),
        ]

        return model_elements

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = (-p * psis * p_{es} -p * psib * (1 - p_{es})) + (a * p^b) / ci

        Note that ci >= 0 for carbon intensity and p represents active power.
        """
        return (
            -(
                scenario(self, "p", model)[t]
                * (1 - scenario(self, "p_es", model)[t])
                * scenario(self, "psib", model)[t]
                * self.tau
            )
            - (
                scenario(self, "p", model)[t]
                * scenario(self, "p_es", model)[t]
                * scenario(self, "psis", model)[t]
                * self.tau
            )
        ) + (scenario(self, "a", model) * scenario(self, "p", model)[t] ** scenario(self, "b", model)) / scenario(
            self, "ci", model
        )[
            t
        ] * self.tau


class CarbonAwareTradingBusLinear(TradingBusLinear):
    """
    Carbon Aware Bus which assumes selling and buying prices are identical.

    .. runblock:: pycon

        >>> from commonpower.models.busses import CarbonAwareTradingBusLinear
        >>> CarbonAwareTradingBusLinear.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = super()._get_model_elements()

        model_elements += [
            ModelElement("ci", et.DATA, "carbon intensity", pyo.NonNegativeReals),
            ModelElement("a", et.CONSTANT, "cost parameter a", pyo.NonNegativeReals),
            ModelElement("b", et.CONSTANT, "cost parameter b", domain=pyo.NonNegativeIntegers, bounds=(1, 2)),
        ]

        return model_elements

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = - p * psi + (a * p^b) / ci)
        """
        return (-scenario(self, "p", model)[t] * scenario(self, "psi", model)[t] * self.tau) + (
            scenario(self, "a", model) * scenario(self, "p", model)[t] ** scenario(self, "b", model)
        ) * self.tau


class ExternalGrid(Bus):
    """
    Bus with a connection to an external grid.
    Does not have costs or dynamics, merely relevant for power flow calculations.

    .. runblock:: pycon

        >>> from commonpower.models.busses import ExternalGrid
        >>> ExternalGrid.info()

    """

    def _get_internal_power_balance_constraints(self) -> List[ModelElement]:
        # overwrites super(), because otherwise p=0 would be enforced
        return []

    def add_node(self, node: Node) -> Node:
        raise NotImplementedError("External grid nodes cannot have sub-nodes")


class EnergyCommunity(StructureNode):
    """
    Structure node representing an energy community.

    .. runblock:: pycon

        >>> from commonpower.models.busses import EnergyCommunity
        >>> EnergyCommunity.info()

    """

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        # define buying and selling price
        model_elements = [
            ModelElement("p_sum", et.VAR, "sum of active power within coalition", pyo.Reals, bounds=[-1e6, 1e6]),
            ModelElement("psib", et.DATA, "buying price", pyo.Reals),
            ModelElement("psis", et.DATA, "selling price", pyo.Reals),
        ]

        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        mb = MIPExpressionBuilder(self)
        mb.from_geq("p_sum", 0, "p_eb", is_new=True)

        def p_sum_fcn(model, t):
            return (
                quicksum([n.get_pyomo_element("p", model)[t] for n in self.nodes])
                == self.get_pyomo_element("p_sum", model)[t]
            )

        p_sum_c = ModelElement("p_sum_c", et.CONSTRAINT, "sum of active power within coalition", expr=p_sum_fcn)

        return [p_sum_c] + mb.model_elements

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int) -> Expression:
        """
        .. math::
            cost = \\sum_{j \\in coalition} \\sum_{i \\in components} cost_ji
            + \\sum_{j \\in coalition} (p_j) * psi
        """
        if self.nodes:
            return (
                quicksum([n.cost_fcn(scenario, model, t) for n in self.nodes])
                + (
                    scenario(self, "p_sum", model)[t]
                    * (1 - scenario(self, "p_eb", model)[t])
                    * scenario(self, "psis", model)[t]
                    * self.tau
                )
                + (
                    scenario(self, "p_sum", model)[t]
                    * scenario(self, "p_eb", model)[t]
                    * scenario(self, "psib", model)[t]
                    * self.tau
                )
            )
        else:
            return 0.0

    def add_node(self, node: Bus) -> Node:
        """
        Adds a subordinate bus.
        The added node's id is set according to its position in the model hierarchy.
        The passed node is flagged as structure member.

        Args:
            node (Bus): Bus istance to add.

        Returns:
            Node: Node instance.
        """

        assert isinstance(node, Bus), "Only buses can be direct children of an energy community"

        super().add_node(node)
        node.set_as_structure_member()
        return self
