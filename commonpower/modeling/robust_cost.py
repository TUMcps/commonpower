"""
Functionality for creating robust cost functions.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Callable

import numpy as np
from pyomo.core import ConcreteModel, ConstraintList, Expression, Param, Var, quicksum

from commonpower.modeling.util import SubscriptableFloat

if TYPE_CHECKING:
    from commonpower.core import Node
    from commonpower.modeling.base import ModelEntity
    from commonpower.modeling.robust_constraints import RobustConstraintBuilder


class CostScenario:
    def __init__(self, element_mapping: dict[str, str] = None):
        """
        Utility class to map variables and parameters to their corresponding scenario variables.

        Args:
            element_mapping (dict[str, str], optional): Mapping of variables and
                parameters to their scenario equivalents.
                If not given, the nominal scenario is assumed. Defaults to None.
        """
        self.element_mapping = element_mapping

    def __call__(self, entity: ModelEntity, element_name: str, model: ConcreteModel) -> Var | Param:
        """
        Retrieves the pyomo element for the given element name.
        It does so across all scenarios defined by the uncertainties in the robust constraint.

        Args:
            element_name (str): Local name of the element.
            model (ConcreteModel): Model instance.

        Returns:
            Var | Param: Pyomo element.
        """
        el_name_in_scenario = self.element_mapping[element_name] if self.element_mapping else element_name
        return entity.get_pyomo_element(el_name_in_scenario, model)


class _CostSignatureExtractor(CostScenario):
    def __init__(self):
        """
        Utility class to extract the signature of a robust cost function from its expression.
        The extractor "fakes" being a cost scenario and stores all element calls when
        an expression that is was passed to is called.
        This way, we obtain all (uncertain) parameters and variables used in the cost expression.
        """
        self.signature = _RobustCostSignature()

    def __call__(self, entity: Node, element_name: str, model: ConcreteModel) -> float:
        """
        This does not return the pyomo element as the ConstraintScenario class does.
        Instead, it stores the element name with the appropriate category in self.signature.

        Args:
            entity (Node): Node instance that is currently being processed.
            element_name (str): Local name of the element.
            model (ConcreteModel): Model instance.

        Returns:
            float: Subscriptable float that "fakes" the behaviour of a model element.
        """

        rc_builder: RobustConstraintBuilder = entity.robust_constraint_builder

        # This will be called multiple times for indexed variables
        if element_name in rc_builder.uncertain_params:
            self.signature.uncertain_params.append(element_name)
        elif element_name in rc_builder.vars:
            self.signature.vars.append(element_name)
        elif element_name in rc_builder.uncertain_vars:
            self.signature.uncertain_vars.append(element_name)
        elif element_name in rc_builder.params:
            self.signature.params.append(element_name)

        self.signature.uncertain_params = list(set(self.signature.uncertain_params))
        self.signature.uncertain_vars = list(set(self.signature.uncertain_vars))
        self.signature.vars = list(set(self.signature.vars))
        self.signature.params = list(set(self.signature.params))

        return SubscriptableFloat(1.0)


class _RobustCostSignature:
    def __init__(self):
        """
        Class to store the signature of a robust cost function.
        The signature includes all variables, parameters, and uncertainties are used in the constraint.
        """
        self.vars: list[str] = []
        self.uncertain_vars: list[str] = []
        self.params: list[str] = []
        self.uncertain_params: list[str] = []

    @classmethod
    def from_fcn(cls, fcn: Callable, horizon: int) -> _RobustCostSignature:
        """
        Extracts the signature of a robust cost function from its expression.

        Args:
            fcn (Callable): Cost function.
        """
        signature_extractor = _CostSignatureExtractor()
        # simulate constraint to extract signature
        for t in range(horizon):
            fcn(signature_extractor, None, t)
        return signature_extractor.signature

    @property
    def n_scenarios(self) -> int:
        """
        Number of scenarios defined by the robust constraint.
        This is based on the number of uncertain variables and parameters.

        Returns:
            int: Number of scenarios.
        """
        return 2**self.n_uncertainties + 1 if self.n_uncertainties > 0 else 1

    @property
    def n_uncertainties(self) -> int:
        """
        Number of uncertainties in the robust constraint, i.e., the sum of uncertain variables and parameters.

        Returns:
            int: Number of uncertainties.
        """
        return len(self.uncertain_vars) + len(self.uncertain_params)


class BaseRobustCost:
    def __init__(
        self,
        discount_factor: float = 1.0,
    ):
        """
        Utility class to create a robust cost function from a given cost function expression.

        Args:
            discount_factor (float, optional): Discount rate to give lower importance to costs
                further in the future. The logic is: sum_t (cost_t * df**t).
                Defaults to 1.0.
        """
        self.discount_factor = np.clip(discount_factor, 0.0, 1.0)

        self.fcn = None
        self.horizon = None
        self.signature = None

    def initialize(self, fcn: Callable, horizon: int):
        """
        Initializes the robust cost function.

        Args:
            fcn (Callable): Cost function. Must take the arguments (scenario, model, t).
            horizon (int): Horizon of the optimization problem.
        """
        self.fcn = fcn
        self.horizon = horizon
        self.signature = _RobustCostSignature.from_fcn(fcn, self.horizon)

    def obj_fcn(self, model: ConcreteModel) -> Expression:
        """
        Creates the objective function for the robust cost.

        Returns:
            Expression: Robust objective function.
        """
        raise NotImplementedError

    def add_additional_constraints(self, model: ConcreteModel) -> None:
        """
        Adds additional constraints for the robust cost function to the given model.

        Args:
            model (ConcreteModel): Model instance.
        """
        return


class NominalCost(BaseRobustCost):
    """
    Cost function only considering the nominal scenario.
    """

    def obj_fcn(self, model: ConcreteModel) -> Expression:
        """
        Creates the objective function for the nominal cost.

        Returns:
            Expression: Nominal objective function.
        """
        return quicksum([self.fcn(CostScenario(), model, t) * self.discount_factor**t for t in range(self.horizon)])


class _ScenarioBasedCost(BaseRobustCost):
    def _scenarios(self) -> list[CostScenario]:
        """
        Generates all possible scenarios based on the uncertainties in the robust constraint.

        Returns:
            list[CostScenario]: List of scenarios.
        """
        scenarios = []
        bound_setups = [tuple(np.repeat("", self.signature.n_uncertainties).tolist())]  # nominal scenario

        if self.signature.n_uncertainties > 0:
            bound_setups += list(itertools.product(["_lb", "_ub"], repeat=self.signature.n_uncertainties))

        for bound_setup in bound_setups:
            element_mapping = {}
            element_mapping.update({el: el + bound_setup[i] for i, el in enumerate(self.signature.uncertain_params)})
            var_elements_offset = len(self.signature.uncertain_params)
            element_mapping.update(
                {el: el + bound_setup[var_elements_offset + i] for i, el in enumerate(self.signature.uncertain_vars)}
            )
            element_mapping.update({el: el for el in self.signature.params + self.signature.vars})
            scenarios.append(CostScenario(element_mapping))

        return scenarios


class WeightedSumRobustCost(_ScenarioBasedCost):
    def __init__(self, discount_factor: float = 1.0, weights: list[float] = None):
        """
        Utility class to create a robust cost function from a given cost function expression.
        The cost is calculated as the weighted sum of the costs over all scenarios.

        Args:
            discount_factor (float, optional): Discount rate to give lower importance to costs
                further in the future. The logic is: sum_t (cost_t * df**t).
                Defaults to 1.0.
            weights (list[float], optional): Weights for each scenario.
                If not given, all scenarios are weighted equally. Defaults to None.
        """
        super().__init__(discount_factor)
        self.weights = weights

    def obj_fcn(self, model: ConcreteModel) -> Expression:
        """
        Creates the objective function for the robust cost.

        Returns:
            Expression: Robust objective function.
        """

        scenarios = self._scenarios()
        if not self.weights:
            self.weights = [1.0] * self.signature.n_scenarios

        return quicksum(
            quicksum([self.fcn(scenario, model, t) * self.discount_factor**t for t in range(self.horizon)]) * weight
            for scenario, weight in zip(scenarios, self.weights)
        )


class WorstCaseRobustCost(_ScenarioBasedCost):
    def obj_fcn(self, model: ConcreteModel) -> Expression:
        """
        Creates the objective function for the worst-case cost.
        This is simply the upper bound across all scenarios.

        Returns:
            Expression: Robust objective function.
        """
        return model.obj_fcn_ub

    def add_additional_constraints(self, model: ConcreteModel) -> None:
        """
        Adds additional constraints for the worst-case cost.
        Specifically, the upper bound constraint for the worst-case cost 'obj_fcn_ub' and
        the constraint 'obj_fcn_ub_c' that the cost in each scenario is less or equal to the worst-case cost.

        Args:
            model (ConcreteModel): Model instance.
        """
        model.obj_fcn_ub = Var()
        model.obj_fcn_ub_c = ConstraintList()

        scenarios = self._scenarios()

        for scenario in scenarios:
            model.obj_fcn_ub_c.add(
                quicksum([self.fcn(scenario, model, t) * self.discount_factor**t for t in range(self.horizon)])
                <= model.obj_fcn_ub
            )
