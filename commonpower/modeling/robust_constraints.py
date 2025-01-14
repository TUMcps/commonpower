"""
Functionality for creating robust constraints.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from pyomo.core import ConcreteModel, Param, Var

from commonpower.modeling.base import ElementTypes, ModelElement, ModelEntity
from commonpower.modeling.param_initialization import ParamInitializer
from commonpower.modeling.util import SubscriptableFloat

if TYPE_CHECKING:
    from commonpower.core import Node


class RobustConstraintBuilder:
    def __init__(
        self,
        entity: Node,
    ):
        """
        Utility class to expand robust constraints into multiple scenarios
        with corresponding constraints.

        High level approach:
        1. Identify all uncertain parameters in the given model elements.
            This could be elements based on data providers with uncertain forecasts
            or uncertain parameters (as defined by their config or uncertainty bounds).
        2. Check which variables are affected by uncertainties in robust constraints,
            these become uncertain variables.
        3. Iterate over all robust constraints and propage uncertainties through them.
            E.g. some variable var1 is identified as uncertain in the first pass because it is used in
            a robust constraint with uncertain parameters. In the next pass, we identify another variable var2
            as uncertain because it is used in a robust constraint with var1, etc.
        4. For each uncertain variable, create a number of scenarios based on the number of uncertainties
            associated with that variable. Each scenario essentially duplicates the original constraint
            with a unique realization of existing uncertainties. We assume that:
            - uncertainties are independent from each other
            - constraints are monotonic with respect to each individual uncertainty which means that
                we can enclose the original constraint with lower and upper bounds for each uncertainty.
        5. Delete the original robust constraints from the model elements,
            add the expanded constraints and all necessary additional variables.

        Args:
            entity (Node): Entity instance to associate the builder with.
        """

        self.entity = entity
        self.element_mapping: dict[str, ModelElement] = {}

        self.uncertain_vars: list[str] = []
        self.inputs: list[str] = []
        self.params: list[str] = []
        self.uncertain_params: list[str] = []
        self.vars: list[str] = []
        self.uncertain_vars: list[str] = []

        self.robust_constraints: list[_RobustConstraintSignature] = []

        for el in entity.model_elements:

            self.element_mapping[el.name] = el

            if el.type == ElementTypes.CONSTANT:

                if (
                    isinstance(self.entity.config.get(el.name, None), ParamInitializer)
                    and self.entity.config[el.name].is_uncertain()
                    or el.uncertainty_bounds is not None
                ):
                    # update uncertainty config in element, override by default
                    if (
                        isinstance(self.entity.config.get(el.name, None), ParamInitializer)
                        and self.entity.config[el.name].is_uncertain()
                    ):
                        el.uncertainty_bounds = self.entity.config[el.name].bounds
                    self.uncertain_params.append(el.name)
                else:
                    self.params.append(el.name)

            elif el.type == ElementTypes.DATA:

                if entity.data_provider_map[el.name].forecaster.is_uncertain:
                    self.uncertain_params.append(el.name)
                else:
                    self.params.append(el.name)

            elif el.type == ElementTypes.INPUT:
                self.inputs.append(el.name)
            elif el.type == ElementTypes.SET:
                self.params.append(el.name)
            elif el.type in [ElementTypes.COST, ElementTypes.VAR, ElementTypes.STATE]:
                self.vars.append(el.name)
            else:
                # Do not handle "normal" constraints
                pass

        self.robust_constraints = [
            _RobustConstraintSignature.from_element(self, self.entity, el)
            for el in self.entity.model_elements
            if el.type == ElementTypes.ROBUST_CONSTRAINT
        ]

        # make sure that we are considering all uncertain vars across multiple robust constraints
        # we might have multiple constraints for the same uncertain var
        # uncertainty can propagate through multiple constraints,
        # which is why we need to iterate until we have identified all uncertain vars
        # in the worst case, we have to iterate n_robust_constraints times
        self.uncertain_vars = list(set([sig.expand_var for sig in self.robust_constraints if sig.n_uncertainties > 0]))
        for _ in range(len(self.robust_constraints)):
            {sig.update_uncertain_vars(self.uncertain_vars) for sig in self.robust_constraints}
            new_uncertain_vars = list(
                set([sig.expand_var for sig in self.robust_constraints if sig.n_uncertainties > 0])
            )
            if new_uncertain_vars == self.uncertain_vars:
                break
            self.uncertain_vars = new_uncertain_vars

        self.vars = list(set(self.vars) - set(self.uncertain_vars))

        # map vars to their respective robust constraint
        # if multiple constraints map to the same var, we aggregate them in a MultiConstraintSignature
        var_constraint_map = {}
        for sig in self.robust_constraints:
            if sig.expand_var in var_constraint_map:
                if isinstance(var_constraint_map[sig.expand_var], _MultiConstraintSignature):
                    var_constraint_map[sig.expand_var].signatures.append(sig)
                else:
                    var_constraint_map[sig.expand_var] = _MultiConstraintSignature(
                        [var_constraint_map[sig.expand_var], sig]
                    )
            else:
                var_constraint_map[sig.expand_var] = sig

        self.var_constraint_map: dict[str, _RobustConstraintSignature | _MultiConstraintSignature] = var_constraint_map

        # only consider uncertain params and data if they are actually used in robust constraints
        # used_uncertain_params = list(set([el for sig in self.robust_constraints for el in sig.uncertain_params]))
        # self.uncertain_params = list(set(self.uncertain_params) & set(used_uncertain_params))

    def _get_bounds_for_uncertain_param(self, el: ModelElement) -> list[ModelElement]:
        """
        Create lower and upper bound variables for uncertain parameters.

        Args:
            el (ModelElement): Uncertain parameter element.

        Returns:
            list[ModelElement]: List of lower and upper bound elements.
        """
        lb = ModelElement(
            f"{el.name}_lb",
            ElementTypes.CONSTANT,
            f"{el.doc} lower bound",
            domain=el.domain,
            bounds=el.bounds,
            indexed=el.indexed,
            uncertainty_bounds=el.uncertainty_bounds,
            initialize=el.uncertainty_bounds[0] if el.uncertainty_bounds is not None else 0,
            # if uncertainty bounds are none, this uncertain param is based on data
        )

        ub = ModelElement(
            f"{el.name}_ub",
            ElementTypes.CONSTANT,
            f"{el.doc} upper bound",
            domain=el.domain,
            bounds=el.bounds,
            indexed=el.indexed,
            uncertainty_bounds=el.uncertainty_bounds,
            initialize=el.uncertainty_bounds[1] if el.uncertainty_bounds is not None else 0,
        )

        return [lb, ub]

    def _get_scenario_vars(self, el: ModelElement, n_scenarios: int) -> list[ModelElement]:
        """
        Creates scenario variables (el1_scn_1, el1_scn_2, ...) for uncertain variables.
        Also creates lower and upper bound variables.

        Args:
            el (ModelElement): Uncertain variable element.
            n_scenarios (int): Number of scenarios.

        Returns:
            list[ModelElement]: List of scenario variables.
        """
        new_vars = []

        for i in range(1, n_scenarios):  # nominal scenario is already defined
            new_vars.append(
                ModelElement(
                    f"{el.name}_scn_{i}",
                    ElementTypes.VAR,
                    f"{el.doc} scenario {i}",
                    domain=el.domain,
                    # if bounds are given in config, we default to overriding them
                    bounds=self.entity.config.get(el.name, None) or el.bounds,
                    indexed=el.indexed,
                )
            )

        if n_scenarios > 1:

            new_vars.append(
                ModelElement(
                    f"{el.name}_lb",
                    ElementTypes.VAR,
                    f"{el.doc} lower bound",
                    domain=el.domain,
                    # if bounds are given in config, we default to overriding them
                    bounds=self.entity.config.get(el.name, None) or el.bounds,
                    indexed=el.indexed,
                )
            )

            new_vars.append(
                ModelElement(
                    f"{el.name}_ub",
                    ElementTypes.VAR,
                    f"{el.doc} upper bound",
                    domain=el.domain,
                    # if bounds are given in config, we default to overriding them
                    bounds=self.entity.config.get(el.name, None) or el.bounds,
                    indexed=el.indexed,
                )
            )

        return new_vars

    def _get_robust_variables(self) -> list[ModelElement]:
        """
        Creates all necessary additional variables to represent all scenarios
        and bounds of robust constraints.

        Returns:
            list[ModelElement]: List of additional variables.
        """
        model_elements = []

        for uparam in self.uncertain_params:
            model_elements += self._get_bounds_for_uncertain_param(self.element_mapping[uparam])

        for uvar, constr_sig in self.var_constraint_map.items():
            model_elements += self._get_scenario_vars(self.element_mapping[uvar], constr_sig.n_scenarios)

        return model_elements

    def _get_scenario_bound_constraints(
        self, s_id: int, expand_var_mapping: str, sig: _RobustConstraintSignature
    ) -> list[ModelElement]:
        """
        Create lower and upper bound constraints for scenario variables.
        These make sure that all scenario variables are constrained within the bound variables.
        This is a seperate function because we cannot define expressions in loops (they override each other).

        Args:
            s_id (int): Scneario id.
            expand_var_mapping (str): Scenario variable name.
            sig (RobustConstraintSignature): Signature of the robust constraint.

        Returns:
            list[ModelElement]: List of lower and upper bound constraints.
        """

        if sig.element.indexed:

            def lb_expr(model, t):
                return (
                    self.entity.get_pyomo_element(expand_var_mapping, model)[t]
                    >= self.entity.get_pyomo_element(sig.expand_var + "_lb", model)[t]
                )

            def ub_expr(model, t):
                return (
                    self.entity.get_pyomo_element(expand_var_mapping, model)[t]
                    <= self.entity.get_pyomo_element(sig.expand_var + "_ub", model)[t]
                )

        else:

            def lb_expr(model):
                return self.entity.get_pyomo_element(expand_var_mapping, model) >= self.entity.get_pyomo_element(
                    sig.expand_var + "_lb", model
                )

            def ub_expr(model):
                return self.entity.get_pyomo_element(expand_var_mapping, model) <= self.entity.get_pyomo_element(
                    sig.expand_var + "_ub", model
                )

        lb = ModelElement(
            f"{sig.element.name}_scn_{s_id}_lb",
            ElementTypes.CONSTRAINT,
            f"{sig.element.doc} scenario {s_id} lower bound",
            expr=lb_expr,
            indexed=sig.element.indexed,
        )

        ub = ModelElement(
            f"{sig.element.name}_scn_{s_id}_ub",
            ElementTypes.CONSTRAINT,
            f"{sig.element.doc} scenario {s_id} upper bound",
            expr=ub_expr,
            indexed=sig.element.indexed,
        )

        return [lb, ub]

    def _expand_robust_constraint(
        self, sig: _RobustConstraintSignature, n_scenarios: int = None, uncertainty_index: int = 0
    ) -> list[ModelElement]:
        """
        Expands a robust constraint into multiple scenario constraints.
        To this end, we create a ConstraintScenario object which maps variables to
        their corresponding scenario variables.
        This object is then used in the constraint expressions to retrieve the
        correct pyomo elements.

        Args:
            sig (RobustConstraintSignature): Signature of the robust constraint.
            n_scenarios (int, optional): Number of scenarios if the signature is part of a MultiConstraintSignature.
                Defaults to None.
            uncertainty_index (int, optional): For MultiConstraintSignatures,
                this is the index of the first uncertainty of the given signature
                in the list of all uncertainties of the MultiConstraintSignature. Defaults to 0.

        Returns:
            list[ModelElement]: List of expanded constraints.
        """

        constraints_list = []

        n_scenarios = n_scenarios or sig.n_scenarios

        if sig.n_uncertainties == 0:
            bound_setups = np.repeat("", n_scenarios, axis=0).tolist()
        else:
            # nominal scenario
            bound_setups = [tuple(np.repeat("", sig.n_uncertainties).tolist())]

            # all combinations of lower and upper bounds for uncertainties of local constraint signature
            # for multiconstraints, we must make sure that we use bounds in the correct order across all constraints
            multi_constraint_n_uncertainties = int(np.log2(n_scenarios - 1))
            local_bound_setups = list(itertools.product(["_lb", "_ub"], repeat=multi_constraint_n_uncertainties))
            bound_setups += [x[uncertainty_index : uncertainty_index + sig.n_uncertainties] for x in local_bound_setups]

        for s_id, bound_setup in enumerate(bound_setups):

            scenario_label = f"_scn_{s_id}" if s_id > 0 else ""

            expand_state_mapping = sig.expand_var + scenario_label

            element_mapping = {}

            element_mapping.update({el: el + bound_setup[i] for i, el in enumerate(sig.uncertain_params)})

            var_elements_offset = len(sig.uncertain_params)
            element_mapping.update(
                {el: el + bound_setup[var_elements_offset + i] for i, el in enumerate(sig.uncertain_vars)}
            )

            element_mapping.update({el: el for el in sig.params + self.inputs + sig.vars})

            scenario = ConstraintScenario(self.entity, expand_state_mapping, element_mapping)

            scn_expr = sig.element.expr(scenario)

            constraints_list.append(
                ModelElement(
                    f"{sig.element.name}{scenario_label}",
                    ElementTypes.CONSTRAINT,
                    f"{sig.element.doc} scenario {s_id}",
                    expr=scn_expr,
                    indexed=sig.element.indexed,
                )
            )

            # constraint for expand_state lower and upper bounds
            if len(bound_setups) > 1:
                constraints_list += self._get_scenario_bound_constraints(s_id, expand_state_mapping, sig)

        return constraints_list

    def expand_robust_constraints(self) -> None:
        """
        Expands all robust constraints into multiple scenario constraints.
        This function will add the expanded constraints to the model elements of the builder's associated entity.
        The original robust constraints are removed from the model elements.
        """

        exp_model_elements = self._get_robust_variables()

        for sig in self.var_constraint_map.values():
            if isinstance(sig, _MultiConstraintSignature):
                for s_idx, s in enumerate(sig.signatures):
                    unc_idx = sum([_sig.n_uncertainties for _sig in sig.signatures[:s_idx]])
                    exp_model_elements += self._expand_robust_constraint(
                        s, n_scenarios=sig.n_scenarios, uncertainty_index=unc_idx
                    )
            else:
                exp_model_elements += self._expand_robust_constraint(sig)

        # remove robust constraints from model elements
        self.entity.model_elements = [
            el for el in self.entity.model_elements if el.type != ElementTypes.ROBUST_CONSTRAINT
        ]
        self.entity.model_elements += exp_model_elements


class _RobustConstraintSignature:
    def __init__(self):
        """
        Class to store the signature of a robust constraint.
        The signature includes all variables, parameters, and uncertainties are used in the constraint.
        """
        self.expand_var: str = None
        self.vars: list[str] = []
        self.uncertain_vars: list[str] = []
        self.params: list[str] = []
        self.uncertain_params: list[str] = []
        self.element: ModelElement = None

    @classmethod
    def from_element(
        cls, builder: RobustConstraintBuilder, entity: Node, element: ModelElement
    ) -> _RobustConstraintSignature:
        """
        Extracts the signature of a robust constraint from the constraint expression.
        We expect one expandable variable in the constraint expression.
        This variable is the anchor point for the expansion of the robust constraint.
        Specifically, we will check how many uncertainties are present in the constraint and,
        based on that, determine the number of scenarios that need to be created.

        Args:
            builder (RobustConstraintBuilder): Robust constraint builder instance.
            entity (Node): Node instance that the builder is associated with.
            element (ModelElement): Constraint element.

        Raises:
            ValueError: If the robust constraint does not have an expandable variable.

        Returns:
            RobustConstraintSignature: Signature of the robust constraint.
        """
        signature_extractor = _ConstraintSignatureExtractor(builder)
        # simulate constraint to extract signature
        simulation = element.expr(signature_extractor)
        if element.indexed:
            for t in range(entity.horizon):
                simulation(None, t)
        else:
            simulation(None)
        signature_extractor.signature.element = element
        if signature_extractor.signature.expand_var is None:
            raise ValueError(f"Robust Constraint {element.name} does not have an expandable variable")
        return signature_extractor.signature

    def update_uncertain_vars(self, uncertain_vars: list[str]):
        """
        Update the list of uncertain variables based on the given list of known uncertain variables.
        This is necessary to make sure that we consider all uncertain variables across multiple robust constraints.

        Args:
            uncertain_vars (list[str]): List of known uncertain variables.
        """
        # matches between vars in the constraint and known uncertain_vars
        self.uncertain_vars = list(set(self.uncertain_vars) | (set(self.vars) & set(uncertain_vars)))
        self.vars = list(set(self.vars) - set(self.uncertain_vars))

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


class _MultiConstraintSignature:
    def __init__(self, signatures: list[_RobustConstraintSignature]):
        """
        Class to store multiple robust constraint signatures that share the same expandable variable.

        Args:
            signatures (list[RobustConstraintSignature]): List of robust constraint signatures.
        """
        self.signatures = signatures

    @property
    def n_scenarios(self) -> int:
        """
        Number of scenarios defined by the multi constraint.
        This is based on the number of uncertain variables and parameters across all signatures.

        Returns:
            int: Number of scenarios.
        """
        total_uncertainties = sum([sig.n_uncertainties for sig in self.signatures])
        return 2**total_uncertainties + 1 if total_uncertainties > 0 else 1


class ConstraintScenario:
    """
    Utility class to map variables and parameters to their corresponding scenario variables.
    This is the corner stone of robust constraint handling.
    The scenario keeps track of all necessary scenarios defined by uncertainties in the constraint
    and allows to expand the constraint for all these scenarios. All of this is handled in the background.

    You need to define exactly one expand variable in the constraint expression.
    This variable is the anchor point for the expansion of the robust constraint.
    This means that all uncertainties are computed with respect to it, i.e., scenarios are created for it.
    The choice of expand variable is not necessarily unique, however, it is useful to think about
    which variables you want to generate scenarios for,
    i.e., which variable is mainly "affected" by uncertainties.

    Multiple robust constraints can share the same expand variable,
    the appropriate aggregation of uncertainties is taken care of.
    """

    def __init__(self, entity: ModelEntity, expand_state_mapping: str, element_mapping: dict[str, str]):
        """
        Utility class to map variables and parameters to their corresponding scenario variables.

        Args:
            entity (ModelEntity): Entity instance.
            expand_state_mapping (str): Expandable variable name.
            element_mapping (dict[str, str]): Mapping of variables and parameters to their scenario equivalents.
        """
        self.entity = entity
        self.expand_var_mapping = expand_state_mapping
        self.element_mapping = element_mapping

    def __call__(self, element_name: str, model: ConcreteModel, expand_var: bool = False) -> Var | Param:
        """
        Retrieves the pyomo element for the given element name.
        It does so across all scenarios defined by the uncertainties in the robust constraint.

        Args:
            element_name (str): Local name of the element.
            model (ConcreteModel): Model instance.
            expand_var (bool, optional): Determines if this element is the variable/state that is expanded.
                This means that all uncertainties are computed with respect to it, i.e., scenarios are created for it.
                Every robust constraint has exactly one expand variable. Defaults to False.

        Returns:
            Var | Param: Pyomo element.
        """
        el_name_in_scenario = self.element_mapping[element_name] if not expand_var else self.expand_var_mapping
        return self.entity.get_pyomo_element(el_name_in_scenario, model)


class _ConstraintSignatureExtractor(ConstraintScenario):
    def __init__(self, builder: RobustConstraintBuilder):
        """
        Utility class to extract the signature of a robust constraint from the constraint expression.
        The extractor "fakes" being a constraint scenario and stores all element calls when
        an expression that was passed to is called.
        This way, we obtain all (uncertain) parameters and variables used in the constraint expression.
        Args:
            builder (RobustConstraintBuilder): Builder instance.
        """
        self.builder = builder
        self.signature = _RobustConstraintSignature()

    def __call__(self, element_name: str, model: ConcreteModel, expand_var: bool = False) -> float:
        """
        This does not return the pyomo element as the ConstraintScenario class does.
        Instead, it stores the element name with the appropriate category in self.signature.

        Args:
            element_name (str): Local name of the element.
            model (ConcreteModel): Model instance.
            expand_var (bool, optional): Determines if this element is the variable/state that is expanded.
                This means that all uncertainties are computed with respect to it, i.e., scenarios are created for it.
                Every robust constraint has exactly one expand variable. Defaults to False.

        Returns:
            float: Subscriptable float that "fakes" the behaviour of a model element.
        """
        # This will be called multiple times for indexed variables
        if expand_var is True:
            self.signature.expand_var = element_name
        else:
            if element_name in self.builder.uncertain_params:
                self.signature.uncertain_params.append(element_name)
            elif element_name in self.builder.vars:
                self.signature.vars.append(element_name)
            elif element_name in self.builder.params:
                self.signature.params.append(element_name)

            self.signature.uncertain_params = list(set(self.signature.uncertain_params))
            self.signature.vars = list(set(self.signature.vars))
            self.signature.params = list(set(self.signature.params))

        return SubscriptableFloat(1.0)
