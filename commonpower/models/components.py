"""
Collection of component models.
"""
from __future__ import annotations

from typing import List

import pyomo.environ as pyo
from pyomo.core import ConcreteModel, Constraint, Expression

from commonpower.core import Component
from commonpower.modelling import ElementTypes as et
from commonpower.modelling import MIPExpressionBuilder, ModelElement


class Load(Component):
    """
    Load.

    .. runblock:: pycon

        >>> from commonpower.models.components import Load
        >>> Load.info()

    """

    CLASS_INDEX = "d"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.DATA, "active power", pyo.NonNegativeReals, bounds=(0, 1e6)),
            ModelElement("q", et.DATA, "reactive power", pyo.Reals, bounds=(-1e6, 1e6)),
        ]
        return model_elements


class RenewableGen(Component):
    """
    Renewable generator, e.g., wind or PV.

    .. runblock:: pycon

        >>> from commonpower.models.components import RenewableGen
        >>> RenewableGen.info()

    """

    CLASS_INDEX = "r"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.DATA, "active power", pyo.NonPositiveReals, bounds=(-1e6, 0)),
        ]
        return model_elements


class RenewableGenCurtail(Component):
    """
    Curtailable renewable generator.

    .. runblock:: pycon

        >>> from commonpower.models.components import RenewableGenCurtail
        >>> RenewableGenCurtail.info()

    """

    CLASS_INDEX = "rc"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("util", et.INPUT, "plant utilization", pyo.NonNegativeReals, bounds=(0, 1)),
            ModelElement("p_pot", et.DATA, "active power potential", pyo.NonPositiveReals, bounds=(-1e6, 0)),
            ModelElement("p", et.VAR, "active power (after curtailing)", pyo.NonPositiveReals, bounds=(-1e6, 0)),
        ]
        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Add curtailment constraint.

        .. math::
            p = util * p_{pot}
        """

        def cp_f(model, t):
            return (
                self.get_pyomo_element("p", model)[t]
                == self.get_pyomo_element("p_pot", model)[t] * self.get_pyomo_element("util", model)[t]
            )

        curtail_p = ModelElement("c_curtailp", et.CONSTRAINT, "curtailment constraint (active power)", expr=cp_f)

        return [curtail_p]


class ConventionalGen(Component):
    """
    Conventional generator which is fully controlled.

    .. runblock:: pycon

        >>> from commonpower.models.components import ConventionalGen
        >>> ConventionalGen.info()

    """

    CLASS_INDEX = "g"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power", pyo.NonPositiveReals),
            ModelElement("q", et.INPUT, "reactive power", pyo.NonPositiveReals),
            ModelElement("a", et.CONSTANT, "cost parameter a"),
            ModelElement("b", et.CONSTANT, "cost parameter b"),
            ModelElement("c", et.CONSTANT, "cost parameter c", pyo.NonNegativeReals),
        ]
        return model_elements

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        The cost function represents:

        .. math::
            cost = a * p^2 + b * p + c.

        Note that p <= 0 for generators.
        """
        return (
            self.get_pyomo_element("a", model) * self.get_pyomo_element("p", model)[t] ** 2
            + self.get_pyomo_element("b", model) * self.get_pyomo_element("p", model)[t]
            + self.get_pyomo_element("c", model)
        ) * self.tau


class ESS(Component):
    """
    Energy Storage System.

    .. runblock:: pycon

        >>> from commonpower.models.components import ESS
        >>> ESS.info()

    """

    CLASS_INDEX = "e"
    MAX_P = 1e3  # maximum absolute charging power (BigM constraint)

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power"),
            ModelElement("q", et.VAR, "reactive power"),
            ModelElement("soc", et.STATE, "state of charge (absolute)", pyo.NonNegativeReals),
            ModelElement("rho", et.CONSTANT, "cost of wear pu", pyo.NonNegativeReals),
            ModelElement("etac", et.CONSTANT, "charge efficiency", bounds=(0.0, 1.0)),
            ModelElement("etad", et.CONSTANT, "discharge efficiency", bounds=(0.0, 1.0)),
            ModelElement("etas", et.CONSTANT, "self-discharge coefficient", bounds=(0.0, 1.0)),
        ]
        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        Sets a binary charge indicator. \\

        .. math::
            p_{ec} = \\left\\{
            \\begin{array}{ll}
            1 & p \\geq 0 \\\\
            0 & \\, \\textrm{otherwise} \\\\
            \\end{array}
            \\right.
        """
        mb = MIPExpressionBuilder(self, self.MAX_P)

        mb.from_geq("p", 0, "p_ec")

        return mb.model_elements

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Cost of wear.

        .. math::
            cost = |p| * rho
        """
        return (
            (
                self.get_pyomo_element("p", model)[t] * (self.get_pyomo_element("p_ec", model)[t])
                - self.get_pyomo_element("p", model)[t] * (1 - self.get_pyomo_element("p_ec", model)[t])
            )
            * self.get_pyomo_element("rho", model)
            * self.tau
        )

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        .. math::
            soc_{t+1} = etas * soc_{t} + etas * p_{ec} * p_t + \\frac{1}{etad} * (1 - p_{ec}) * p_t
        """

        def dynamic_fcn(model, t):
            # p > 0 is charging
            if t == self.horizon:  # horizon+1 cannot have a constraint
                return Constraint.Skip
            else:
                return (self.get_pyomo_element("etas", model) ** self.tau * self.get_pyomo_element("soc", model)[t]) + (
                    self.get_pyomo_element("etac", model)
                    * self.get_pyomo_element("p", model)[t]
                    * self.get_pyomo_element("p_ec", model)[t]
                    * self.tau
                ) + (
                    (1 / self.get_pyomo_element("etad", model))
                    * self.get_pyomo_element("p", model)[t]
                    * (1 - self.get_pyomo_element("p_ec", model)[t])
                    * self.tau
                ) == self.get_pyomo_element(
                    "soc", model
                )[
                    t + 1
                ]

        dyn = ModelElement("dynamic_fcn", et.CONSTRAINT, "dynamic function", expr=dynamic_fcn)

        return [dyn]


class EV(Component):
    """
    Electric Vehicle.
    An EV has a charge requirement which needs to be fulfilled by a certain deadline.
    To enable continous control, we assume that it is connected at the start of each "control cycle" and
    specify its departure and return time for each cycle.
    Once an EV is unplugged, we assume a decreasing state of charge such that soc=soc_init on return.
    This means the EV is modeled to behave identically in every cycle (once unplugged).

    .. runblock:: pycon

        >>> from commonpower.models.components import EV
        >>> EV.info()

    """

    CLASS_INDEX = "ev"
    MAX_P = 1e3

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power"),
            ModelElement("soc", et.STATE, "state of charge", pyo.NonNegativeReals),
            ModelElement("rho", et.CONSTANT, "cost of wear pu", pyo.NonNegativeReals),
            ModelElement(
                "etac",
                et.CONSTANT,
                "charge efficiency",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "etad",
                et.CONSTANT,
                "discharge efficiency",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "etas",
                et.CONSTANT,
                "self-discharge coefficient",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "departure",
                et.CONSTANT,
                "timestep of departure",
                domain=pyo.NonNegativeIntegers,
            ),
            ModelElement(
                "return",
                et.CONSTANT,
                "timestep of return",
                domain=pyo.NonNegativeIntegers,
            ),
            ModelElement(
                "req_soc_rel",
                et.CONSTANT,
                "required final relative soc, i.e., soc/max(soc)",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
        ]
        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        We add a number of constraints to represent the EVs behavior.
        For details refer to the source code.
        """
        # add absolute soc requirement as additional variable with appropriate constraint
        req_soc_abs = ModelElement(
            "req_soc_abs",
            et.VAR,
            "required final absolute soc",
            bounds=(0, 1e6),
            domain=pyo.NonNegativeReals,
            indexed=False,
        )

        def req_soc_rel_abs_f(model):
            return (
                self.get_pyomo_element("req_soc_abs", model)
                == self.get_pyomo_element("req_soc_rel", model) * self.get_pyomo_element("soc", model)[0].ub
            )

        c_req_soc_abs = ModelElement(
            "c_req_soc_abs",
            et.CONSTRAINT,
            "constraint to set required final absolute soc",
            expr=req_soc_rel_abs_f,
            indexed=False,
        )

        mb = MIPExpressionBuilder(self, self.MAX_P, eps=1e-5)
        mb1 = MIPExpressionBuilder(self, self.MAX_P, eps=1e-5)

        mb.from_geq("p", 0, "p_ec")

        mb.from_geq("soc", "req_soc_abs", "is_charged")  # 1 if soc >= req_soc

        local_time = ModelElement(
            "local_time",
            et.CONSTANT,
            "current timestep within the control cycle",
            domain=pyo.NonNegativeIntegers,
            initialize=0,
        )

        steps_until_next_cycle = ModelElement(
            "steps_until_next_cycle",
            et.VAR,
            "steps_until_next_cycle",
            bounds=(0, 1e6),
            domain=pyo.NonNegativeReals,
            indexed=False,
        )
        c_steps_until_next_cycle = ModelElement(
            "c_steps_until_next_cycle",
            et.CONSTRAINT,
            "constraint to update steps until next cycle",
            expr=lambda model: self.get_pyomo_element("steps_until_next_cycle", model)
            == self.horizon - self.get_pyomo_element("local_time", model),
            indexed=False,
        )

        mb1.from_geq(
            [x for x in range(0, self.horizon + 1)],
            "steps_until_next_cycle",
            out="next_cycle_binary",
        )

        local_time_indexed = ModelElement(
            "local_time_indexed",
            et.VAR,
            "local_time_indexed",
            bounds=(0, 1e6),
            domain=pyo.NonNegativeReals,
        )
        c_local_time_indexed = ModelElement(
            "c_local_time_indexed",
            et.CONSTRAINT,
            "constraint to provide indexed local time",
            expr=lambda model, t: self.get_pyomo_element("local_time_indexed", model)[t]
            == (1 - self.get_pyomo_element("next_cycle_binary", model)[t])
            * (t + self.get_pyomo_element("local_time", model))
            + self.get_pyomo_element("next_cycle_binary", model)[t]
            * (t + self.get_pyomo_element("local_time", model) - self.horizon),
        )

        # calculate when the EV is plugged in. This might be replaced with DATA in future variants of this model.
        mb.from_or(
            mb.from_gt("departure", "local_time_indexed"),
            mb.from_geq("local_time_indexed", "return"),
            out="is_plugged_in",
        )

        # make sure that the EV is charged at the time of departure

        # depature_indicator == 1 only at time of departure
        mb.from_and(
            mb.from_geq("local_time_indexed", "departure"),
            mb.from_geq("departure", "local_time_indexed"),
            out="departure_indicator",
        )

        # when departure_indicator == 1, is_charged cannot be 0
        def soc_req_f(model, t):
            return (
                self.get_pyomo_element("departure_indicator", model)[t]
                <= self.get_pyomo_element("is_charged", model)[t]
            )

        c_soc_req = ModelElement(
            "c_soc_req",
            et.CONSTRAINT,
            "makes sure that the EV is charged on departure",
            expr=soc_req_f,
        )

        # make sure the EV cannot transfer power when unplugged

        def ev_unplugged_f(model, t):
            return self.get_pyomo_element("p", model)[t] * (1 - self.get_pyomo_element("is_plugged_in", model)[t]) == 0

        c_ev_unplugged = ModelElement(
            "c_ev_unplugged",
            et.CONSTRAINT,
            "makes sure that the EV cannot transfer power once it is unplugged",
            expr=ev_unplugged_f,
        )

        # make sure that soc == soc_init on return

        # return_indicator == 1 only at time of return
        mb.from_and(
            mb.from_geq("local_time_indexed", "return"),
            mb.from_geq("return", "local_time_indexed"),
            out="return_indicator",
        )

        unplugged_consumption = ModelElement(
            "unplugged_consumption",
            et.VAR,
            "power consumption when unplugged",
            bounds=(-1e6, 0),
            domain=pyo.NonPositiveReals,
            indexed=True,
        )

        # The solver will set the unplugged_consumption accordingly.
        def soc_on_return_f(model, t):
            return (
                self.get_pyomo_element("soc", model)[t] * self.get_pyomo_element("return_indicator", model)[t]
                == self.get_pyomo_element("soc_init", model) * self.get_pyomo_element("return_indicator", model)[t]
            )

        c_soc_on_return_f = ModelElement(
            "c_unplugged_consumption",
            et.CONSTRAINT,
            "constraint to set the soc to its inital value on return",
            expr=soc_on_return_f,
            indexed=True,
        )

        return (
            [
                req_soc_abs,
                c_req_soc_abs,
                local_time,
                steps_until_next_cycle,
                c_steps_until_next_cycle,
            ]
            + mb1.model_elements
            + [local_time_indexed, c_local_time_indexed]
            + mb.model_elements
            + [
                c_soc_req,
                c_ev_unplugged,
                unplugged_consumption,
                c_soc_on_return_f,
            ]
        )  # this order is necessary due to the interdependence of variables

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Cost of wear.

        .. math::
            cost = |p| * rho
        """
        return (
            (
                self.get_pyomo_element("p", model)[t] * (self.get_pyomo_element("p_ec", model)[t])
                - self.get_pyomo_element("p", model)[t] * (1 - self.get_pyomo_element("p_ec", model)[t])
            )
            * self.get_pyomo_element("rho", model)
            * self.tau
        )

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        While plugged in, the dynamics follow those of a generic ESS.

        .. math::
            soc_{t+1} = etas * soc_{t} + etas * p_{ec} * p_t + \\frac{1}{etad} * (1 - p_{ec}) * p_t
        """

        def dynamic_fcn(model, t):
            # p > 0 is charging
            if t == self.horizon:  # horizon+1 cannot have a constraint
                return Constraint.Skip
            else:
                return (
                    self.get_pyomo_element("is_plugged_in", model)[t]
                    * self.get_pyomo_element("etas", model) ** self.tau
                    * self.get_pyomo_element("soc", model)[t]
                ) + (
                    (1 - self.get_pyomo_element("is_plugged_in", model)[t]) * self.get_pyomo_element("soc", model)[t]
                ) + (
                    self.get_pyomo_element("etac", model)
                    * self.get_pyomo_element("p", model)[t]
                    * self.get_pyomo_element("p_ec", model)[t]
                    * self.tau
                ) + (
                    (1 / self.get_pyomo_element("etad", model))
                    * self.get_pyomo_element("p", model)[t]
                    * (1 - self.get_pyomo_element("p_ec", model)[t])
                    * self.tau
                ) + (
                    (1 - self.get_pyomo_element("is_plugged_in", model)[t])
                    * self.get_pyomo_element("unplugged_consumption", model)[t]
                ) == self.get_pyomo_element(
                    "soc", model
                )[
                    t + 1
                ]

        dyn_p = ModelElement("dynamic_fcn", et.CONSTRAINT, "dynamic function", expr=dynamic_fcn)

        return [dyn_p]

    def _additional_updates(self) -> None:
        """
        Updates a local time variable used for tracking departure and return times.
        """
        local_time = self.get_value(self.instance, "local_time")
        if local_time == self.horizon - 1:
            self.set_value(self.instance, "local_time", 0)
        else:
            self.set_value(self.instance, "local_time", local_time + 1)


class EVData(Component):
    """
    Electric Vehicle with schedule data.
    An EV has a charge requirement which needs to be fulfilled by a certain deadline.
    To enable continous control, we assume that it is connected at the start of each "control cycle" and
    specify its departure and return time for each cycle.
    Once an EV is unplugged, we assume a decreasing state of charge such that soc=soc_init on return.
    This means the EV is modeled to behave identically in every cycle (once unplugged).

    .. runblock:: pycon

        >>> from commonpower.models.components import EVData
        >>> EVData.info()

    """

    CLASS_INDEX = "evd"
    MAX_P = 1e3

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power"),
            ModelElement("soc", et.STATE, "state of charge", pyo.NonNegativeReals),
            ModelElement("rho", et.CONSTANT, "cost of wear pu", pyo.NonNegativeReals),
            ModelElement(
                "etac",
                et.CONSTANT,
                "charge efficiency",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "etad",
                et.CONSTANT,
                "discharge efficiency",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "etas",
                et.CONSTANT,
                "self-discharge coefficient",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
            ModelElement(
                "is_plugged_in",
                et.DATA,
                "presence indicator",
                domain=pyo.Binary,
            ),
            ModelElement(
                "req_soc_rel",
                et.CONSTANT,
                "required final relative soc, i.e., soc/max(soc)",
                bounds=(0.0, 1.0),
                domain=pyo.NonNegativeReals,
            ),
        ]
        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        We add a number of constraints to represent the EVs behavior.
        For details refer to the source code.
        """
        # add absolute soc requirement as additional variable with appropriate constraint
        req_soc_abs = ModelElement(
            "req_soc_abs",
            et.VAR,
            "required final absolute soc",
            bounds=(0, 1e6),
            domain=pyo.NonNegativeReals,
            indexed=False,
        )

        def req_soc_rel_abs_f(model):
            return (
                self.get_pyomo_element("req_soc_abs", model)
                == self.get_pyomo_element("req_soc_rel", model) * self.get_pyomo_element("soc", model)[0].ub
            )

        c_req_soc_abs = ModelElement(
            "c_req_soc_abs",
            et.CONSTRAINT,
            "constraint to set required final absolute soc",
            expr=req_soc_rel_abs_f,
            indexed=False,
        )

        # We have the is_plugged_in data.
        # We need an edge detection to determine the departure and return indicators.
        # If the edge is 1, the EV has departed in this time step.
        # If the edge is -1, the EV has returned in this time step.
        is_plugged_in_edge = ModelElement(
            "is_plugged_in_edge",
            et.VAR,
            "edge indicator of is_plugged_in",
            bounds=(-1, 1),
            domain=pyo.Integers,
            indexed=True,
        )

        # Essentially edge detection via kernel (1,-1)
        def is_plugged_in_edge_f(model, t):
            if t == 0:  # plugged in by definition
                return self.get_pyomo_element("is_plugged_in_edge", model)[t] == 0
            else:
                return (
                    self.get_pyomo_element("is_plugged_in_edge", model)[t]
                    == self.get_pyomo_element("is_plugged_in", model)[t - 1]
                    - self.get_pyomo_element("is_plugged_in", model)[t]
                )

        c_is_plugged_in_edge = ModelElement(
            "c_is_plugged_in_edge",
            et.CONSTRAINT,
            "sets the edge indicator of is_plugged_in",
            expr=is_plugged_in_edge_f,
        )

        mb = MIPExpressionBuilder(self, self.MAX_P, eps=1e-5)

        mb.from_geq("p", 0, "p_ec")

        mb.from_geq("soc", "req_soc_abs", "is_charged")  # 1 if soc >= req_soc

        mb.from_gt("is_plugged_in_edge", 0, "departure_indicator")  # 1 if departure edge
        mb.from_gt(0, "is_plugged_in_edge", "return_indicator")  # 1 if return edge

        # make sure that the EV is charged at the time of departure
        # when departure indicator == 1, is_charged cannot be 0
        def soc_req_f(model, t):
            return (
                self.get_pyomo_element("departure_indicator", model)[t]
                <= self.get_pyomo_element("is_charged", model)[t]
            )

        c_soc_req = ModelElement(
            "c_soc_req",
            et.CONSTRAINT,
            "makes sure that the EV is charged on departure",
            expr=soc_req_f,
        )

        # make sure the EV cannot transfer power when unplugged
        def ev_unplugged_f(model, t):
            return self.get_pyomo_element("p", model)[t] * (1 - self.get_pyomo_element("is_plugged_in", model)[t]) == 0

        c_ev_unplugged = ModelElement(
            "c_ev_unplugged",
            et.CONSTRAINT,
            "makes sure that the EV cannot transfer power once it is unplugged",
            expr=ev_unplugged_f,
        )

        # make sure that soc == soc_init on return
        unplugged_consumption = ModelElement(
            "unplugged_consumption",
            et.VAR,
            "power consumption when unplugged",
            bounds=(-1e6, 0),
            domain=pyo.NonPositiveReals,
            indexed=True,
        )

        # The solver will set the unplugged_consumption accordingly.
        # We need to project the arrival edge from -1 to 1, and the departure edge to 0.
        def soc_on_return_f(model, t):
            return (
                self.get_pyomo_element("soc", model)[t] * self.get_pyomo_element("return_indicator", model)[t]
                == self.get_pyomo_element("soc_init", model) * self.get_pyomo_element("return_indicator", model)[t]
            )

        c_unplugged_consumption = ModelElement(
            "c_unplugged_consumption",
            et.CONSTRAINT,
            "constraint to set the soc to its inital value on return",
            expr=soc_on_return_f,
            indexed=True,
        )

        return (
            [
                req_soc_abs,
                c_req_soc_abs,
                is_plugged_in_edge,
                c_is_plugged_in_edge,
            ]
            + mb.model_elements
            + [
                c_soc_req,
                c_ev_unplugged,
                unplugged_consumption,
                c_unplugged_consumption,
            ]
        )

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Cost of wear.

        .. math::
            cost = |p| * rho
        """
        return (
            (
                self.get_pyomo_element("p", model)[t] * (self.get_pyomo_element("p_ec", model)[t])
                - self.get_pyomo_element("p", model)[t] * (1 - self.get_pyomo_element("p_ec", model)[t])
            )
            * self.get_pyomo_element("rho", model)
            * self.tau
        )

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        While plugged in, the dynamics follow those of a generic ESS.

        .. math::
            soc_{t+1} = etas * soc_{t} + etas * p_{ec} * p_t + \\frac{1}{etad} * (1 - p_{ec}) * p_t
        """

        def dynamic_fcn(model, t):
            # p > 0 is charging
            if t == self.horizon:  # horizon+1 cannot have a constraint
                return Constraint.Skip
            else:
                return (
                    self.get_pyomo_element("is_plugged_in", model)[t]
                    * self.get_pyomo_element("etas", model) ** self.tau
                    * self.get_pyomo_element("soc", model)[t]
                ) + (
                    (1 - self.get_pyomo_element("is_plugged_in", model)[t]) * self.get_pyomo_element("soc", model)[t]
                ) + (
                    self.get_pyomo_element("etac", model)
                    * self.get_pyomo_element("p", model)[t]
                    * self.get_pyomo_element("p_ec", model)[t]
                    * self.tau
                ) + (
                    (1 / self.get_pyomo_element("etad", model))
                    * self.get_pyomo_element("p", model)[t]
                    * (1 - self.get_pyomo_element("p_ec", model)[t])
                    * self.tau
                ) + (
                    (1 - self.get_pyomo_element("is_plugged_in", model)[t])
                    * self.get_pyomo_element("unplugged_consumption", model)[t]
                ) == self.get_pyomo_element(
                    "soc", model
                )[
                    t + 1
                ]

        dyn_p = ModelElement("dynamic_fcn", et.CONSTRAINT, "dynamic function", expr=dynamic_fcn)

        return [dyn_p]


class _FlexLoad_Episodic(Component):
    """
    Flexible (controllable) load. TODO: Implement continuous version of this.
    Flexible loads have an energy requirement which needs to be fullfiled by a certain deadline.
    They cannot be interrupted and once they are started, they consume a constant amount of power
    until the energy requirement is fulfilled.
    Examples of this type of load are household appliances like washing/drying machines.
    This component only supports episodic control as it needs to be reset after one activation cycle.

    The load is activated if the input "act" > 0.5 for one time step.

    .. runblock:: pycon

        >>> from commonpower.models.components import FlexLoad_Episodic
        >>> FlexLoad_Episodic.info()

    """

    CLASS_INDEX = "f"
    MAX_P = 1e3

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("act", et.INPUT, "activate component", pyo.NonNegativeReals, bounds=(0, 1)),
            ModelElement("p", et.VAR, "active power", pyo.NonNegativeReals),
            ModelElement("sum_p", et.STATE, "sum of consumed active power", pyo.NonNegativeReals),
            ModelElement("is_active", et.STATE, "binary indicator: load is active", pyo.Binary),
            # we have to adjust the deadline after each control step (-1)
            ModelElement("deadline", et.CONSTANT, "deadline (time step)", pyo.NonNegativeIntegers),
            ModelElement("req_p", et.CONSTANT, "total requirement of active power", pyo.NonNegativeReals),
            ModelElement(
                "run_time",
                et.CONSTANT,
                "number of time steps the component is running after started",
                pyo.NonNegativeIntegers,
            ),
        ]
        return model_elements

    def _get_additional_constraints(self) -> List[ModelElement]:
        """
        The constraints are roughly given by:
        is_available <- deadline >= t \\
        is_done <- sum_p >= req_p \\
        1 <- is_available OR is_done \\
        consuming <- triggered OR is_active \\
        triggered <- ((NOT is_done AND NOT is_active) AND (act >= 0.5))
        """
        mb = MIPExpressionBuilder(self, self.MAX_P)

        mb.from_geq("sum_p", "req_p", "is_done")  # 1 if sum >= req

        # 1 if t < deadline (we iterate through the range when building the constraint)
        mb.from_gt("deadline", [x for x in range(self.horizon + 1)], "is_available")

        # either is_done and/or is_available must be true -> Make sure the load is satisfied by the deadline
        mb.enforce_value(mb.from_or("is_done", "is_available"), 1)

        mb.from_and(
            mb.from_and(mb.from_not("is_done", "not_done"), mb.from_not("is_active", "not_active")),
            mb.from_gt("act", 0.5),
            "triggered",
        )

        mb.from_or("triggered", "is_active", "consuming")

        def p_level_f(model, t):
            return self.get_pyomo_element("p", model)[t] == self.get_pyomo_element("consuming", model)[t] * (
                self.get_pyomo_element("req_p", model) / self.get_pyomo_element("run_time", model)
            )

        p_level = ModelElement("clevel_p", et.CONSTRAINT, "active power level", expr=p_level_f)

        return mb.model_elements + [p_level]

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        .. math::
            sump_{t+1} = sump_{t} + p_{t} \\\\
            isactive_{t+1} = consuming_t \\text{ AND NOT } isdone_{t+1}
        """

        def dynamic_p_fcn(model, t):
            if t == self.horizon:
                return Constraint.Skip
            else:
                return (
                    self.get_pyomo_element("sum_p", model)[t + 1]
                    == self.get_pyomo_element("sum_p", model)[t] + self.get_pyomo_element("p", model)[t]
                )

        dyn_p = ModelElement("dynamic_fcn_p", et.CONSTRAINT, "dynamic function", expr=dynamic_p_fcn)

        def update_status_f(model, t):
            if t == self.horizon:
                return Constraint.Skip
            else:
                return (
                    self.get_pyomo_element("is_active", model)[t + 1]
                    == self.get_pyomo_element("consuming", model)[t] * self.get_pyomo_element("not_done", model)[t + 1]
                )

        upt_status = ModelElement("update_status_f", et.CONSTRAINT, "update is_active status", expr=update_status_f)

        return [dyn_p, upt_status]

    def _additional_updates(self) -> None:
        """
        Updates the internal deadline.
        """
        dln = self.get_value(self.instance, "deadline")
        if dln > 0:
            self.set_value(self.instance, "deadline", dln - 1)


class HeatPumpWithoutStorageButCOP(Component):
    """
    Heat pump without storage but considering a timeseries-based coefficient of performance. \\
    Default values for H_FH, H_out, tau_building, Cw_FH from
    https://www.researchgate.net/publication/257778680_Multi-objective_optimal_control_of
    _an_air-to-water_heat_pump_for_residential_heating/link/545207b60cf2bf864cbac189/download \\
    H_FH = 1.1 [kW/K] \\
    H_out = 0.26 [kW/K] \\
    tau_building = 240 [h] \\
    Cw_FH = 1.1625 [kWh/K] (note: in the paper they provide this in J/K)

    Dynamics here: https://www.imrtweb.ethz.ch/users/geering/diss_full/diss_bianchi.pdf (Hausmodell 2. Ordnung)

    COP data here: https://www.nature.com/articles/s41597-019-0199-y

    .. runblock:: pycon

        >>> from commonpower.models.components import HeatPumpWithoutStorageButCOP
        >>> HeatPumpWithoutStorageButCOP.info()

    """

    CLASS_INDEX = "hp"

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power"),
            # ModelElement("q", et.INPUT, "reactive power"),
            ModelElement("T_indoor_setpoint", et.CONSTANT, "temperature that should be reached", pyo.Reals),
            ModelElement("T_indoor", et.STATE, "indoor temperature", pyo.Reals),
            ModelElement("T_ret_FH", et.STATE, "water temperature of floor heating system", pyo.Reals),
            # ModelElement("T_sup_HP", et.STATE, "heat pump supply water temperature", pyo.Reals),
            ModelElement("T_outside", et.DATA, "outside temperature", pyo.Reals),
            ModelElement("COP", et.DATA, "coefficient of performance", pyo.NonNegativeReals),
            ModelElement(
                "H_FH", et.CONSTANT, "thermal conductivity between floor heating and house", pyo.NonNegativeReals
            ),  # [kW/K]
            ModelElement(
                "H_out", et.CONSTANT, "thermal conductivity between house and surrounding", pyo.NonNegativeReals
            ),  # [kW/K]
            ModelElement("tau_building", et.CONSTANT, "building time constant ", pyo.NonNegativeReals),
            # [h]
            ModelElement(
                "Cw_FH",
                et.CONSTANT,
                "thermal capacity of volume of water in floor heating system",
                pyo.NonNegativeReals,
            ),  # [kWh/K]
            ModelElement("c", et.CONSTANT, "weighting coefficient", pyo.NonNegativeReals),
        ]
        return model_elements

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        def dynamic_temp_indoor_fcn(model, t):
            if t == self.horizon:
                return Constraint.Skip
            else:
                return self.get_pyomo_element("T_indoor", model)[t + 1] == self.get_pyomo_element("T_indoor", model)[
                    t
                ] + self.tau * (
                    -(
                        (self.get_pyomo_element("H_FH", model) + self.get_pyomo_element("H_out", model))
                        / (self.get_pyomo_element("H_out", model) * self.get_pyomo_element("tau_building", model))
                    )
                    * self.get_pyomo_element("T_indoor", model)[t]
                    + (
                        self.get_pyomo_element("H_FH", model)
                        / (self.get_pyomo_element("H_out", model) * self.get_pyomo_element("tau_building", model))
                    )
                    * self.get_pyomo_element("T_ret_FH", model)[t]
                    + self.get_pyomo_element("T_outside", model)[t] / self.get_pyomo_element("tau_building", model)
                )

        def dynamic_temp_ret_fh_fcn(model, t):
            if t == self.horizon:
                return Constraint.Skip
            else:
                return self.get_pyomo_element("T_ret_FH", model)[t + 1] == self.get_pyomo_element("T_ret_FH", model)[
                    t
                ] + self.tau * (
                    (self.get_pyomo_element("H_FH", model) / self.get_pyomo_element("Cw_FH", model))
                    * self.get_pyomo_element("T_indoor", model)[t]
                    - (self.get_pyomo_element("H_FH", model) / self.get_pyomo_element("Cw_FH", model))
                    * self.get_pyomo_element("T_ret_FH", model)[t]
                    + (
                        self.get_pyomo_element("p", model)[t]
                        * self.get_pyomo_element("COP", model)[t]
                        / self.get_pyomo_element("Cw_FH", model)
                    )
                )

        dyn_temp_indoor = ModelElement(
            "dynamic_fcn_T_indoor", et.CONSTRAINT, "dynamic function", expr=dynamic_temp_indoor_fcn
        )
        dyn_temp_ret_fh = ModelElement(
            "dynamic_fcn_T_ret_FH", et.CONSTRAINT, "dynamic function", expr=dynamic_temp_ret_fh_fcn
        )

        return [dyn_temp_indoor, dyn_temp_ret_fh]

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Cost of discomfort.

        .. math::
            cost = c * (T_{indoor} - T_{indoor\\_setpoint})^2
        """
        return (
            self.get_pyomo_element("c", model)
            * (self.get_pyomo_element("T_indoor", model)[t + 1] - self.get_pyomo_element("T_indoor_setpoint", model))
            ** 2
            * self.tau
        )


""" SIMPLIFIED MODELS """


class ESSLinear(Component):
    """
    Simplified ESS model without efficiencies and without cost function.

    .. runblock:: pycon

        >>> from commonpower.models.components import ESSLinear
        >>> ESSLinear.info()

    """

    CLASS_INDEX = "el"
    MAX_P = 1e3  # maximum absolute charging power (BigM constraint)

    @classmethod
    def _get_model_elements(cls) -> List[ModelElement]:
        model_elements = [
            ModelElement("p", et.INPUT, "active power"),
            ModelElement("q", et.VAR, "reactive power"),
            ModelElement("soc", et.STATE, "state of charge (absolute)", pyo.NonNegativeReals),
        ]
        return model_elements

    def _get_dynamic_fcn(self) -> List[ModelElement]:
        """
        .. math::
            soc_{t+1} = soc_{t} + p_t
        """

        def dynamic_fcn(model, t):
            # p > 0 is charging
            if t == self.horizon:
                return Constraint.Skip
            else:
                return (
                    self.get_pyomo_element("soc", model)[t] + self.get_pyomo_element("p", model)[t] * self.tau
                    == self.get_pyomo_element("soc", model)[t + 1]
                )

        dyn = ModelElement("dynamic_fcn", et.CONSTRAINT, "dynamic function", expr=dynamic_fcn)

        return [dyn]
