"""
Collection of power flow models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from pyomo.core import ConcreteModel, Constraint, quicksum

from commonpower.core import PowerFlowModel

if TYPE_CHECKING:
    from commonpower.core import Bus, Line


class PowerBalanceModel(PowerFlowModel):
    """
    Models pure power balance accross the system.
    """

    def _set_sys_constraints(self, model: ConcreteModel, nodes: List[Bus], lines: List[Line]) -> None:
        """
        .. math::
            \\sum_i p_i = 0 \\\\
            \\sum_i q_i = 0
        """

        def pb_sys_p(model, t):
            return quicksum([n.get_pyomo_element("p", model)[t] for n in nodes]) == 0.0

        def pb_sys_q(model, t):
            return quicksum([n.get_pyomo_element("q", model)[t] for n in nodes]) == 0.0

        model.sys_pb_p = Constraint(model.t, expr=pb_sys_p, doc="global active power balance")
        model.sys_pb_q = Constraint(model.t, expr=pb_sys_q, doc="global reactive power balance")


class DCPowerFlowModel(PowerFlowModel):
    """
    Models DC power flow constraints.
    Based on https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wpen2014-12.pdf.
    """

    def _set_sys_constraints(self, model: ConcreteModel, nodes: List[Bus], lines: List[Line]) -> None:
        """
        .. math::
            \\sum_i p_i = 0 \\\\
            \\sum_i q_i = 0
        """

        def pb_sys_p(model, t):
            return quicksum([n.get_pyomo_element("p", model)[t] for n in nodes]) == 0.0

        def pb_sys_q(model, t):
            return quicksum([n.get_pyomo_element("q", model)[t] for n in nodes]) == 0.0

        model.sys_pb_p = Constraint(model.t, expr=pb_sys_p, doc="global active power balance")
        model.sys_pb_q = Constraint(model.t, expr=pb_sys_q, doc="global reactive power balance")

    def _set_bus_constraint(self, model: ConcreteModel, nid: int, node: Bus, connected_lines: list[Line]):
        """
        Set DC bus constraints and voltage angle of first bus fixed at zero.

        .. math::
            p_i = \\sum_{j=1}^{N} ( B_{ij}(d_i - d_j) ) \\\\
            d_0 = 0
        """
        if nid == 0:
            # fix d (voltage angle) of the first bus at 0 (by convention)
            def slack_d(model, t):
                return node.get_pyomo_element("d", model)[t] == 0.0

            model.c_slack_d = Constraint(model.t, expr=slack_d, doc="fix slack bus voltage angle")

        def dcpf(model, t):
            return node.get_pyomo_element("p", model)[t] == quicksum(
                [
                    (
                        line.get_pyomo_element("B", model)
                        * (node.get_pyomo_element("d", model)[t] - line.dst.get_pyomo_element("d", model)[t])
                        if node is line.src
                        else line.get_pyomo_element("B", model)
                        * (node.get_pyomo_element("d", model)[t] - line.src.get_pyomo_element("d", model)[t])
                    )
                    for line in connected_lines
                ]
            )

        setattr(
            model,
            f"c_dcpf_{node.id}",
            Constraint(model.t, expr=dcpf, doc=f"dc power flow constraint for bus {node.id}"),
        )

    def _set_line_constraint(self, model: ConcreteModel, lid: int, line: Line):
        """
        Sets line flow constraints.
        Technically,we want a limit on the line current I_l.
        However, we will use the line power flow p_l (I~p/v in DCOPF).

        .. math::
            p_l = B_l (d_{src} - d_{dst})

        The system is then factually constrained by the bounds on I_l.
        """

        def dcpf(model, t):
            return line.get_pyomo_element("p", model)[t] == line.get_pyomo_element("B", model) * (
                line.src.get_pyomo_element("d", model)[t] - line.dst.get_pyomo_element("d", model)[t]
            )

        setattr(
            model,
            f"c_dcpf_{line.id}",
            Constraint(model.t, expr=dcpf, doc=f"dc power flow constraint for line {line.id}"),
        )
