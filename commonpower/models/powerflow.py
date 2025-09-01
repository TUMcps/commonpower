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


class LinDistFlowPowerFlowModel(PowerFlowModel):
    """
    Models LinDistFlow power flow constraints.
    Based on https://ieeexplore.ieee.org/document/19266.
    """

    def __init__(
        self,
        base_voltage: float = 0.4,  # in kV, default is 0.4kV for distribution networks
        base_apparent_power: float = 1e2,  # in kVA, default is 100kVA for distribution networks
    ):
        self.base_voltage = base_voltage
        self.base_apparent_power = base_apparent_power

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
        Set bus constraints and voltage magnitude of first bus fixed at 1.
        LinDistFlow implements a radial network power flow model where power flows from parent to child buses.
        The constraints represent power balance at each bus.

        .. math::
            p_i = \\sum_{j \\in children(i)} p_j + \\sum_{k \\in parents(i)} p_{k}  \\\\
            q_i = \\sum_{j \\in children(i)} q_j + \\sum_{k \\in parents(i)} q_{k}  \\\\
            v_0 = 1
        """
        if nid == 0:
            # fix v (voltage magnitude) of the first bus at 1 (by convention)
            def slack_v(model, t):
                return node.get_pyomo_element("v", model)[t] == 1.0

            model.c_slack_v = Constraint(model.t, expr=slack_v, doc="fix slack bus voltage magnitude")

        def lindistpf_p(model, t):
            # LinDistFlow active power balance
            # p_i = sum of power flowing out and power flowing in
            line_power_out = quicksum(
                [line.get_pyomo_element("p", model)[t] for line in connected_lines if node is line.src]
            )
            line_power_in = quicksum(  # only one in radial network
                [line.get_pyomo_element("p", model)[t] for line in connected_lines if node is line.dst]
            )
            return node.get_pyomo_element("p", model)[t] + line_power_in == line_power_out

        def lindistpf_q(model, t):
            # LinDistFlow reactive power balance
            # q_i = sum of power flowing out and power flowing in
            line_power_out = quicksum(
                [line.get_pyomo_element("q", model)[t] for line in connected_lines if node is line.src]
            )
            line_power_in = quicksum(  # only one in radial network
                [line.get_pyomo_element("q", model)[t] for line in connected_lines if node is line.dst]
            )
            return node.get_pyomo_element("q", model)[t] + line_power_in == line_power_out

        setattr(
            model,
            f"c_lindistpf_p_{node.id}",
            Constraint(model.t, expr=lindistpf_p, doc=f"LinDistFlow active power constraint for bus {node.id}"),
        )

        setattr(
            model,
            f"c_lindistpf_q_{node.id}",
            Constraint(model.t, expr=lindistpf_q, doc=f"LinDistFlow reactive power constraint for bus {node.id}"),
        )

    def _set_line_constraint(self, model: ConcreteModel, lid: int, line: Line):
        """
        Sets LinDistFlow voltage constraints.
        The voltage drop constraint relates voltage magnitudes between connected buses.
        Index i is the child bus (line.dst) and k is the parent bus (line.src).

        .. math::
            v_i = v_k - 2(R_{ik} p_i + X_{ik} q_i), \\\\
            where R = 1/G (resistance) and X = 1/B (reactance).
        """

        def lindistpf_voltage(model, t):
            # LinDistFlow voltage constraint: v_i = v_k - 2(R*P + X*Q)
            # All values must be in per-unit (pu) for dimensional consistency

            # Convert base units to SI: V_base [V], S_base [VA]
            V_base = self.base_voltage * 1e3  # kV to V
            S_base = self.base_apparent_power * 1e3  # kVA to VA
            Z_base = V_base**2 / S_base  # Ohm

            # Actual impedances in Ohm (assuming G, B in 1/kOhm, so 1/G, 1/B in kOhm)
            resistance_actual_kohm = 1 / line.get_pyomo_element("G", model)  # kOhm
            reactance_actual_kohm = 1 / line.get_pyomo_element("B", model)  # kOhm
            resistance_actual = resistance_actual_kohm * 1e3  # Ohm
            reactance_actual = reactance_actual_kohm * 1e3  # Ohm

            # Convert to per-unit
            resistance_pu = resistance_actual / Z_base
            reactance_pu = reactance_actual / Z_base

            # Power flows in W/VA (convert kW/kVAr to W/VA)
            p_actual = line.get_pyomo_element("p", model)[t] * 1e3  # kW to W
            q_actual = line.get_pyomo_element("q", model)[t] * 1e3  # kVAr to VAr
            p_pu = p_actual / S_base
            q_pu = q_actual / S_base

            return line.dst.get_pyomo_element("v", model)[t] == line.src.get_pyomo_element("v", model)[t] - 2 * (
                resistance_pu * p_pu + reactance_pu * q_pu
            )

        setattr(
            model,
            f"c_lindistpf_v_{line.id}",
            Constraint(model.t, expr=lindistpf_voltage, doc=f"LinDistFlow voltage constraint for line {line.id}"),
        )
