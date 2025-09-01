"""
Funcionality to import exisint topologies from other libraries/tools.
"""
from __future__ import annotations

from typing import Union

from pandapower.auxiliary import pandapowerNet

from commonpower.core import *
from commonpower.extensions.factories import Factory
from commonpower.modeling.base import *
from commonpower.modeling.param_initialization import *
from commonpower.models.buses import *
from commonpower.models.components import *
from commonpower.models.lines import *
from commonpower.models.powerflow import *


class PandaPowerImporter:
    def __init__(self, bus_type: Bus = Bus, base_voltage_kV: float = 0.4):
        """
        This class can be used to import pandapower networks into CommonPower.
        It specifies a mapping between pandapower and CommonPower elements and outputs the converted system.
        At the moment, this importer only imports busses and lines, and
        does not consider any components or data sources.
        We assume a net-wide voltage level and import trafos as (very) high admittance lines.

        Args:
            node_type (Node): (Sub-)Class of Node to be mapped to pandapower busses. Defaults to Node.
            base_voltage_kV (float, optional): Base voltage of the network in kV. Defaults to 0.4 (400V).
        """
        self.mapping = {
            # voltage in p.u.
            "bus": (bus_type, {"name": lambda x: x["name"]}, {"v": lambda x: (x["min_vm_pu"], x["max_vm_pu"])}),
            "line": (
                BasicLine,
                {
                    "name": lambda x: x["name"],
                    "src": lambda x: self.get_bus_node_mapping(x["from_bus"]),
                    "dst": lambda x: self.get_bus_node_mapping(x["to_bus"]),
                },
                {
                    "I": lambda x: (-x["max_i_ka"], x["max_i_ka"]),
                    # Multiply with net voltage (400V) to get power limits (in kW)
                    "p": lambda x: (-x["max_i_ka"] * base_voltage_kV * 1e3, x["max_i_ka"] * base_voltage_kV * 1e3),
                    "G": lambda x: 1 / (x["r_ohm_per_km"] * x["length_km"] / 1e3),  # convert to kOhm
                    "B": lambda x: 1 / (x["x_ohm_per_km"] * x["length_km"] / 1e3),  # convert to kOhm
                },
            ),
            # We treat Trafos as Lines with very high admittance
            "trafo": (
                BasicLine,
                {
                    "name": lambda x: x["name"],
                    "src": lambda x: self.get_bus_node_mapping(x["hv_bus"]),
                    "dst": lambda x: self.get_bus_node_mapping(x["lv_bus"]),
                },
                {"I": lambda x: (-1e6, 1e6), "p": lambda x: (-1e6, 1e6), "G": lambda x: 1e6, "B": lambda x: 1e6},
            ),
        }

        self.bus_node_mapping = {}

    def import_net(
        self,
        net: pandapowerNet,
        power_flow_model: PowerFlowModel,
        node_factory: Union[None, Factory] = None,
        restrict_factory_to: str = "",
    ) -> System:
        """
        Imports the given pandapowerNet as commonpower net and, if given, generates busses with components
        based on the passed factory.

        Args:
            net (pandapowerNet): Network to import.
            power_flow_model (PowerFlowModel): Power flow model instance to be used.
            node_factory (Union[None, Factory], optional): Factory to generate nodes and their sub-components.
                Defaults to None.
            restrict_factory_to (str, optional): Restricts the use of the factory to nodes containing this string in
                their pandapowerNet name. All other nodes are generated purely based on the Importer's mapping.
                Defaults to "".

        Returns:
            System: Generated system.
        """

        sys = System(power_flow_model=power_flow_model)

        for importee, map in self.mapping.items():  # this assumes we will fetch busses before lines
            pp_df = getattr(net, importee)  # this is the pandapower df with all elements of this type

            for idx, row in pp_df.iterrows():
                entity_class_map = map[0]
                entity_constructor_kwargs = {n: fcn(row) for n, fcn in map[1].items()}

                entity_config = {}
                for n, fcn in map[2].items():
                    try:
                        entity_config[n] = fcn(row)
                    except KeyError:  # sometimes not all mapping elements might exist in the pandapower df
                        pass

                if importee == "bus":
                    if node_factory and (restrict_factory_to in row["name"] or not restrict_factory_to):
                        node_factory.attach_node(sys, entity_constructor_kwargs, entity_config)
                    else:
                        sys.add_node(entity_class_map(**entity_constructor_kwargs, config=entity_config))
                    self.bus_node_mapping[idx] = sys.nodes[-1]  # store mapping to created node instance
                elif importee in ["line", "trafo"]:
                    sys.add_line(entity_class_map(**entity_constructor_kwargs, config=entity_config))

        return sys

    def get_bus_node_mapping(self, bus: str) -> Bus:
        return self.bus_node_mapping[bus]
