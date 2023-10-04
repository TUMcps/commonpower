import unittest


import pandapower.networks as pn

from commonpower.core import *
from commonpower.models.components import *
from commonpower.models.lines import *
from commonpower.models.powerflow import *
from commonpower.utils.param_initialization import *
from commonpower.modelling import *
from commonpower.extensions.network_import import *


class TestImport(unittest.TestCase):
    def test_pandapower_import(self):
        nets = [
            pn.case39(),
            pn.create_kerber_vorstadtnetz_kabel_2(),
            pn.create_synthetic_voltage_control_lv_network(network_class="suburb_1"),
        ]

        for net in nets:
            sys = PandaPowerImporter().import_net(net, DCPowerFlowModel())

            """ sys.initialize(horizon=24, tau=timedelta(minutes=60))
            t0 = "30.03.2016 12:00"
            sys.reset(t0) """

            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
