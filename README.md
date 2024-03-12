CommonPower
===========

Introduction
-------------

CommonPower provides a flexible framework to model power systems, interface to single-agent and multi-agent RL controllers,
and maintain safety based on a symbolic representation of the system equations.
Alternatively, the system model can directly be used to solve a given use case via the built-in model predictive controller.
Following a modular design philosophy, CommonPower is an easily extendable tool for the development and benchmarking
of RL controllers in the context of smart grids. The initial focus is on energy management and economic dispatch.
Additionally, CommonPower readily allows the influence of forecast quality to be studied.
The primary features are

- an object-oriented approach to modelling power system entities,

- a Pyomo-based symbolic math representation of entities to obtain all relevant system equations in the background,

- interfaces for single/multi-agent reinforcement learning and optimal control,

- a flexible interface to make use of diverse data sources and forecasting models.

Documentation
--------------

Our documentation is available on [ReadtheDocs](https://commonpower.readthedocs.io).

Example
--------

The following code is an illustrative example of a multi-agent scenario with three housholds and heterogeneous controllers.
Two of the households are controlled by a multi-agent RL algorithm, the third by a model predictive controller.
This example covers the system creation, training, and deployment.

```python
from commonpower.core import System
from commonpower.models.components import Load, RenewableGen, ESSLinear
from commonpower.models.busses import RTPricedBus, ExternalGrid
from commonpower.models.powerflow import PowerBalanceModel
from commonpower.data_forecasting import CSVDataSource, DataProvider, PersistenceForecaster, PerfectKnowledgeForecaster
from commonpower.control.runners import DeploymentRunner, MAPPOTrainer
from commonpower.control.controllers import OptimalController, RLControllerMA
from commonpower.control.wrappers import MultiAgentWrapper
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer

pv_data = CSVDataSource("<path_to_data>")
load_data = CSVDataSource("<path_to_data>")
price_data = CSVDataSource("<path_to_data>")

# create 3 identical households
households = []
for i in range(3):
    household = RTPricedBus(f"household{i}").add_data_provider(DataProvider(price_data, PersistenceForecaster()))
    household.add_node(
        RenewableGen(f"pv{i}").add_data_provider(DataProvider(pv_data, PersistenceForecaster()))
    ).add_node(
        Load(f"load{i}").add_data_provider(DataProvider(load_data, PerfectKnowledgeForecaster()))
    ).add_node(
        ESSLinear(f"ess{i}", {
            "p": [-1.5, 1.5], # active power limits in kW
            "q": [0.0, 0.0],  # reactive power limits in kW
            "soc": [0.2, 9],  # state of charge limits in kWh
            "soc_init": 5.0  # initial state of charge
        })
    )
    households.append(household)

substation = ExternalGrid("substation")

sys = System(PowerBalanceModel()).add_node(households[0]).add_node(households[1]).add_node(households[2]).add_node(substation)

mpc_controller = OptimalController("mpc1").add_entity(households[0])
rl_agent1 = RLControllerMA("agent1", safety_layer=ActionProjectionSafetyLayer()).add_entity(households[1])
rl_agent2 = RLControllerMA("agent2", safety_layer=ActionProjectionSafetyLayer()).add_entity(households[2])

# traning
train_runner = MAPPOTrainer(sys, alg_config={"<your>": "<config>"}, wrapper=MultiAgentWrapper)
train_runner.run()

# deployment
deploy_runner = DeploymentRunner(sys, wrapper=MultiAgentWrapper)
deploy_runner.run()
```

For more examples, have a look at our [Tutorials](https://commonpower.readthedocs.io/en/latest/tutorials.html).


Reference
----------

CommonPower was developed and is maintained by the Cyber-Physical Systems Group at the Chair for Robotics and Embedded Systems at Technical University of Munich.

If you use CommonPower, please cite it as: 
```
@article{eichelbeck2023commonpower,
  title={CommonPower: Supercharging machine learning for smart grids},
  author={Eichelbeck, Michael and Markgraf, Hannah and Althoff, Matthias},
  year={2023}
}
```

Installing CommonPower
----------------------

You will need [Python](https://www.python.org/downloads/) >= 3.8 installed on your system.

We recommend using a [virtual environment](https://docs.python.org/3/library/venv.html) to work with CommonPower. 
To create a virtual environment run
```bash
python -m venv </path/to/new/virtual/environment>
```
You can then activate the virtual environment.

Linux: 
```bash
source <path/to/venv>/bin/activate
```

Windows:
```bash
<path/to/venv>\Scripts\activate.bat
```

You can then proceed to install CommonPower.
For local development, install the library in editable mode:
```bash
cd <your/working/directory>
git clone "https://github.com/TUMcps/commonpower.git"
pip install -e <absolute/path/to/the/commonpower/directory>
```

Otherwise, install CommonPower via PyPI:
```bash
pip install commonpower
```

Multi-agent reinforcement learning
----------------------------------

At the moment, CommonPower supports multi-agent reinforcement learning using the IPPO/MAPPO implementation detailed in this [paper](https://arxiv.org/abs/2103.01955). 
Since we had to make a few adjustments, we forked the original repository. Please clone our [fork](https://github.com/TUMcps/on-policy), cd into the repository and install the package to your virtual environment using
`pip install -e .`.

Gurobi
------

We use Gurobi as a default solver for our optimization problems. As a student, you can obtain an [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). 
There are two options: If you want to run CommonPower on you laptop, you can use the named-user license. To run it on a server, you need the WLS license.
After obtaining a license, follow the Gurobi [quickstart guide](https://www.gurobi.com/documentation/quickstart.html) (choose the appropriate one for your system) to install Gurobi and retrieve your license. 
If you use Gurobi on a server (with the WLS license) and receive the error that it requires two many cores, you can just [submit a ticket](https://support.gurobi.com/hc/en-us/requests/new?ticket_form_id=360000629792) with the error message and your WLS license will be upgraded.

Get started
------------

Have a look at the [Introduction Tutorial](https://commonpower.readthedocs.io/en/latest/tutorials/Introduction.html) to learn more about how CommonPower is structured.
