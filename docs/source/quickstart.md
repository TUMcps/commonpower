Quick-start
===========

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
Since we had to make a few adjustments, we forked the original repository. Please clone our [fork](https://github.com/Chrisschmit/on-policy), cd into the repository and install the package to your virtual environment using
`pip install -e .`.

Gurobi
------

We use Gurobi as a default solver for our optimization problems. As a student, you can obtain an [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). 
There are to options: If you want to run CommonPower on you laptop, you can use the named-user license. To run it on a server, you need the WLS license.
After obtaining a license, follow the Gurobi [quickstart guide](https://www.gurobi.com/documentation/quickstart.html) (choose the appropriate one for your system) to install Gurobi and retrieve your license. 
If you use Gurobi on a server (with the WLS license) and receive the error that it requires two many cores, you can just [submit a ticket](https://support.gurobi.com/hc/en-us/requests/new?ticket_form_id=360000629792) with the error message and your WLS license will be upgraded.


Get started
------------

Have a look at the Introduction tutorial to learn more about how CommonPower is structured.
