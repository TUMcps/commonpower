from typing import Callable

import pyomo.contrib.pyros as pyros
import pyomo.environ as pyo
from pyomo.core import ConcreteModel
from pyomo.opt import TerminationCondition
from pyomo.opt.solver import OptSolver

from commonpower.modeling.base import ElementTypes, ModelEntity
from commonpower.modeling.util import get_element_from_model
from commonpower.utils.default_solver import get_default_solver


def _filter_elements_from_registry(mdl: ConcreteModel, fcn: Callable, with_info: bool = False) -> list[ModelEntity]:

    if with_info:
        return [
            (get_element_from_model(name=el_id, model=mdl, local_id=el_id.split(".")[-1], global_id=el_id), el_info)
            for el_id, el_info in mdl.REGISTRY.items()
            if fcn(el_info) is True
        ]
    else:
        return [
            get_element_from_model(name=el_id, model=mdl, local_id=el_id.split(".")[-1], global_id=el_id)
            for el_id, el_info in mdl.REGISTRY.items()
            if fcn(el_info) is True
        ]


def robust_solve(mdl: ConcreteModel, global_solver: OptSolver) -> bool:

    uncertainty_info = [
        x
        for x in zip(
            *_filter_elements_from_registry(
                mdl=mdl,
                fcn=lambda el_info: (el_info["type"] == ElementTypes.DATA or el_info["type"] == ElementTypes.CONSTANT)
                and el_info["uncertainty_bounds"] is not None,
                with_info=True,
            )
        )
    ]

    if not uncertainty_info:
        # standard solve without uncertainties
        result = global_solver.solve(mdl, warmstart=True)
        mdl.solutions.store_to(result)
        if result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            return True
        else:
            return False

    uncertain_params, uncertain_params_info = uncertainty_info

    uncertainty_bounds = []
    for el_info in uncertain_params_info:
        el_uncertainty_bounds = el_info["uncertainty_bounds"]
        if isinstance(el_uncertainty_bounds, list):  # Indexed parameter
            uncertainty_bounds += el_uncertainty_bounds
        else:
            uncertainty_bounds.append(el_uncertainty_bounds)

    uncertainty_set = pyros.BoxSet(uncertainty_bounds)

    """ model_vars = _filter_elements_from_registry(
        mdl=mdl,
        fcn=lambda el_info: el_info["type"] == ElementTypes.VAR,
    ) """

    model_inputs = _filter_elements_from_registry(
        mdl=mdl,
        fcn=lambda el_info: el_info["type"] == ElementTypes.INPUT,
    )

    pyros_solver: pyros.PyROS = pyo.SolverFactory("pyros")

    local_solver = get_default_solver()

    global_solver.options.update({"NonConvex": 2})
    local_solver.options.update({"NonConvex": 2})

    result = pyros_solver.solve(
        model=mdl,
        first_stage_variables=[],
        second_stage_variables=model_inputs,
        uncertain_params=uncertain_params,
        uncertainty_set=uncertainty_set,
        local_solver=local_solver,
        global_solver=global_solver,
        load_solution=True,  # Load the solution back into the model
        solve_master_globally=True,
    )

    if result.pyros_termination_condition not in [
        pyros.pyrosTerminationCondition.robust_feasible,
        pyros.pyrosTerminationCondition.robust_optimal,
    ]:
        return True

    return False
