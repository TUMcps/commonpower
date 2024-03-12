from pyomo.environ import SolverFactory
from pyomo.opt.solver import OptSolver

#: Default solver options for Gurobi solver
DEFAULT_SOLVER_OPTIONS = {
    "MIPFocus": 1,  #: focus on feasible solutions not optimality
    "IntFeasTol": 1e-9,  #: integrality tolerance. Important for MIPs, see MIPExpressionBuilder
    "TimeLimit": 60,  #: limit solve time to 60s
}


def get_default_solver() -> OptSolver:
    """
    Returns a solver instance with certain option values.
    """
    solver = SolverFactory('gurobi', solver_io='python', manage_env=True)
    solver.options.update(DEFAULT_SOLVER_OPTIONS)

    return solver
