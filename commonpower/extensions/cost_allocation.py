"""
Module for cost allocation in energy communities.
"""
import logging
import random
from datetime import datetime, timedelta
from math import factorial
from typing import List, Union

import numpy as np
from pydantic import BaseModel
from pyomo.core import ConcreteModel, Constraint, Objective, Param, Set, Var, minimize, quicksum, value

from commonpower.control.controllers import BaseController, OptimalController
from commonpower.control.runners import DeploymentRunner
from commonpower.core import System
from commonpower.data_forecasting.forecasters import PerfectKnowledgeForecaster
from commonpower.models.buses import EnergyCommunity
from commonpower.utils.default_solver import get_default_solver


class ShapleySimulationResult(BaseModel):
    """
    Result of a Shapley simulation, containing the realized cost,
    Shapley values, stand-alone costs, and additional info.
    """

    realized_cost: float
    shapleys: list[float]
    stand_alone_costs: list[float]
    info: dict


class BaseShapleySimulator:
    """
    Base Shapley simulator class, providing internal logic.
    """

    def __init__(
        self,
        sys: System,
        community_controller: OptimalController,
        horizon: timedelta = timedelta(days=1),
        dt: timedelta = timedelta(minutes=60),
        n_sample_coalitions: int = 100,
        seed: int = None,
    ):
        """
        Simulator that simulates one (possibly) imperfect knowledge system and computes Shapley values.
        The given system is a template representing the entire energy community (grand coalition).
        For the simulation of coalitions, the template system is cloned and rearranged accordingly.

        Args:
            sys (System): (Template) system to simulate. Must contain exactly one top-level EnergyCommunity node.
            community_controller (OptimalController): (Template) controller for the community.
            horizon (timedelta, optional): Forecast horizon. Defaults to timedelta(days=1).
            dt (timedelta, optional): Sample time aka frequency. Defaults to timedelta(minutes=60).
            n_sample_coalitions (int, optional): Number of coalitions to sample. Defaults to 100.
            seed (int, optional): Seed for the coalition sampling. Defaults to None.
        """

        community_idx = [i for i, node in enumerate(sys.nodes) if isinstance(node, EnergyCommunity)]
        assert len(community_idx) == 1, "There must be exactly one EnergyCommunity in the system."
        self.community_idx = community_idx[0]

        self.sys = sys
        self.community_controller = community_controller
        self.horizon = horizon
        self.dt = dt
        self.n_sample_coalitions = n_sample_coalitions
        self.seed = seed

    def _get_coalitions(self, grand_coalition: List[int]) -> List[List[int]]:
        """
        Genenerate sub-coalitions of the grand coalition efficiently.
        Utilizes self.seed and returns self.n_sample_coalitions coalitions.

        Args:
            grand_coalition (List[int]): Grand coalition of players.

        Returns:
            List[List[int]]: List of coalitions.
        """
        if self.seed is not None:
            random.seed(self.seed)

        def perm_generator(seq):
            seen = set()
            length = len(seq)
            while True:
                perm = tuple(random.sample(seq, length))
                if perm not in seen:
                    seen.add(perm)
                    yield perm

        perm_gen = perm_generator(grand_coalition)

        coalitions = [
            list(next(perm_gen)) for _ in range(min(self.n_sample_coalitions, factorial(len(grand_coalition))))
        ]

        return coalitions

    def _get_coalitions_for_i(self, coalitions: List[List[int]], grand_coalition: List[int], i: int) -> List[List[int]]:
        """
        Get all coalitions without player i, including the empty coalition.
        For any set of players, we return only one ordered coalition.

        Args:
            coalitions (List[List[int]]): Sampled coalitions.
            grand_coalition (List[int]): Grand coalition of players.
            i (int): Index of player.

        Returns:
            List[List[int]]: List of ordered coalitions without player i.
        """
        coalitions_for_i = []
        grand_coalition_without_i = grand_coalition.copy()
        grand_coalition_without_i.remove(i)
        coalitions_for_i.append(grand_coalition_without_i)

        for c in coalitions:
            if c.index(i) != 0:
                coalitions_for_i.append(c[: c.index(i)])

        coalitions_for_i.append([])

        # remove duplicates
        coalitions_for_i = [list(c) for c in list(set(tuple(sorted(i)) for i in coalitions_for_i))]

        return coalitions_for_i

    def _get_shapleys(
        self,
        sys: System,
        sim_steps: int,
        fixed_start: datetime,
    ) -> tuple[List[float], dict, dict]:
        """
        Compute Shapley values for the given system.
        (1) We sample coalitions from the powerset of the grand coalition.
        (2) For each player i, we compute the marginal contribution of i to each coalition.
        (3) We average the marginal contributions over all coalitions to get the Shapley value for i.

        We reuse the costs of previously simulated coalitions during the process.
        We apply a normalization step to ensure efficiency, i.e.,
        the sum of Shapley values equals the cost of the grand coalition.

        Args:
            sys (System): (Template) system to simulate.
            sim_steps (int): Number of simulation steps.
            fixed_start (datetime): Start time of the simulation.

        Returns:
            tuple[List[float], dict, dict]: Shapley values, costs of coalitions, deployment runners of coalitions.
        """
        grand_coalition = self._get_grand_coalition()
        n = len(grand_coalition)  # grand coalition size
        # Might have to change this if we want to model community assets
        coalitions = self._get_coalitions(grand_coalition)
        cost_dict = {}  # dict to store the cost of each coalition
        runner_dict = {}
        shapleys = []  # list to store the shapley value for each player

        for i in grand_coalition:
            # For each ordering get the set of players before i
            # also add empty coalition and grand coalition if not already present
            coalitions_for_i = self._get_coalitions_for_i(coalitions, grand_coalition, i)

            shapley_i = 0
            for count, c in enumerate(coalitions_for_i):
                logging.info(f"Coalition {count} of {len(coalitions_for_i)} for player {i}")
                s = len(c)
                weight = (factorial(s) * factorial(n - s - 1)) / (factorial(n))

                ci_str = str(sorted(c + [i]))
                if ci_str in cost_dict:
                    cost_with_i = cost_dict[ci_str]
                else:
                    cost_with_i, runner_with_i = self._get_cost_for_coalition(
                        sys, sorted(c + [i]), sim_steps, fixed_start
                    )

                    cost_dict[ci_str] = cost_with_i
                    runner_dict[ci_str] = runner_with_i

                c_str = str(sorted(c))
                if c_str in cost_dict:
                    cost_without_i = cost_dict[c_str]
                else:
                    if c:
                        cost_without_i, runner_without_i = self._get_cost_for_coalition(
                            sys, sorted(c), sim_steps, fixed_start
                        )
                    else:  # empty coalition
                        cost_without_i = 0

                    cost_dict[c_str] = cost_without_i
                    runner_dict[c_str] = runner_without_i

                mc = cost_with_i - cost_without_i
                shapley_i += weight * mc

            shapleys.append(shapley_i)

        # We need an adjustment to ensure efficiency (after sampling)
        grand_coalition_cost = cost_dict[str(sorted(grand_coalition))]
        shapley_sum = sum(shapleys)
        shapleys = [(shapley / shapley_sum) * grand_coalition_cost for shapley in shapleys]
        return shapleys, cost_dict, runner_dict

    def _get_cost_for_coalition(
        self,
        sys: System,
        coalition: List[int],
        sim_steps: int,
        fixed_start: datetime,
    ) -> tuple[float, DeploymentRunner]:
        """
        Simulate a coalition and return the total cost.
        The computed cost includes the "value" of the terminal state,
        which is computed as perfect knowledge cost over the horizon from the terminal time step.
        This makes different trajectory costs more comparable.
        The template system is cloned, and rearanged to represent the coalition.

        Args:
            sys (System): (Template) system to simulate.
            coalition (List[int]): Indices of the coalition members.
            sim_steps (int): Number of simulation steps.
            fixed_start (datetime): Start time of the simulation.

        Returns:
            tuple[float, DeploymentRunner]: Total cost of the coalition, deployment runner of the coalition.
        """
        local_sys = sys.empty_copy()
        local_coalition_controller = self.community_controller.empty_copy()
        local_community = local_sys.nodes[self.community_idx]

        # prune community to current coalition
        self._get_sub_coalition(local_sys, local_community, coalition)
        # configure community with controller
        local_coalition_controller.add_entity(local_community)

        my_glob_cntrl = OptimalController("global")
        local_deployment_runner = DeploymentRunner(
            local_sys, global_controller=my_glob_cntrl, horizon=self.horizon, dt=self.dt, seed=self.seed
        )

        local_deployment_runner.run(n_steps=sim_steps, fixed_start=fixed_start)

        costs_df = (
            local_deployment_runner.history.filter_for_entities(local_community, False)
            .filter_for_element_names(["cost"])
            .to_df()
        )
        coalition_cost = sum(costs_df[local_community.get_pyomo_element_id("cost")].tolist())

        # coalition_cost = sum(get_adjusted_cost(local_deployment_runner.history, local_community))

        return coalition_cost, local_deployment_runner

    def _get_sub_coalition(self, sys: System, community: EnergyCommunity, coalition: List[int]) -> None:
        """
        Rearange the system to represent the given coalition.
        (1) Prune the community.
        (2) Add the pruned nodes to the system.

        Args:
            sys (System): Reference to the system.
            community (EnergyCommunity): Reference to the community.
            coalition (List[int]): Indices of the coalition members.
        """
        for i in sorted(list(range(len(community.nodes))), reverse=True):
            if i not in coalition:
                community.nodes[i].set_as_stand_alone()
                sys.nodes.append(community.nodes[i])
                community.nodes.pop(i)

    def _get_grand_coalition(self) -> List[int]:
        """
        Get the indices of the grand coalition.

        Returns:
            List[int]: Grand coalition indices.
        """
        return list(range(len(self.sys.nodes[self.community_idx].nodes)))

    def simulate(
        self,
        sim_steps: int,
        fixed_start: datetime,
        **kwargs,
    ) -> ShapleySimulationResult:
        """
        Simulate the system and compute Shapley values.
        Must be implemented by subclasses.

        Args:
            sim_steps (int): Number of simulation steps.
            fixed_start (datetime): Start time of the simulation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class NativeShapleySimulator(BaseShapleySimulator):
    """
    This simulator simulates the (possibly) imperfect knowledge system
    and computes Shapley values based on the realized costs.
    """

    def simulate(
        self,
        sim_steps: int,
        fixed_start: datetime,
    ) -> ShapleySimulationResult:
        """
        Simulate the system and compute a simulation result, including
        the realized cost, Shapley values, and stand-alone costs.

        Args:
            sim_steps (int): Number of simulation steps.
            fixed_start (datetime): Start time of the simulation.

        Returns:
            ShapleySimulationResult: Result of the simulation.
        """
        shapleys, cost_dict, _ = self._get_shapleys(self.sys, sim_steps, fixed_start)

        grand_coalition = self._get_grand_coalition()
        realized_cost = cost_dict[str(sorted(grand_coalition))]

        stand_alone_costs = [cost_dict[str([i])] for i in grand_coalition]

        return ShapleySimulationResult(
            realized_cost=realized_cost,
            shapleys=shapleys,
            stand_alone_costs=stand_alone_costs,
            info={},
        )


class UncertaintyAwareShapleySimulator(BaseShapleySimulator):
    """
    This simulator simulates the imperfect knowledge system and computes Shapley values
    based on the perfect knowledge system.
    """

    def __init__(
        self,
        sys: System,
        community_controller: BaseController,
        horizon: timedelta = timedelta(days=1),
        dt: timedelta = timedelta(minutes=60),
        n_sample_coalitions: int = 100,
        seed=None,
    ):
        """
        Simulator that simulates the imperfect knowledge system to obtain the realized cost
        but computes Shapley values based on the perfect knowledge system.
        This can be used for allocation methods that make this distinction.

        The given system is a template representing the entire energy community (grand coalition).
        For the simulation of coalitions, the template system is cloned and rearranged accordingly.

        Args:
            sys (System): (Template) system to simulate.
            community_controller (BaseController): (Template) controller for the community.
            horizon (timedelta, optional): Forecast horizon. Defaults to timedelta(days=1).
            dt (timedelta, optional): Sample time aka frequency. Defaults to timedelta(minutes=60).
            n_sample_coalitions (int, optional): Number of coalitions to sample. Defaults to 100.
            seed (int, optional): Seed for the coalition sampling. Defaults to None.
        """
        super().__init__(sys, community_controller, horizon, dt, n_sample_coalitions, seed)
        self.sys_pk = self._make_sys_perfect_knowledge(sys)

    def _make_sys_perfect_knowledge(self, sys: System) -> System:
        """
        Make a perfect knowledge system by replacing all existing forecasters with perfect knowledge forecasters.

        Args:
            sys (System): _description_

        Returns:
            System: _description_
        """
        sys_pk = sys.empty_copy()

        all_entities = sys_pk.get_children()
        for entity in all_entities:
            for dp in entity.data_providers:
                pk_forecaster = PerfectKnowledgeForecaster(
                    frequency=dp.frequency,
                    horizon=dp.horizon,
                )
                dp.forecaster = pk_forecaster

        return sys_pk

    def simulate(
        self,
        sim_steps: int,
        fixed_start: datetime,
    ) -> ShapleySimulationResult:
        """
        Simulate the system and compute a simulation result, including
        the realized cost, Shapley values (perfect knowledge), and stand-alone costs (perfect knowledge).

        Args:
            sim_steps (int): Number of simulation steps.
            fixed_start (datetime): Start time of the simulation.

        Returns:
            ShapleySimulationResult: Simulation result.
        """
        grand_coalition = self._get_grand_coalition()

        realized_cost, _ = self._get_cost_for_coalition(self.sys, grand_coalition, sim_steps, fixed_start)

        shapleys, cost_dict, _ = self._get_shapleys(self.sys_pk, sim_steps, fixed_start)

        stand_alone_costs = [cost_dict[str([i])] for i in grand_coalition]

        return ShapleySimulationResult(
            realized_cost=realized_cost,
            shapleys=shapleys,
            stand_alone_costs=stand_alone_costs,
            info={},
        )


class CostAllocator:
    """
    Cost allocator based directly on Shapley values of imperfect knowledge system.
    """

    def __init__(
        self,
        simulation_result: ShapleySimulationResult,
    ):
        """
        Cost allocator.

        Args:
            simulation_result (ShapleySimulationResult): Simulation result from ShapleySimulator.
        """
        self.simulation_result = simulation_result

    def allocate(self, **kwargs) -> list[float]:
        """
        The cost for each player is her Shapley value.

        Returns:
            list[float]: Cost allocation.
        """
        if not self.simulation_result:
            raise AttributeError("No simulation result available. Please run the simulation first.")

        return self.simulation_result.shapleys


class POIKCostAllocator(CostAllocator):
    """
    Pareto optimal imperfect knowledge cost allocation, based on the paper [1].

    [1] M. Eichelbeck and M. Althoff, "Fair Cost Allocation in Energy Communities under Forecast Uncertainty,"
        in IEEE Open Access Journal of Power and Energy, 2024, doi: 10.1109/OAJPE.2024.3520418.
    """

    def allocate(
        self,
        p: int = 2,
        mu: float = 1.0,
    ) -> list[float]:
        """
        Allocate costs based on the Pareto optimal imperfect knowledge method.

        Args:
            p (int, optional): Norm-order of the objective function representing minimum deviation.
                The value -1 represents the infinity norm. Defaults to 2.
            mu (float, optional): Objective function weight. Defaults to 1.0.

        Returns:
            list[float]: Cost allocation.
        """
        res: dict = self.sweep(p, mu)[0]
        return res[f"p{p}_mu{mu}"]

    def sweep(
        self,
        p: Union[int, List[int]],  # order of norm for pk payment deviation (-1 for infinity norm)
        mu: Union[float, List[float]],  # weight coefficient for pk payment deviation
    ) -> tuple[dict[str, tuple[list[float]]], list[tuple[float, float]]]:
        """
        Sweep over a list of norm orders and weight coefficients to obtain multiple cost allocations.
        This can be used to obtain a Pareto front.

        Args:
            p (Union[int, List[int]]): Norm-order of the objective function representing minimum deviation.
                The value -1 represents the infinity norm.
            mu (Union[float, List[float]]): Objective function weight.

        Returns:
            tuple[dict[str, tuple[list[float]]], list[tuple[float, float]]]: The first tuple contains
                (allocated payments, coi shares, regrets) for each combination of p and mu.
                The second tuple contains the normalized objective function values for each Pareto point.
        """

        p = [p] if not isinstance(p, list) else p
        mu = [mu] if not isinstance(mu, list) else mu

        result_dict = {}
        pareto_points = []

        # all _solve operations return (payments, gammas, regrets), (normalized_gamma, normalized_z)

        result_1_norm, pareto_point1 = self._solve_1_norm()
        gammas_1_norm = result_1_norm[1]
        regrets_1_norm = result_1_norm[2]

        for p in p:

            for mu in mu:

                if p == 1:  # 1-norm
                    res = result_1_norm
                    pareto_point = pareto_point1
                elif p == 0 or p < -1:
                    raise AttributeError(f"invalid norm order {p}")
                elif p == -1:  # infinity norm
                    res, pareto_point = self._solve_inf_norm(mu, gammas_1_norm, regrets_1_norm)
                else:
                    res, pareto_point = self._solve_p_norm(mu, p, gammas_1_norm, regrets_1_norm)

                result_dict[f"p{p}_mu{mu}"] = res

                pareto_points.append(pareto_point)

        return result_dict, pareto_points

    def _solve_inf_norm(
        self,
        mu: float,
        gammas_1_norm: List[float],
        regrets_1_norm: List[float],
    ) -> tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]:
        """
        Compute the Pareto optimal cost allocation for the infinity norm.
        This is a linear program.

        Args:
            mu (float): Objective function weight.
            gammas_1_norm (List[float]): Worst-case gamma values.
            regrets_1_norm (List[float]): Best-case regret values.

        Returns:
            tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]: Allocation result, consisting of
                (payments, coi shares, regrets), (normalized_gamma, normalized_z)
        """
        coi = self.simulation_result.realized_cost - sum(self.simulation_result.shapleys)
        N = len(self.simulation_result.shapleys)
        regrets_pk = [
            self.simulation_result.shapleys[i] - self.simulation_result.stand_alone_costs[i] for i in range(N)
        ]
        max_regret_pk = max(regrets_pk)

        mdl = ConcreteModel()
        mdl.N = Set(initialize=range(N))
        mdl.gamma = Var(mdl.N, bounds=(0, 1))
        mdl.e = Var(mdl.N)
        mdl.z = Var()  # upper bound for e
        mdl.y = Var()  # upper bound for gamma

        mdl.pk_cost = Param(mdl.N, initialize={i: self.simulation_result.shapleys[i] for i in mdl.N})
        mdl.stand_alone_cost = Param(mdl.N, initialize={i: self.simulation_result.stand_alone_costs[i] for i in mdl.N})
        mdl.coi = Param(initialize=coi)

        mdl.mu = Param(initialize=mu)

        mdl.c1 = Constraint(expr=sum(mdl.gamma[i] for i in mdl.N) == 1)
        mdl.c2 = Constraint(
            mdl.N,
            expr=lambda mdl, i: mdl.e[i] == mdl.pk_cost[i] - mdl.stand_alone_cost[i] + mdl.gamma[i] * mdl.coi,
        )
        mdl.c3 = Constraint(mdl.N, expr=lambda mdl, i: mdl.e[i] <= mdl.z)
        mdl.c4 = Constraint(mdl.N, expr=lambda mdl, i: mdl.gamma[i] <= mdl.y)

        # scaling for objective
        range_e = max_regret_pk + coi / N - max(regrets_1_norm)
        range_gamma = max(gammas_1_norm) - 1 / N

        mdl.obj = Objective(
            expr=(((1 - mdl.mu) * mdl.z) / range_e + (mdl.mu * mdl.y) / range_gamma),
            sense=minimize,
        )

        solver = get_default_solver()
        solver.solve(mdl)

        gammas = [mdl.gamma[i].value for i in mdl.N]
        coi_shares = [gammas[i] * coi for i in mdl.N]

        payments = [coi_shares[i] + self.simulation_result.shapleys[i] for i in mdl.N]
        regrets = [payments[i] - self.simulation_result.stand_alone_costs[i] for i in mdl.N]

        normalized_z = (value(mdl.z) - max(regrets_1_norm)) / range_e
        # obtain the values for the normalized cost functions
        normalized_gamma = (value(mdl.y) - (1 / N)) / range_gamma

        return (payments, gammas, regrets), (normalized_gamma, normalized_z)

    def _solve_1_norm(self) -> tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]:
        """
        Compute the Pareto optimal cost allocation for the 1-norm.
        Here, the norm of gamma is constant (1) and can therefore be omitted in the objective function.
        By definition this represents the worst-case minimum deviation and best-case minimum regret.
        This is a linear program.

        Returns:
            tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]: Allocation result, consisting of
                (payments, coi shares, regrets), (normalized_gamma, normalized_z)
        """

        coi = self.simulation_result.realized_cost - sum(self.simulation_result.shapleys)
        N = len(self.simulation_result.shapleys)

        mdl = ConcreteModel()
        mdl.N = Set(initialize=range(N))
        mdl.gamma = Var(mdl.N, bounds=(0, 1))
        mdl.e = Var(mdl.N)
        mdl.z = Var()

        mdl.pk_cost = Param(mdl.N, initialize={i: self.simulation_result.shapleys[i] for i in mdl.N})
        mdl.stand_alone_cost = Param(mdl.N, initialize={i: self.simulation_result.stand_alone_costs[i] for i in mdl.N})
        mdl.coi = Param(initialize=coi)

        mdl.c1 = Constraint(expr=sum(mdl.gamma[i] for i in mdl.N) == 1)
        mdl.c2 = Constraint(
            mdl.N,
            expr=lambda mdl, i: mdl.e[i] == mdl.pk_cost[i] - mdl.stand_alone_cost[i] + mdl.gamma[i] * mdl.coi,
        )
        mdl.c3 = Constraint(mdl.N, expr=lambda mdl, i: mdl.e[i] <= mdl.z)
        mdl.obj = Objective(
            expr=(mdl.z),
            sense=minimize,
        )

        solver = get_default_solver()
        solver.solve(mdl)

        gammas = [value(mdl.gamma[i]) for i in mdl.N]
        coi_shares = [gammas[i] * coi for i in mdl.N]

        payments = [coi_shares[i] + self.simulation_result.shapleys[i] for i in mdl.N]

        normalized_z = 0  # by definition
        normalized_gamma = 1  # by definition

        return (payments, gammas, [payments[i] - self.simulation_result.stand_alone_costs[i] for i in mdl.N]), (
            normalized_gamma,
            normalized_z,
        )

    def _solve_p_norm(
        self,
        mu: float,
        p: int,
        gammas_1_norm: List[float],
        regrets_1_norm: List[float],
    ) -> tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]:
        """
        Compute the Pareto optimal cost allocation for some p-norm.
        This is a qudratic program for p=2.

        Args:
            mu (float): Objective function weight.
            p (int): Norm-order of the objective function representing minimum deviation.
            gammas_1_norm (List[float]): Worst-case gamma values.
            regrets_1_norm (List[float]): Best-case regret values.

        Returns:
            tuple[tuple[List[float], List[float], List[float]], tuple[float, float]]: Allocation result, consisting of
                (payments, coi shares, regrets), (normalized_gamma, normalized_z)
        """

        coi = self.simulation_result.realized_cost - sum(self.simulation_result.shapleys)
        N = len(self.simulation_result.shapleys)
        regrets_pk = [
            self.simulation_result.shapleys[i] - self.simulation_result.stand_alone_costs[i] for i in range(N)
        ]
        max_regret_pk = max(regrets_pk)
        max_regret_1norm = max(regrets_1_norm)

        mdl = ConcreteModel()
        mdl.N = Set(initialize=range(N))
        mdl.gamma = Var(mdl.N, bounds=(0, 1))
        mdl.e = Var(mdl.N)
        mdl.z = Var()

        mdl.pk_cost = Param(mdl.N, initialize={i: self.simulation_result.shapleys[i] for i in mdl.N})
        mdl.stand_alone_cost = Param(mdl.N, initialize={i: self.simulation_result.stand_alone_costs[i] for i in mdl.N})
        mdl.coi = Param(initialize=coi)

        mdl.mu = Param(initialize=mu)

        mdl.c1 = Constraint(expr=sum(mdl.gamma[i] for i in mdl.N) == 1)
        mdl.c2 = Constraint(
            mdl.N,
            expr=lambda mdl, i: mdl.e[i] == mdl.pk_cost[i] - mdl.stand_alone_cost[i] + mdl.gamma[i] * mdl.coi,
        )
        mdl.c3 = Constraint(mdl.N, expr=lambda mdl, i: mdl.e[i] <= mdl.z)

        range_e = max_regret_pk + coi / N - max_regret_1norm
        # to avoid the p-root in the objective, we exponentiate the gamma objective by p
        range_gamma = np.linalg.norm(gammas_1_norm, p) ** p - N ** (1 - p)

        mdl.obj = Objective(
            expr=(
                ((1 - mdl.mu) * mdl.z) / range_e + (mdl.mu * quicksum(mdl.gamma[i] ** p for i in mdl.N)) / range_gamma
            ),
            sense=minimize,
        )

        solver = get_default_solver()
        solver.solve(mdl)

        gammas = [mdl.gamma[i].value for i in mdl.N]
        coi_shares = [gammas[i] * coi for i in mdl.N]

        payments = [coi_shares[i] + self.simulation_result.shapleys[i] for i in mdl.N]

        normalized_z = (value(mdl.z) - max_regret_1norm) / range_e
        # obtain the values for the normalized cost functions by exponentiating the gamma objective by 1/p
        normalized_gamma = ((value(quicksum(mdl.gamma[i] ** p for i in mdl.N)) - (N ** (1 - p))) / range_gamma) ** (
            1 / p
        )

        return (
            payments,
            gammas,
            [payments[i] - self.simulation_result.stand_alone_costs[i] for i in mdl.N],
        ), (normalized_gamma, normalized_z)
