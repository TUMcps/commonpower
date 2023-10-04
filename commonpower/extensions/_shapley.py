from itertools import chain, combinations
from math import factorial
from random import uniform


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1)))


def shapley_values(players: list[int], fcn: callable):

    n = len(players)
    coalitions = [list(c) for c in powerset(players)]

    shapleys = []

    for i in range(len(players)):
        coalitions_without_i = [c for c in coalitions if i not in c]
        shapley_i = 0
        for c in coalitions_without_i:
            s = len(c)
            weight = (factorial(s) * factorial(n - s - 1)) / (factorial(n))
            mc = fcn(c + [i]) - fcn(c)
            shapley_i += weight * mc
        shapleys.append(shapley_i)

    return shapleys


if __name__ == "__main__":

    n = range(4)
    demand = [uniform(-5, 5) for _ in n]

    def cost_fcn(c: list[int]):

        if not c:
            return 0

        overall_demand = sum([demand[i] for i in c])
        if overall_demand >= 0:
            overall_cost = 0.5 * overall_demand + 0.3 * overall_demand**6 + 1
        else:
            overall_cost = 0.2 * overall_demand

        return overall_cost

    vals = shapley_values(n, cost_fcn)

    print(vals)
    print(sum(vals))
    print(cost_fcn(n))
