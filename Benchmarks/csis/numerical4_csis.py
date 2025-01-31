from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval

if __name__ == '__main__':
    model = Numerical4CSIS
    prob = CSISProblem(model)

    # M, epsilon = 3, 0.5
    M, epsilon = 3, 0.4
    # M, epsilon = 3, 0.3
    # M, epsilon = 3, 0.2
    # M, epsilon = 3, 0.1
    # M, epsilon = 3, 0.0

    prob.set_options(degree_poly=5, lamda=0.9999, alpha=0.01, beta=1e-20, C=-1,
                     coe_lb=-1e3, coe_ub=1e3, M=M, K=10, epsilon=epsilon, N1=1e3, random_seed=0)

    # safe_set = Interval([-1, -1], [1, 1])
    safe_set = Interval([-2, -2], [2, 2])

    control_set = Interval([-2, -2], [2, 2])
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)

    prob.solve(x_data, fx_data)

    prob.plot()

    prob.get_probability(1e6)

    prob.get_probability(1e6, iteration=0)

