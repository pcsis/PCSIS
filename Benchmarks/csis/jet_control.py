from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval

if __name__ == '__main__':
    model = JetControl
    prob = CSISProblem(model)

    # epsilon = 0.3
    # M, epsilon = 10, 0.0
    # M, epsilon = 3, 0.1
    M, epsilon = 3, 0.0

    prob.set_options(degree_poly=7, lamda=0.999, alpha=0.01, beta=1e-20, C=-1, N1=1e3,
                     coe_lb=-1e3, coe_ub=1e3, M=M, K=10, epsilon=epsilon, random_seed=0)

    safe_set = Interval([-3, -3], [3, 3])

    control_set = Interval([-1, -1], [1, 1])
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)

    prob.solve(x_data, fx_data)

    prob.plot()

    prob.get_probability(1e6)
