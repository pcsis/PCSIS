from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval

if __name__ == '__main__':
    model = LotkaCSIS
    prob = CSISProblem(model)

    # M, epsilon = 3, 0.5
    # M, epsilon = 3, 0.4
    # M, epsilon = 3, 0.3
    # M, epsilon = 3, 0.2
    # M, epsilon = 3, 0.1
    M, epsilon = 3, 0.0

    prob.set_options(template_type="poly", degree_poly=8, gamma=0.9, alpha=0.005, beta=1e-20, C=-1,
                     coe_lb=-1e3, coe_ub=1e3, M=M, K=10, epsilon=epsilon, N1=1e3, random_seed=0)

    l = 3.5
    safe_set = Interval([-l, -l], [l, l])

    control_set = Interval([-1, -1], [1, 1])
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)

    prob.solve(x_data, fx_data)
    prob.plot()
    prob.get_probability(1e6)

    # prob.get_probability(1e6, iteration=0)




