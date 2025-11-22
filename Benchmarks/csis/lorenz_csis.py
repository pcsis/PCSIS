from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval

if __name__ == '__main__':
    model = LorenzCSIS
    prob = CSISProblem(model)

    M, epsilon = 2, 0.1
    # M, epsilon = 2, 0.0

    prob.set_options(degree_poly=2, gamma=0.9999, alpha=0.01, beta=1e-20, C=-1, N1=1e3,
                     coe_lb=-1e3, coe_ub=1e3, M=M, K=10, epsilon=epsilon, random_seed=0)

    safe_set = Interval([-15] * 12, [15] * 12)

    u = 10
    control_set = Interval([-u, -u, -u], [u, u, u])
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)

    coe = prob.solve(x_data, fx_data)

    prob.get_probability(1e6, coe)