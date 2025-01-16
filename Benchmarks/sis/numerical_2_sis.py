from pcsis.model import *
from pcsis.sis_problem import SISProblem
from pcsis.interval import Interval
from pcsis.ellipsoid import Ellipsoid

if __name__ == '__main__':
    model = Numerical2SIS
    prob = SISProblem(model)

    alpha, beta = 0.01, 1e-20
    prob.set_options(degree_poly=12, lamda=0.9999, alpha=alpha, beta=beta, C=-1,
                     U_al=1e3, random_seed=0)

    safe_set = Interval([-1, -1], [1, 1])
    x_safe, fx_safe, x_unsafe, fx_unsafe = prob.generate_data(safe_set)
    prob.set_obj(N1=1000)

    h1 = prob.solve(x_safe, fx_safe, x_unsafe, fx_unsafe)
    prob.get_probability(1e6)

    N2 = 1e6
    sim_data = prob.monte_carlo(N2, 3e2)
    print(r'P_x [x \in S] = ', len(sim_data) / N2, ', estimated by Monte Carlo method')

    prob.plot(h1, sim_data=sim_data)











