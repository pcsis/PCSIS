from pcsis.model import *
from pcsis.sis_problem import SISProblem
from pcsis.interval import Interval
from pcsis.ellipsoid import Ellipsoid

if __name__ == '__main__':
    model = Numerical1SIS
    prob = SISProblem(model)

    alpha, beta = 0.005, 0.0001
    prob.set_options(degree_poly=16, lamda=0.9999, alpha=alpha, beta=beta, C=-1,
                     U_al=1e3, random_seed=0)

    safe_set = Ellipsoid([0, 0], [1, 1])
    x_safe, fx_safe, x_unsafe, fx_unsafe = prob.generate_data(safe_set)
    prob.set_obj(N1=1000)

    h1 = prob.solve(x_safe, fx_safe, x_unsafe, fx_unsafe)
    prob.get_probability(1e6)

    sim_data = prob.monte_carlo(1e6, 1e2)
    print(r'P_x [x \in S] = ', len(sim_data) / 1e6, ', estimated by Monte Carlo method')

    prob.plot(h1, sim_data=sim_data)



