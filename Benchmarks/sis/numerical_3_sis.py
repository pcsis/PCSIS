from pcsis.model import *
from pcsis.sis_problem import SISProblem
from pcsis.interval import Interval
from pcsis.ellipsoid import Ellipsoid

if __name__ == '__main__':
    model = Numerical3SIS
    prob = SISProblem(model)

    alpha, beta = 0.01, 1e-20
    prob.set_options(degree_poly=4, lamda=0.9999, alpha=alpha, beta=beta, C=-1,
                     U_al=1e3, random_seed=0,
                     plot_dim=[[3, 4]], plot_project_values=[{1: 0, 5: 0, 0: 0, 2: 0}])
                   # plot_dim=[[0, 1]], plot_project_values=[{2: 0, 3: 0, 4: 0, 5: 0}])

    safe_set = Interval([-0.5] * 6, [2] * 6)
    # safe_set = Ellipsoid([-0.5] * 6, [1/4] * 6)

    x_safe, fx_safe, x_unsafe, fx_unsafe = prob.generate_data(safe_set)
    prob.set_obj(N1=1000)

    h1 = prob.solve(x_safe, fx_safe, x_unsafe, fx_unsafe)

    prob.get_probability(1e6)

    N2 = 1e6
    sim_data = prob.monte_carlo(N2, 3e2)
    print(r'P_x [x \in S] = ', len(sim_data) / N2, ', estimated by Monte Carlo method')

    # sim_data_proj = prob.monte_carlo(N2, 3e2, proj_dim=[3, 4])
    # sim_data_proj = np.array([sim_data_proj[:, 3], sim_data_proj[:, 4]]).T
    # prob.plot(h1, sim_data=sim_data_proj)
