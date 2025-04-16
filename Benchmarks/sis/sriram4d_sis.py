from pcsis.model import *
from pcsis.sis_problem import SISProblem
from pcsis.interval import Interval
from pcsis.ellipsoid import Ellipsoid
import time

if __name__ == '__main__':
    model = Sriram4DSIS
    prob = SISProblem(model)

    alpha, beta = 0.005, 1e-20
    prob.set_options(degree_poly=6, lamda=0.99, alpha=alpha, beta=beta, C=-1,
                     U_al=1e3, N1=1e3, random_seed=0, plot_project_values=[{2: 0, 3: 0}], obj_sample="random")

    safe_set = Ellipsoid([0.5] * 4, [1 / 4] * 4)

    x_safe, fx_safe, x_unsafe, fx_unsafe = prob.generate_data(safe_set)

    h1 = prob.solve(x_safe, fx_safe, x_unsafe, fx_unsafe)

    prob.get_probability(1e6)

    N2 = 1e6
    start_time = time.time()
    sim_data = prob.monte_carlo(1e6, 5e2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The time required for solution is {elapsed_time} seconds.")
    print(r'P_x [x \in S] = ', len(sim_data) / N2, ', estimated by Monte Carlo method')



