from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval
import math

if __name__ == '__main__':
    model = Spacecraft_RendezvousCSIS
    prob = CSISProblem(model)

    prob.set_options(template_type="poly", degree_poly=4, lamda=0.999, alpha=0.015, beta=1e-20, C=-0.01, N1=1e3,
                     coe_lb=-1e3, coe_ub=1e3, M=5, K=10, epsilon=0.05, update_epsilon=0.3, random_seed=385,
                     plot_dim=[[0, 1]], plot_project_values=[{2: 0, 3: 0, 4: 0, 5: 0}])

    safe_set = Interval([-1, -1, -1, -1, -1, -1],
                        [1, 1, 1, 1, 1, 1])
    control_set = Interval([-1, -1, -1],
                           [1, 1, 1])
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)
    prob.solve(x_data, fx_data)

    prob.plot()

    prob.get_probability(1e6)
