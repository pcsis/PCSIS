from pcsis.model import *
from pcsis.csis_problem import CSISProblem
from pcsis.interval import Interval
import math

if __name__ == '__main__':
    model = PendulumCSIS
    prob = CSISProblem(model)

    # epsilon = 0.3
    # epsilon = 0.2
    # epsilon = 0.1
    epsilon = 0.2
    prob.set_options(degree_poly=6, lamda=0.99, alpha=0.005, beta=1e-20, C=-1, N1=1e3, obj_sample="grid",
                     U_al=1e3,  M=10, K=5, K_prime=10, epsilon=epsilon, epsilon_prime=1, random_seed=0)

    safe_set = Interval([-math.pi * 5 / 6, -4], [math.pi * 5 / 6, 4])
    control_set = Interval(-8, 8)
    x_data, u_data, fx_data = prob.generate_data(safe_set, control_set)

    coe = prob.solve(x_data, fx_data)

    # prob.synthesize_controller(x_data=x_data, u_data=u_data, coe=coe, method="MLP")
    prob.synthesize_controller(x_data=x_data, u_data=u_data, coe=coe, method="Polynomial", poly_degree=4)

    prob.sim_traj([[-2, 3], [-1, 3], [0, 3], [1, 3], [2, 3]], [20, 200, 200, 200, 60])

    prob.plot()

    prob.get_probability(1e6)
    prob.get_probability(1e6, iteration=0)
