from __future__ import annotations
from .template import Templates
from .solver import Solver
from .solver_xi import SolverXi
from .interval import Interval
from .ellipsoid import Ellipsoid
from .plot_manager import PlotManager
import numpy as np
import math
import itertools
import time
import warnings


class CSISProblem:
    def __init__(self, model: callable):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model = model()
        self.templates = None
        self.solver = None
        self.solver_xi = None
        self.verbose = None
        self.lamda = -1
        self.alpha = 0
        self.beta = 0
        self.N = 0
        self.M = 0
        self.C = -1
        self.K = 5
        self.K_prime = 10
        self.epsilon = 0.3
        self.epsilon_prime = 0.0
        self.update_epsilon = 1
        self.safe_set = None
        self.control_set = None
        self.h_x = None
        self.h1_fx = None
        self.h_fx = None
        self.is_in_safe_set = None
        self.split_num = None
        self.plot_manager = None
        self.h_str_list = []
        self.coe_list = []
        self.volume_list = []
        self.controller = None
        self.traj = None
        self.N1 = 0
        self.obj_sample = None

    def set_options(self, **kwargs):
        self.verbose = kwargs.get("verbose", 1)

        templ_type = kwargs.get("template_type", "poly")
        degree_poly = kwargs.get("degree_poly", 6)
        degree_ex = kwargs.get("degree_ex", 6)
        self.templates = Templates(degree_systems=self.model.degree_state, temp_type=templ_type,
                                   degree_poly=degree_poly, degree_ex=degree_ex, verbose=self.verbose)

        U_al = kwargs.get("U_al", 1e3)
        self.solver = Solver(num_vars=self.templates.num_vars, coe_lb=-U_al, coe_ub=U_al)
        self.solver_xi = SolverXi(num_vars=self.templates.num_vars + 1, coe_lb=-U_al, coe_ub=U_al)

        self.lamda = kwargs.get("lamda", 0.99)

        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)
        N = kwargs.get("N", None)
        if alpha is not None and beta is not None and N is not None:
            assert alpha >= (2 / N * (math.log(1 / beta) + self.templates.num_vars))
        elif alpha is not None and beta is not None and N is None:
            N = 2 / alpha * (math.log(1 / beta) + self.templates.num_vars)
            N = math.ceil(N)
            print("N: ", N)
        elif alpha is None and beta is not None and N is not None:
            alpha = 2 / N * (math.log(1 / beta) + self.templates.num_vars)
            print("alpha: ", alpha)
        elif alpha is not None and beta is None and N is not None:
            beta = 1 / math.exp(0.5 * alpha * N - self.templates.num_vars)
            print("beta: ", beta)
        else:
            raise ValueError("At least two of alpha, beta, and N must be entered!")
        self.alpha = alpha
        self.beta = beta
        self.N = N

        self.split_num = kwargs.get("M", 10)
        self.M = self.split_num ** self.model.degree_input

        self.C = kwargs.get("C", -100)

        random_seed = kwargs.get("random_seed", False)
        if random_seed is not False:
            np.random.seed(random_seed)

        self.K = kwargs.get("K", 5)
        self.K_prime = kwargs.get("K_prime", 10)

        self.epsilon = kwargs.get("epsilon", 0.3)
        assert 0 <= self.epsilon <= 1
        self.update_epsilon = kwargs.get("update_epsilon", 1)

        self.epsilon_prime = kwargs.get("epsilon_prime", 0.0)

        plot_dim = kwargs.get("plot_dim", [[0, 1]])
        plot_project_values = kwargs.get("plot_project_values", {})
        self.plot_manager = PlotManager(dim=plot_dim, project_values=plot_project_values, K=self.K, is_iteration=True,
                                        v_filled=False, grid=False)
                                        # save=True, prob_name=self.model.__class__.__name__)
        self.N1 = int(kwargs.get("N1", 1000))
        self.obj_sample = kwargs.get("obj_sample", "random")

    def generate_data(self, safe_set, control_set: Interval, method="random"):
        self.safe_set = safe_set
        self.plot_manager.safe_set = safe_set
        self.control_set = control_set
        x_data = None
        fx_data = None
        # if self.verbose >= 1:
        #     print("{} samples are required".format(self.N))
        u_data = [np.linspace(control_set.inf[i], control_set.sup[i], self.split_num) for i in range(len(control_set.inf))]
        u_data = np.array(list(itertools.product(*u_data)))
        if isinstance(safe_set, Interval):
            assert self.model.degree_state == safe_set.inf.shape[0]
            x_data = safe_set.generate_data(self.N, method)
        elif isinstance(safe_set, Ellipsoid):
            assert self.model.degree_state == safe_set.degree
            x_data = safe_set.generate_data(self.N, method)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        fx_data = np.array(self.model.fx(x_data, u_data)).T
        return x_data, u_data, fx_data

    def _categorize_data(self, fx_data, h_fx):
        safe_set = self.safe_set
        h_unsafe = np.zeros((h_fx.shape[1],))
        if isinstance(safe_set, Interval):
            self.is_in_safe_set = np.all((fx_data >= safe_set.inf) & (fx_data <= safe_set.sup), axis=1)
        elif isinstance(safe_set, Ellipsoid):
            self.is_in_safe_set = safe_set.is_in_safe_set(fx_data)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        h_fx[~self.is_in_safe_set] = h_unsafe
        s1, s2 = h_fx.shape
        self.is_in_safe_set = self.is_in_safe_set.reshape(s1 // self.M, self.M)
        return h_fx

    def _set_obj(self, num_obj, method="grid"):
        if isinstance(self.safe_set, Interval):
            obj_data = self.safe_set.generate_data(num_obj, method)
        elif isinstance(self.safe_set, Ellipsoid):
            obj_data = self.safe_set.generate_data(num_obj, method)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        obj_h_x_data = self.templates.calc_values(obj_data)
        self.solver.set_objective(obj_h_x_data)

    def _solve_0(self, x_data, fx_data):
        h_x = self.templates.calc_values(x_data)
        h_fx = self.templates.calc_values(fx_data)

        s1, s2 = h_fx.shape
        h1_fx_reshaped = h_fx.reshape(s1 // self.M, self.M, s2)
        self.h_x = h_x.copy()
        self.h1_fx = h1_fx_reshaped.copy()

        h_fx = self._categorize_data(fx_data, h_fx)
        h_fx_reshaped = h_fx.reshape(s1//self.M, self.M, s2)

        self.h_fx = h_fx_reshaped

        weight = np.full((self.M,), 1/self.M)
        h_fx_sum = np.tensordot(h_fx_reshaped, weight, axes=(1, 0))

        count_true = np.sum(self.is_in_safe_set, axis=1)
        count_false = self.is_in_safe_set.shape[1] - count_true
        C_M = count_false * self.C / self.M

        constraint = self.lamda * h_x - h_fx_sum
        constraint = np.insert(constraint, 0, -1, axis=1)
        self.solver_xi.add_constraint(constraint, C_M)

        solution = self.solver_xi.solve()

        xi, coe = solution[0], solution[1:]
        h_str = self.templates.output(coe)
        print(h_str)
        self.h_str_list.append(h_str)
        self.coe_list.append(coe)
        return xi, coe

    def _solve_xi(self, coe_last):
        weight = np.full((self.M,), self.epsilon / self.M)
        h_fx_other = np.tensordot(self.h_fx, weight, axes=(1, 0))

        count_true = np.sum(self.is_in_safe_set, axis=1)
        count_false = self.is_in_safe_set.shape[1] - count_true
        CE_M = count_false * self.C * self.epsilon / self.M

        max_h_fx = np.dot(self.h1_fx, coe_last)
        max_h_fx[~self.is_in_safe_set] = self.C
        max_u = np.argmax(max_h_fx, axis=1)
        h_fx_max = self.h_fx[np.arange(self.h_fx.shape[0]), max_u]
        check_C = (h_fx_max[:, 0] == 0).astype(int)
        C1_E_M = check_C * self.C * (1.0 - self.epsilon)
        C_M = CE_M + C1_E_M

        h_fx_max = h_fx_max * (1.0 - self.epsilon)

        h_fx_sum = h_fx_other + h_fx_max

        constraint = self.lamda * self.h_x - h_fx_sum
        constraint = np.insert(constraint, 0, -1, axis=1)
        self.solver_xi.clean_constraint()
        self.solver_xi.add_constraint(constraint, C_M)

        # self.solver_xi.set_init_value(coe_last)
        solution = self.solver_xi.solve()
        xi, coe = solution[0], solution[1:]
        h_str = self.templates.output(coe)
        print(h_str)
        self.h_str_list.append(h_str)
        self.coe_list.append(coe)
        return xi, coe

    def _solve_max(self, coe_last):
        weight = np.full((self.M,), self.epsilon / self.M)
        h_fx_other = np.tensordot(self.h_fx, weight, axes=(1, 0))

        count_true = np.sum(self.is_in_safe_set, axis=1)
        count_false = self.is_in_safe_set.shape[1] - count_true
        CE_M = count_false * self.C * self.epsilon / self.M

        max_h_fx = np.dot(self.h1_fx, coe_last)
        max_h_fx[~self.is_in_safe_set] = self.C
        max_u = np.argmax(max_h_fx, axis=1)
        h_fx_max = self.h_fx[np.arange(self.h_fx.shape[0]), max_u]
        check_C = (h_fx_max[:, 0] == 0).astype(int)
        C1_E_M = check_C * self.C * (1.0 - self.epsilon)
        C_M = CE_M + C1_E_M

        h_fx_max = h_fx_max * (1.0 - self.epsilon)

        h_fx_sum = h_fx_other + h_fx_max

        constraint = self.lamda * self.h_x - h_fx_sum
        self.solver.clean_constraint()
        self.solver.add_constraint(constraint, C_M)

        # out_v_x = np.dot(self.h_x, coe_last)
        # out_x_index = out_v_x > 0
        # h_x_cons = self.h_x[out_x_index]
        # h_x_cons = -h_x_cons
        # h_x_cons[:, 0] += 0.01
        # self.solver.add_constraint(h_x_cons)

        # out_v_x = np.dot(self.h_x, coe_last)
        # out_x_index = out_v_x < 0
        # h_x_obj = self.h_x[out_x_index]
        # self.solver.set_objective(h_x_obj)

        # self.solver.set_init_value(coe_last)

        solution, obj_max = self.solver.solve()
        h_str = self.templates.output(solution)
        print(h_str)
        self.h_str_list.append(h_str)
        self.coe_list.append(solution)
        return solution, obj_max

    def calc_volume(self, coe, iteration):
        value_v_x = np.dot(self.h_x, coe)
        volume = np.count_nonzero(value_v_x > 0)
        print('Iteration {}: The invariant set contains {} states (out of a total of {} states)'
              .format(iteration, volume, self.N))
        return volume

    def get_probability(self, num_sample, iteration=None):
        num_sample = int(num_sample)
        if isinstance(self.safe_set, Interval):
            sample_data = np.random.uniform(self.safe_set.inf, self.safe_set.sup, size=(num_sample, len(self.safe_set.inf)))
        elif isinstance(self.safe_set, Ellipsoid):
            sample_data = self.safe_set.generate_data(num_sample, "random")
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        h_x_data = self.templates.calc_values(sample_data)

        if iteration is None:
            index = np.argmax(self.volume_list)
        else:
            index = iteration
        value_v_x = np.dot(h_x_data, self.coe_list[index])
        volume = np.count_nonzero(value_v_x > 0) / num_sample
        print(r'P_x [x \in S] = ', volume, ', estimated by Monte Carlo method')
        probability = 1 - self.alpha / volume
        print(r'1 - alpha / P_x [x \in S] = ', probability, ', estimated by Monte Carlo method')

        return volume

    def solve(self, x_data, fx_data):
        start_time = time.time()
        self._set_obj(self.N1, self.obj_sample)
        iteration = 0

        xi, coe_last = self._solve_0(x_data, fx_data)
        if xi <= 1e-8:
            volume = self.calc_volume(coe_last, iteration)
            self.volume_list.append(volume)
        else:
            self.volume_list.append(0.0)
        print()

        while xi > 0 and iteration < self.K:
            xi, coe_last = self._solve_xi(coe_last)
            iteration += 1
            self.volume_list.append(self.calc_volume(coe_last, iteration))
            print()

        if iteration >= self.K and xi > 0:
            print("Error!")

        # coe_last = self._solve_0_max(x_data, fx_data)
        # self.calc_volume(coe_last, iteration)

        obj_max_last, obj_max = -np.infty, np.infty
        while iteration < self.K_prime and abs(obj_max - obj_max_last) > self.epsilon_prime:
            obj_max_last = obj_max
            coe_last, obj_max = self._solve_max(coe_last)
            iteration += 1
            self.volume_list.append(self.calc_volume(coe_last, iteration))
            print()

            self.epsilon = self.epsilon * self.update_epsilon

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The time required for solution is {elapsed_time} seconds.")
        # print(f'The volume of the final invariant set is {max(self.volume_list)}')

        return coe_last

    def synthesize_controller(self, x_data, u_data, coe, method="Polynomial", poly_degree=4):
        max_h_fx = np.dot(self.h1_fx, coe)
        max_h_fx[~self.is_in_safe_set] = self.C
        max_u = np.argmax(max_h_fx, axis=1)

        u_data = np.repeat(u_data.T, self.N, axis=0)
        y = u_data[np.arange(u_data.shape[0]), max_u.flatten()]

        if method == "Polynomial":
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
        elif method == "MLP":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0)
        else:
            raise NotImplementedError()
        model.fit(x_data, y)
        self.controller = model

    def sim_traj(self, init_state, sim_time):
        if self.model.degree_state != 2:
            raise NotImplementedError()
        traj_all = []
        for i in range(len(init_state)):
            traj = [init_state[i]]
            x = np.array(init_state[i]).reshape(1, -1)
            for j in range(sim_time[i]):
                u = self.controller.predict(x).reshape(1, -1)
                x = self.model.fx(x, u)
                x = np.array(x)
                x = x.reshape(2)
                traj.append(x)
                x = x.reshape(1, -1)
            traj_all.append(traj)
        self.traj = traj_all

    def plot(self):
        for i in range(len(self.volume_list)):
            if self.volume_list[i] > 0:
                self.plot_manager.add_v(safe_set=self.safe_set, v_str=self.h_str_list[i])
        self.plot_manager.add_safe_set(safe_set=self.safe_set)

        if self.traj is not None:
            for i in self.traj:
                self.plot_manager.add_traj(i)
        self.plot_manager.show()

    # def _solve_0_max(self, x_data, fx_data):
    #     h_x = self.templates.calc_values(x_data)
    #     h_fx = self.templates.calc_values(fx_data)
    #
    #     s1, s2 = h_fx.shape
    #     h1_fx_reshaped = h_fx.reshape(s1 // self.M, self.M, s2)
    #     self.h_x = h_x.copy()
    #     self.h1_fx = h1_fx_reshaped.copy()
    #
    #     h_fx = self._categorize_data(fx_data, h_fx)
    #     h_fx_reshaped = h_fx.reshape(s1 // self.M, self.M, s2)
    #
    #     self.h_fx = h_fx_reshaped
    #
    #     weight = np.full((self.M,), 1/self.M)
    #     h_fx_sum = np.tensordot(h_fx_reshaped, weight, axes=(1, 0))
    #
    #     count_true = np.sum(self.is_in_safe_set, axis=1)
    #     count_false = self.is_in_safe_set.shape[1] - count_true
    #     C_M = count_false * self.C / self.M
    #
    #     constraint = self.lamda * h_x - h_fx_sum
    #     self.solver.add_constraint(constraint, C_M)
    #     self.solver.set_objective(h_x)
    #     solution = self.solver.solve()
    #     h_str = self.templates.output(solution)
    #     print(h_str)
    #     self.plot_manager.add_v(safe_set=self.safe_set, v_str=h_str)
    #
    #     return solution



