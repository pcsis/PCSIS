from __future__ import annotations
from .template import Templates
from .solver import Solver
from .interval import Interval
from .ellipsoid import Ellipsoid
from .plot_manager import PlotManager
import numpy as np
import math
import time
import warnings


class SISProblem:
    def __init__(self, model: callable):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model = model()

        self.templates = None
        self.solver = None
        self.verbose = None
        self.gamma = -1
        self.alpha = 0
        self.beta = 0
        self.N = 0
        self.C = -1
        self.safe_set = None
        self.plot_manager = None
        self.traj = None
        self.coe = None
        self.N1 = 0
        self.obj_sample = None
        self.epsilon_0 = 1e-6
        self.x0 = None

    def set_options(self, **kwargs):
        self.verbose = kwargs.get("verbose", 1)

        templ_type = kwargs.get("template_type", "poly")
        degree_poly = kwargs.get("degree_poly", 6)
        degree_ex = kwargs.get("degree_ex", 6)
        self.templates = Templates(degree_systems=self.model.degree_state, temp_type=templ_type,
                                   degree_poly=degree_poly, degree_ex=degree_ex, verbose=self.verbose)

        coe_b = kwargs.get("U_al", 1e3)
        self.solver = Solver(num_vars=self.templates.num_vars, coe_lb=-coe_b, coe_ub=coe_b)

        self.gamma = kwargs.get("gamma", 0.99)

        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)
        N = kwargs.get("N", None)
        num_vars = self.templates.num_vars + 1
        if alpha is not None and beta is not None and N is not None:
            assert alpha >= (2 / N * (math.log(1 / beta) + num_vars))
        elif alpha is not None and beta is not None and N is None:
            N = 2 / alpha * (math.log(1 / beta) + num_vars)
            N = math.ceil(N)
            print("N: ", N)
        elif alpha is None and beta is not None and N is not None:
            alpha = 2 / N * (math.log(1 / beta) + num_vars)
            print("alpha: ", alpha)
        elif alpha is not None and beta is None and N is not None:
            beta = 1 / math.exp(0.5 * alpha * N - num_vars)
            print("beta: ", beta)
        else:
            raise ValueError("At least two of alpha, beta, and N must be entered!")
        self.alpha = alpha
        self.beta = beta
        self.N = N

        self.C = kwargs.get("C", -100)

        random_seed = kwargs.get("random_seed", False)
        if random_seed is not False:
            np.random.seed(random_seed)

        plot_dim = kwargs.get("plot_dim", [[0, 1]])
        plot_project_values = kwargs.get("plot_project_values", {})
        self.plot_manager = PlotManager(dim=plot_dim, project_values=plot_project_values, grid=False,
                                        v_filled=True,
                                        save=True, prob_name=self.model.__class__.__name__, save_file='jpg')

        self.N1 = int(kwargs.get("N1", 1000))
        self.obj_sample = kwargs.get("obj_sample", "random")

        self.epsilon_0 = kwargs.get("epsilon_0", 1e-6)
        x0 = kwargs.get("x0", np.zeros(self.model.degree_state))
        self.x0 = x0.reshape(1, -1)

    def generate_data(self, safe_set, method="random"):
        self.safe_set = safe_set
        self.plot_manager.safe_set = safe_set
        x_data = None
        fx_data = None
        # if self.verbose >= 1:
        #     print("{} samples are required".format(self.N))
        if isinstance(safe_set, Interval):
            assert self.model.degree_state == safe_set.inf.shape[0]
            x_data = safe_set.generate_data(self.N, method)
        elif isinstance(safe_set, Ellipsoid):
            assert self.model.degree_state == safe_set.degree
            x_data = safe_set.generate_data(self.N, method)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        fx_data = np.array(self.model.fx(x_data, None)).T
        return self._categorize_data(x_data, fx_data, safe_set)

    def _categorize_data(self, x_data, fx_data, safe_set):
        is_in_safe_set = []
        if isinstance(safe_set, Interval):
            is_in_safe_set = np.all((fx_data >= safe_set.inf) & (fx_data <= safe_set.sup), axis=1)
        elif isinstance(safe_set, Ellipsoid):
            is_in_safe_set = safe_set.is_in_safe_set(fx_data)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        x_safe = x_data[is_in_safe_set]
        fx_safe = fx_data[is_in_safe_set]
        x_unsafe = x_data[~is_in_safe_set]
        fx_unsafe = fx_data[~is_in_safe_set]
        return x_safe, fx_safe, x_unsafe, fx_unsafe

    # def simu_traj(self, init_state, steps):
    #     traj = [init_state[0, :]]
    #     state = init_state
    #     for i in range(steps):
    #         state = self.model.fx(state, None)
    #         state = np.array(state).reshape([1, 2])
    #         traj.append(state[0, :])
    #     return traj

    def calc_volume(self, h_x, coe):
        value_v_x = np.dot(h_x, coe)
        volume = np.count_nonzero(value_v_x > 0)
        print('The invariant set contains {} states (out of a total of {} states)'
              .format(volume, self.N))

    def _set_obj(self, N1, method="grid"):
        if isinstance(self.safe_set, Interval):
            obj_data = self.safe_set.generate_data(N1, method)
        elif isinstance(self.safe_set, Ellipsoid):
            obj_data = self.safe_set.generate_data(N1, method)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        obj_h_x_data = self.templates.calc_values(obj_data)
        self.solver.set_objective(obj_h_x_data)

    def solve(self, x_safe, fx_safe, x_unsafe, fx_unsafe):
        start_time = time.time()
        self._set_obj(self.N1, self.obj_sample)
        h_x_safe = self.templates.calc_values(x_safe)
        h_fx_safe = self.templates.calc_values(fx_safe)

        h_x_unsafe = self.templates.calc_values(x_unsafe)

        constraint_safe = self.gamma * h_x_safe - h_fx_safe
        constraint_unsafe = self.gamma * h_x_unsafe

        self.solver.add_constraint_verification(constraint_safe, constraint_unsafe, self.C)

        h_x_0 = self.templates.calc_values(self.x0)
        self.solver.add_constraint_x0(h_x_0, self.epsilon_0)

        solution, obj_max = self.solver.solve()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The time required for solution is {elapsed_time} seconds.")

        self.coe = solution
        h_str = self.templates.output(solution)
        print("The function h1(x) is: ")
        print(h_str)

        # self.calc_volume(np.vstack((h_x_safe, h_x_unsafe)), solution)

        return h_str

    def plot(self, h_str, sim_data=None):
        if sim_data is not None:
            self.plot_manager.add_monte_carlo(sim_data)

        self.plot_manager.add_v(safe_set=self.safe_set, v_str=h_str)
        self.plot_manager.add_safe_set(safe_set=self.safe_set)

        if self.traj is not None:
            for i in self.traj:
                self.plot_manager.add_traj(i)

        self.plot_manager.show()

    def monte_carlo(self, num_sample, sim_time, proj_dim=None):
        num_sample, sim_time = int(num_sample), int(sim_time)
        is_in_safe_set = np.ones(num_sample, dtype=bool)

        if proj_dim is None:
            if isinstance(self.safe_set, Interval):
                x_data = np.random.uniform(self.safe_set.inf, self.safe_set.sup, size=(num_sample, len(self.safe_set.inf)))
            elif isinstance(self.safe_set, Ellipsoid):
                x_data = self.safe_set.generate_data(num_sample, "random")
            else:
                raise NotImplementedError("Sampling within this type of safe set is not implemented")
        else:
            if isinstance(self.safe_set, Interval):
                inf_proj = np.array([self.safe_set.inf[proj_dim[0]], self.safe_set.inf[proj_dim[1]]])
                sup_proj = np.array([self.safe_set.sup[proj_dim[0]], self.safe_set.sup[proj_dim[1]]])
                x_data_proj = np.random.uniform(inf_proj, sup_proj, size=(num_sample, 2))
                x_data = np.zeros(shape=(num_sample, len(self.safe_set.inf)))
                x_data[:, proj_dim[0]] = x_data_proj[:, 0]
                x_data[:, proj_dim[1]] = x_data_proj[:, 1]
            else:
                raise NotImplementedError("Sampling within this type of safe set is not implemented")

        x_data_i = x_data.copy()

        for i in range(sim_time):
            fx_data = np.array(self.model.fx(x_data_i, None)).T
            if isinstance(self.safe_set, Interval):
                is_in_safe_set_i = np.all((fx_data >= self.safe_set.inf) & (fx_data <= self.safe_set.sup), axis=1)
            elif isinstance(self.safe_set, Ellipsoid):
                is_in_safe_set_i = self.safe_set.is_in_safe_set(fx_data)
            else:
                raise NotImplementedError("Sampling within this type of safe set is not implemented")
            is_in_safe_set = is_in_safe_set & is_in_safe_set_i

            x_data_i = fx_data

        x_data = x_data[is_in_safe_set]
        return x_data

    def get_probability(self, num_sample):
        num_sample = int(num_sample)
        if isinstance(self.safe_set, Interval):
            sample_data = np.random.uniform(self.safe_set.inf, self.safe_set.sup, size=(num_sample, len(self.safe_set.inf)))
        elif isinstance(self.safe_set, Ellipsoid):
            sample_data = self.safe_set.generate_data(num_sample, "random")
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        h_x_data = self.templates.calc_values(sample_data)

        value_v_x = np.dot(h_x_data, self.coe)
        volume = np.count_nonzero(value_v_x > 0) / num_sample
        print(r'P_x [x \in \tilde{S}] =', volume)
        probability = 1 - self.alpha / volume
        print(r'1 - alpha / P_x [x \in S] = ', probability, ', estimated by Monte Carlo method')
        return volume

    def sim_traj(self, init_state, sim_time):
        if self.model.degree_state != 2:
            raise NotImplementedError()
        traj_all = []
        for i in range(len(init_state)):
            traj = [init_state[i]]
            x = np.array(init_state[i]).reshape(1, -1)
            for j in range(sim_time[i]):
                x = self.model.fx(x)
                x = np.array(x)
                x = x.reshape(2)
                traj.append(x)
                x = x.reshape(1, -1)
            traj_all.append(traj)
        self.traj = traj_all
