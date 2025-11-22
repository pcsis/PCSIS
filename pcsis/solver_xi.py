from __future__ import annotations
import gurobipy as gp
import numpy as np


class SolverXi:
    def __init__(self, name: str = "PAC_CBF", num_vars: int = 0, coe_lb=-1e3, coe_ub=1e3, coe_lambda=10,
                 verbose: int = 0):
        self.solver_model = gp.Model(name)
        self.solver_model.setParam('OutputFlag', verbose)
        # self.solver_model.setParam('MIPGap', 1e-8)
        # self.solver_model.setParam('OptimalityTol', 1e-8)
        # self.solver_model.setParam('FeasibilityTol', 1e-6)
        self.num_vars = num_vars

        # lb = np.full(num_vars, coe_lb)
        # ub = np.full(num_vars, coe_ub)
        # lb[0] = 0.0
        # ub[0] = coe_lambda
        # self.var: [lambda, coe]
        self.var = self.solver_model.addMVar(shape=(num_vars,), lb=coe_lb, ub=coe_ub,
                                             vtype=gp.GRB.CONTINUOUS, name="var")

    def set_init_value(self, init_value):
        self.var.start = init_value

    def clean_constraint(self):
        self.solver_model.remove(self.solver_model.getConstrs())
        self.solver_model.update()

    def add_constraint(self, cons, constant=0):
        cons_xi = np.zeros(self.num_vars)
        cons_xi[0] = -1
        cons_xi = cons_xi.reshape(1, self.num_vars)
        cons = np.append(cons, cons_xi, axis=0)
        if isinstance(constant, np.ndarray):
            constant = np.append(constant, 0)

        self.solver_model.addConstr(cons @ self.var <= constant, name="cons")

    def add_constraint_x0(self, cons_x_0, epsilon_0):
        self.solver_model.addConstr(cons_x_0 @ self.var - epsilon_0 >= 0, "cons_x0")

    def solve(self):
        obj = np.zeros(self.num_vars)
        obj[0] = 1
        self.solver_model.setObjective(obj @ self.var, gp.GRB.MINIMIZE)

        self.solver_model.update()
        self.solver_model.optimize()

        assert self.solver_model.status == gp.GRB.OPTIMAL
        print("Optimal objective: ", self.solver_model.objVal)

        return self.var.X
