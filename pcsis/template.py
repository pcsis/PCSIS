import math
import numpy as np
from pcsis.interval import Interval
import itertools


class Templates:
    def __init__(self, degree_systems: int, temp_type: str = "poly", degree_poly: int = 6, degree_ex: int = 6, verbose: bool = True):
        if not isinstance(degree_systems, int):
            raise TypeError("'num_variables' must be of integer type!")

        self.degree_systems = degree_systems
        self.temp_type = temp_type
        self.verbose = verbose
        self.num_vars = 0

        if temp_type == "poly":
            if not isinstance(degree_poly, int):
                raise TypeError("'poly_degree' must be of integer type!")
            self.degree_poly = degree_poly
            self.num_vars = math.comb(self.degree_systems + self.degree_poly, self.degree_poly)
        elif temp_type == "ex":
            if not isinstance(degree_ex, int):
                raise TypeError("'ex_degree' must be of integer type!")
            self.degree_ex = degree_ex
            self.num_vars = math.comb(self.degree_systems + self.degree_ex, self.degree_ex)

    def calc_values(self, data):
        result = None
        if self.temp_type == "poly":
            result = self._calc_values_poly(data)
        elif self.temp_type == "ex":
            result = self._calc_values_ex(data)
        return result

    def _calc_values_poly(self, data):
        terms = []

        for degree_combination in itertools.product(range(self.degree_poly + 1), repeat=self.degree_systems):
            if sum(degree_combination) <= self.degree_poly:
                terms.append(degree_combination)

        terms = np.array(terms)

        num_points = data.shape[0]
        num_terms = terms.shape[0]

        result = np.ones((num_points, num_terms))
        for i, degree_combination in enumerate(terms):
            for j in range(self.degree_systems):
                result[:, i] *= data[:, j] ** degree_combination[j]

        return result

    def _calc_values_ex(self, data):
        data = np.exp(data)
        terms = []

        for degree_combination in itertools.product(range(self.degree_ex + 1), repeat=self.degree_systems):
            if sum(degree_combination) <= self.degree_ex:
                terms.append(degree_combination)

        terms = np.array(terms)

        num_points = data.shape[0]
        num_terms = terms.shape[0]

        result = np.ones((num_points, num_terms))
        for i, degree_combination in enumerate(terms):
            for j in range(self.degree_systems):
                result[:, i] *= data[:, j] ** degree_combination[j]

        return result

    def output(self, coefficients):
        if self.temp_type == "poly":
            return self._output_poly(coefficients)
        elif self.temp_type == "ex":
            return self._output_ex(coefficients)

    def _output_poly(self, coefficients):
        terms = []
        index = 0

        for powers in itertools.product(range(self.degree_poly + 1), repeat=self.degree_systems):
            if sum(powers) <= self.degree_poly:
                coef = coefficients[index]
                term_parts = []
                for var_idx, power in enumerate(powers):
                    if power > 0:
                        term_parts.append(f"x{var_idx}^{power}" if power > 1 else f"x{var_idx}")
                term_str = " * ".join(term_parts)
                if term_str:
                    terms.append(f"{coef:+g} * {term_str}")
                else:
                    terms.append(f"{coef:+g}")
                index += 1

        polynomial_str = " ".join(terms)
        if polynomial_str.startswith("+"):
            polynomial_str = polynomial_str[1:].strip()

        return polynomial_str

    def _output_ex(self, coefficients):
        terms = []
        index = 0

        for powers in itertools.product(range(self.degree_ex + 1), repeat=self.degree_systems):
            if sum(powers) <= self.degree_ex:
                coef = coefficients[index]
                term_parts = []
                for var_idx, power in enumerate(powers):
                    if power > 0:
                        term_parts.append(f"exp(x{var_idx})^{power}" if power > 1 else f"exp(x{var_idx})")
                term_str = " * ".join(term_parts)
                if term_str:
                    terms.append(f"{coef:+g} * {term_str}")
                else:
                    terms.append(f"{coef:+g}")
                index += 1

        polynomial_str = " ".join(terms)
        if polynomial_str.startswith("+"):
            polynomial_str = polynomial_str[1:].strip()

        return polynomial_str