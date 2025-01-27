from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from numbers import Real


class Interval:
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = inf if isinstance(inf, np.ndarray) else np.atleast_1d(inf).astype(float)
        sup = sup if isinstance(sup, np.ndarray) else np.atleast_1d(sup).astype(float)
        assert inf.shape == sup.shape
        mask = np.logical_not(np.isnan(inf) | np.isnan(sup))  # NAN indicates empty
        assert np.all(inf[mask] <= sup[mask])
        self._inf = inf
        self._sup = sup
        self._degree = self._inf.shape[0]

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        """
        center of this interval
        :return:
        """
        return (self._inf + self._sup) * 0.5

    @property
    def rad(self) -> np.ndarray:
        """
        radius of this interval
        :return:
        """
        return (self._sup - self._inf) * 0.5

    @property
    def inf(self) -> np.ndarray:
        return self._inf

    @property
    def sup(self) -> np.ndarray:
        return self._sup

    @property
    def bd(self) -> np.ndarray:
        return np.stack([self.inf, self.sup]).T

    @property
    def T(self):
        """
        shorthand for transpose the interval
        :return:
        """
        return self.transpose()

    @property
    def shape(self) -> tuple:
        return self._inf.shape

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def is_empty(self) -> np.ndarray:
        return np.logical_or(np.isnan(self._inf), np.isnan(self._sup))

    @property
    def info(self):
        info = "Interval BEGIN \n"
        info += ">>> dimension \n"
        info += str(self.shape) + "\n"
        info += str(self.inf) + "\n"
        info += str(self.sup) + "\n"
        info += "Interval END \n"
        return info

    def __str__(self):
        return self.info

    # =============================================== operations
    def __getitem__(self, item):
        inf, sup = self.inf[item], self.sup[item]
        inf = inf if isinstance(inf, np.ndarray) else [inf]
        sup = sup if isinstance(sup, np.ndarray) else [sup]
        return Interval(inf, sup)

    def __setitem__(self, key, value):
        def _setitem_by_interval(x: Interval):
            self._inf[key] = x.inf
            self._sup[key] = x.sup

        def _setitem_by_number(x: (Real, np.ndarray)):
            self._inf[key] = x
            self._sup[key] = x

        if isinstance(value, Interval):
            _setitem_by_interval(value)
        elif isinstance(value, (Real, np.ndarray)):
            _setitem_by_number(value)
        else:
            raise NotImplementedError

    def __add__(self, other):
        def _add_interval(x: Interval):
            return Interval(self.inf + x.inf, self.sup + x.sup)

        if isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf + other, self.sup + other)
        elif isinstance(other, Interval):
            return _add_interval(other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        def _sub_interval(x: Interval):
            assert np.allclose(self.shape, x.shape)
            return Interval(self.inf - x.sup, self.sup - x.inf)

        if isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf - other, self.sup - other)
        elif isinstance(other, Interval):
            return _sub_interval(other)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self - other

    def __pos__(self):
        return self

    def __neg__(self):
        return Interval(-self.sup, -self.inf)

    def __mul__(self, other):
        def _mul_interval(x: Interval):
            bd = np.stack(
                [self.inf * x.inf, self.inf * x.sup, self.sup * x.inf, self.sup * x.sup]
            )
            inf, sup = np.min(bd, axis=0), np.max(bd, axis=0)
            return Interval(inf, sup)

        def _mul_real(x: (Real, np.ndarray)):
            inff, supp = self.inf * x, self.sup * x
            inf, sup = np.minimum(inff, supp), np.maximum(inff, supp)
            return Interval(inf, sup)

        if isinstance(other, (Real, np.ndarray)):
            return _mul_real(other)
        elif isinstance(other, Interval):
            return _mul_interval(other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, (Real, np.ndarray)):
            return self * other
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        def _rtdiv_real(x: (Real, np.ndarray)):
            if x == 1:
                inf = np.full_like(self.inf, np.nan)
                sup = np.full_like(self.sup, np.nan)
                ind0, ind1 = self._inf < 0, self._sup > 0
                # empty set if [0,0] by default

                # [1/u,1/l] if 0 not in [l, u]
                ind = (self._inf > 0) | (self._sup < 0)
                inf[ind] = 1 / self._sup[ind]
                sup[ind] = 1 / self._inf[ind]

                # [1/u,+inf] if l==0 and u>0
                ind = (self._inf == 0) & ind1
                inf[ind] = 1 / self._sup[ind]
                sup[ind] = np.inf

                # [-inf,1/l] if l<0 and u==0
                ind = ind0 & (self._sup == 0)
                inf[ind] = -np.inf
                sup[ind] = 1 / self._inf[ind]

                # [-inf,+inf] if l<0 and u>0
                ind = ind0 & ind1
                inf[ind] = -np.inf
                sup[ind] = np.inf

                return Interval(inf, sup)
            else:
                return x * (1 / self)

        if isinstance(other, (Real, np.ndarray)):
            return _rtdiv_real(other)
        else:
            raise NotImplementedError

    def __itruediv__(self, other):
        return self / other

    # =============================================== non-periodic functions

    def __matmul__(self, other):
        def _matmul_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0] = 0
            negx[x < 0] = x[x < 0]
            inf = self.inf @ posx + self.sup @ negx
            sup = self.sup @ posx + self.inf @ negx
            return Interval(inf, sup)

        def _matmul_interval(x: Interval):
            def _mm(l, r):
                if l.ndim == 1 and r.ndim == 1:
                    return l * r, 0
                elif l.ndim == 1 and r.ndim == 2:
                    return l[..., None] * r, 0
                elif l.ndim == 1 and r.ndim > 2:
                    return l[..., None] * r, -2
                elif l.ndim >= 2 and r.ndim == 1:
                    return l * r[None, ...], -1
                else:
                    return l[..., np.newaxis] * r[..., np.newaxis, :, :], -2

            def _mmm(la, lb, ra, rb):
                ll, lr = _mm(la, lb)
                rl, rr = _mm(ra, rb)
                assert lr == rr
                return np.sum(np.maximum(ll, rl), axis=lr)

            def posneg(m):
                pos, neg = m, -m
                pos[pos < 0] = 0
                neg[neg < 0] = 0
                return pos, neg

            (linfp, linfn), (lsupp, lsupn) = posneg(self.inf), posneg(self.sup)
            (rinfp, rinfn), (rsupp, rsupn) = posneg(x.inf), posneg(x.sup)
            inf = _mmm(linfp, rinfp, lsupn, rsupn) - _mmm(lsupp, rinfn, linfn, rsupp)
            sup = _mmm(lsupp, rsupp, linfn, rinfn) - _mmm(linfp, rsupn, lsupn, rinfp)
            return Interval(inf, sup)

        if isinstance(other, np.ndarray):
            return _matmul_matrix(other)
        elif isinstance(other, Interval):
            return _matmul_interval(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        def _rmm_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0] = 0
            negx[x < 0] = x[x < 0]
            inf = posx @ self.inf + negx @ self.sup
            sup = posx @ self.sup + negx @ self.inf
            return Interval(inf, sup)

        if isinstance(other, np.ndarray):
            return _rmm_matrix(other)
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        return self @ other

    def __abs__(self):
        inf, sup = self.inf, self.sup

        ind = self._sup < 0
        inf[ind], sup[ind] = abs(self._sup[ind]), abs(self._inf[ind])

        ind = (self._inf <= 0) & (self._sup >= 0)
        inf[ind] = 0
        sup[ind] = np.maximum(abs(self._inf[ind]), abs(self._sup[ind]))

        return Interval(inf, sup)

    def __pow__(self, power, modulo=None):
        def _pow_int(x: int):
            if x >= 0:
                inff, supp = self.inf ** x, self.sup ** x
                inf, sup = np.minimum(inff, supp), np.maximum(inff, supp)
                if x % 2 == 0 and x != 0:
                    ind = (self._inf <= 0) & (self._sup >= 0)
                    inf[ind] = 0
                return Interval(inf, sup)
            else:
                return (1 / self) ** (-x)

        def _pow_real(x):
            if x >= 0:
                inf, sup = self.inf ** x, self.sup ** x
                ind = self._inf < 0
                inf[ind] = np.nan
                sup[ind] = np.nan
                return Interval(inf, sup)
            else:
                return (1 / self) ** (-x)

        def _pow_num(x):
            if abs(round(x) - x) <= np.finfo(float).eps:
                return _pow_int(int(x))
            else:
                return _pow_real(x)

        if isinstance(power, (Real, int)):
            return _pow_num(power)
        else:
            raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        return self ** other

    @staticmethod
    def exp(x: Interval):
        return Interval(np.exp(x.inf), np.exp(x.sup))

    @staticmethod
    def log(x: Interval):
        inf, sup = np.log(x.inf), np.log(x.sup)

        ind = (x.inf < 0) & (x.sup >= 0)
        inf[ind] = np.nan

        ind = x.sup < 0
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def sqrt(x: Interval):
        inf, sup = np.sqrt(x.inf), np.sqrt(x.sup)

        ind = (x.inf < 0) & (x.sup >= 0)
        inf[ind] = np.nan

        ind = x.sup < 0
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arcsin(x: Interval):
        inf, sup = np.arcsin(x.inf), np.arcsin(x.sup)

        ind = (x.inf >= -1) & (x.inf <= 1) & (x.sup > 1)
        sup[ind] = np.nan

        ind = (x.inf < -1) & (x.sup >= -1) & (x.sup <= 1)
        inf[ind] = np.nan

        ind = (x.inf < -1) & (x.sup > 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arccos(x: Interval):
        inf, sup = np.arccos(x.sup), np.arccos(x.inf)

        ind = (x.inf >= -1) & (x.inf <= 1) & (x.sup > 1)
        sup[ind] = np.nan

        ind = (x.inf < -1) & (x.sup >= -1) & (x.sup <= 1)
        inf[ind] = np.nan

        ind = (x.inf < -1) & (x.sup > 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arctan(x: Interval):
        return Interval(np.arctan(x.inf), np.arctan(x.sup))

    @staticmethod
    def sinh(x: Interval):
        return Interval(np.sinh(x.inf), np.sinh(x.sup))

    @staticmethod
    def cosh(x: Interval):
        inf, sup = np.cosh(x.sup), np.cosh(x.inf)

        ind = (x.inf <= 0) & (x.sup >= 0)
        inf[ind] = 1
        sup[ind] = np.cosh(np.maximum(abs(x.inf[ind]), abs(x.sup[ind])))

        ind = x.inf > 0
        inf[ind] = np.cosh(x.inf[ind])
        sup[ind] = np.cosh(x.sup[ind])

        return Interval(inf, sup)

    @staticmethod
    def tanh(x: Interval):
        return Interval(np.tanh(x.inf), np.tanh(x.sup))

    @staticmethod
    def arcsinh(x: Interval):
        return Interval(np.arcsinh(x.inf), np.arcsinh(x.sup))

    @staticmethod
    def arccosh(x: Interval):
        inf, sup = np.arccosh(x.inf), np.arccosh(x.sup)

        ind = (x.inf < 1) & (x.sup >= 1)
        inf[ind] = np.nan

        ind = x.sup < 1
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arctanh(x: Interval):
        inf, sup = np.arctanh(x.inf), np.arctanh(x.sup)

        ind = (x.inf > -1) & (x.inf < 1) & (x.sup >= 1)
        sup[ind] = np.nan

        ind = (x.inf <= -1) & (x.sup > -1) & (x.sup < 1)
        inf[ind] = np.nan

        ind = (x.inf <= -1) & (x.sup >= 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def sigmoid(x: Interval):
        return 1 / (1 + Interval.exp(-x))

    # =============================================== periodic functions

    # @staticmethod
    # def sin(x: Interval):
    #     print(x)
    #     print(x - np.pi * 0.5)
    #     return Interval.cos(x - np.pi * 0.5)

    @staticmethod
    def sin(x: Interval):
        ind0 = (x.sup - x.inf) >= 2 * np.pi  # xsup -xinf >= 2*pi
        yinf, ysup = np.mod(x.inf, np.pi * 2), np.mod(x.sup, np.pi * 2)

        ind1 = yinf < np.pi * 0.5  # yinf in R1
        ind2 = ysup < np.pi * 0.5  # ysup in R1
        ind3 = np.logical_not(ind1) & (yinf < np.pi * 1.5)  # yinf in R2
        ind4 = np.logical_not(ind2) & (ysup < np.pi * 1.5)  # ysup in R2
        ind5 = yinf >= np.pi * 1.5  # yinf in R3
        ind6 = ysup >= np.pi * 1.5  # ysup in R3
        ind7 = yinf > ysup  # yinf > ysup
        ind8 = np.logical_not(ind7)  # yinf <=ysup

        inf, sup = x.inf, x.sup

        ind = (ind1 & ind2 & ind8) | (ind5 & ind2) | (ind5 & ind6 & ind8)
        inf[ind] = np.sin(yinf[ind])
        sup[ind] = np.sin(ysup[ind])

        ind = (ind1 & ind4) | (ind5 & ind4)
        inf[ind] = np.minimum(np.sin(yinf[ind]), np.sin(ysup[ind]))
        sup[ind] = 1

        ind = (ind3 & ind2) | (ind3 & ind6)
        inf[ind] = -1
        sup[ind] = np.maximum(np.sin(yinf[ind]), np.sin(ysup[ind]))

        ind = ind3 & ind4 & ind8
        inf[ind] = np.sin(ysup[ind])
        sup[ind] = np.sin(yinf[ind])

        ind = (
                ind0
                | (ind1 & ind2 & ind7)
                | (ind1 & ind6)
                | (ind3 & ind4 & ind7)
                | (ind5 & ind6 & ind7)
        )
        inf[ind] = -1
        sup[ind] = 1

        return Interval(inf, sup)

    # @staticmethod
    # def cos(x: Interval):
    #
    #     def aux_max_cos(I: Interval):
    #         k = np.ceil(I.inf / (2 * np.pi))
    #
    #         a = I.inf - 2 * np.pi * k
    #         b = I.sup - 2 * np.pi * k
    #
    #         m = np.maximum(np.cos(a), np.cos(b))
    #         return np.maximum(np.sign(b), m)
    #
    #     inf = -aux_max_cos(x - np.pi)
    #     sup = aux_max_cos(x)
    #     return Interval(inf, sup)

    @staticmethod
    def cos(x: Interval):
        ind0 = (x.sup - x.inf) >= 2 * np.pi  # xsup -xinf >= 2*pi
        yinf, ysup = np.mod(x.inf, np.pi * 2), np.mod(x.sup, np.pi * 2)

        ind1 = yinf < np.pi  # yinf in R1
        ind2 = ysup < np.pi  # ysup in R1
        ind3 = np.logical_not(ind1)  # yinf in R2
        ind4 = np.logical_not(ind2)  # ysup in R2
        ind5 = yinf > ysup  # yinf > ysup
        ind6 = np.logical_not(ind5)  # yinf <= ysup

        inf, sup = x.inf, x.sup

        ind = ind3 & ind4 & ind6
        inf[ind] = np.cos(yinf[ind])
        sup[ind] = np.cos(ysup[ind])

        ind = ind3 & ind2
        inf[ind] = np.minimum(np.cos(yinf[ind]), np.cos(ysup[ind]))
        sup[ind] = 1

        ind = ind1 & ind4
        inf[ind] = -1
        sup[ind] = np.maximum(np.cos(yinf[ind]), np.cos(ysup[ind]))

        ind = ind1 & ind2 & ind6
        inf[ind] = np.cos(ysup[ind])
        sup[ind] = np.cos(yinf[ind])

        ind = ind0 | (ind1 & ind2 & ind5) | (ind3 & ind4 & ind5)
        inf[ind] = -1
        sup[ind] = 1

        return Interval(inf, sup)

    @staticmethod
    def tan(x: Interval):
        inf = np.full_like(x.inf, -np.inf)
        sup = np.full_like(x.sup, np.inf)

        tan_inf = np.tan(x.inf)
        tan_sup = np.tan(x.sup)

        ind = ((x.sup - x.inf) < np.pi) & (tan_inf <= tan_sup)
        inf[ind] = tan_inf[ind]
        sup[ind] = tan_sup[ind]

        return Interval(inf, sup)

    @staticmethod
    def cot(x: Interval):
        # TODO need check
        ind0 = (x.sup - x.inf) >= np.pi  # xsup -xinf >= pi
        zinf, zsup = np.mod(x.inf, np.pi), np.mod(x.sup, np.pi)

        inf, sup = x.inf, x.sup

        ind = zinf <= zsup
        inf[ind] = 1 / np.tan(zsup[ind])
        sup[ind] = 1 / np.tan(zinf[ind])

        ind = ind0 | (zinf > zsup)
        inf[ind] = -np.inf
        sup[ind] = np.inf

        return Interval(inf, sup)

    # =============================================== class method
    @classmethod
    def functional(cls):
        return {
            "exp": cls.exp,
            "log": cls.log,
            "sqrt": cls.sqrt,
            "arcsin": cls.arcsin,
            "arccos": cls.arccos,
            "arctan": cls.arctan,
            "sinh": cls.sinh,
            "cosh": cls.cosh,
            "tanh": cls.tanh,
            "arcsinh": cls.arcsinh,
            "arccosh": cls.arccosh,
            "arctanh": cls.arctanh,
            "sin": cls.sin,
            "cos": cls.cos,
            "tan": cls.tan,
            "sigmoid": cls.sigmoid,
        }


    @staticmethod
    def empty(s):
        inf, sup = np.full(s, np.nan, dtype=float), np.full(s, np.nan, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def rand(*shape):
        inf, sup = np.random.rand(*shape), np.random.rand(*shape)
        inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
        return Interval(inf, sup)

    @staticmethod
    def zeros(shape):
        inf, sup = np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def ones(shape):
        inf, sup = np.ones(shape, dtype=float), np.ones(shape, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def identity(shape):
        inf, sup = -np.ones(shape, dtype=float), np.ones(shape, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def concatenate(boxes, axis=None):
        """
        concatenate boxes along specified axis
        @param boxes: given list of boxes
        @param axis: axis along
        @return: concatenated boxes
        """
        infs = [box.inf for box in boxes]
        sups = [box.sup for box in boxes]
        return Interval(np.concatenate(infs, axis=axis), np.concatenate(sups, axis=axis))

    @staticmethod
    def stack(boxes, axis=0):
        """
        stack boxes along specified axis
        @param boxes: given list of boxes
        @param axis: axis along
        @return: stacked boxes
        """
        infs = [box.inf for box in boxes]
        sups = [box.sup for box in boxes]
        return Interval(np.stack(infs, axis=axis), np.stack(sups, axis=axis))

    @staticmethod
    def vstack(boxes):
        """
        works for up to 3-dimensional arrays
        @param boxes: given list of boxes
        @return: vstack of boxes
        """
        infs = [box.inf for box in boxes]
        sups = [box.sup for box in boxes]
        return Interval(np.vstack(infs), np.vstack(sups))

    @staticmethod
    def hstack(boxes):
        """
        works for up to 3-dimensional arrays
        @param boxes: given list of boxes
        @return:  hstack of boxes
        """
        infs = [box.inf for box in boxes]
        sups = [box.sup for box in boxes]
        return Interval(np.hstack(infs), np.hstack(sups))

    @staticmethod
    def split(box: Interval, indices_or_sections, axis=0):
        infs = np.split(box.inf, indices_or_sections, axis)
        sups = np.split(box.sup, indices_or_sections, axis)
        num_boxes = len(infs)
        return [Interval(infs[box_idx].squeeze(axis), sups[box_idx].squeeze(axis)) for box_idx in range(num_boxes)]

    @staticmethod
    def squeeze(box: Interval, axis=None):
        inf = np.squeeze(box.inf, axis)
        sup = np.squeeze(box.sup, axis)
        return Interval(inf, sup)

    # =============================================== public method

    def proj(self, dims):
        assert len(self.shape) == 1
        return Interval(self.inf[dims], self.sup[dims])

    def transpose(self, *axes):
        return Interval(self._inf.transpose(*axes), self._sup.transpose(*axes))

    def sum(self, axis=None):
        return Interval(self._inf.sum(axis), self._sup.sum(axis))

    def decompose(self, index):
        '''
        decompse given interval along specific indices
        @param index: specific index
        @return: decomposed intervals
        '''
        inf, sup = self.inf.copy(), self.sup.copy()
        c = (self.inf[index] + self.sup[index]) * 0.5
        inf[index] = c
        sup[index] = c
        return Interval(self.inf, sup), Interval(inf, self.sup)

    def partition(self, r: float):
        def __ll2arr(ll, fill_value: float):
            lens = [lst.shape[0] for lst in ll]
            max_len = max(lens)
            mask = np.arange(max_len) < np.array(lens)[:, None]
            arr = np.ones((len(lens), max_len, 2), dtype=float) * fill_value
            arr[mask] = np.concatenate(ll)
            return arr, mask

        def __get_seg(dim_idx: int, seg_num: int):
            if seg_num <= 1:
                return np.array([self.inf[dim_idx], self.sup[dim_idx]], dtype=float).reshape(
                    (1, -1)
                )
            else:
                samples = np.linspace(self.inf[dim_idx], self.sup[dim_idx], num=seg_num + 1)
                this_segs = np.zeros((seg_num, 2), dtype=float)
                this_segs[:, 0] = samples[:-1]
                this_segs[:, 1] = samples[1:]
                return this_segs

        assert len(self.shape) == 1
        nums = np.floor((self.sup - self.inf) / r).astype(dtype=int) + 1
        segs, _ = __ll2arr([__get_seg(i, nums[i]) for i in range(self.shape[0])], np.nan)
        idx_list = [np.arange(nums[i]) for i in range(self.shape[0])]
        ext_idx = np.array(np.meshgrid(*idx_list)).T.reshape((-1, len(idx_list)))
        aux_idx = np.tile(np.arange(self.shape[0]), ext_idx.shape[0])
        bounds = segs[aux_idx, ext_idx.reshape(-1)].reshape((-1, self.shape[0], 2))
        return [Interval(bound[:, 0], bound[:, 1]) for bound in bounds]

    def rectangle(self):
        assert len(self.shape) == 1 and self.shape[0] == 2  # enforce 2d
        pts = np.zeros((4, 2), dtype=float)
        pts[[0, 3], 0] = self.inf[0]
        pts[[1, 2], 0] = self.sup[0]
        pts[[0, 1], 1] = self.inf[1]
        pts[[2, 3], 1] = self.sup[1]
        return pts

    def union(self, xs: [Interval]):
        """
        get the union of given intervals
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def intersection(self, xs: [Interval]):
        """
        get the intersections of given intervals
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def contains(self, x):
        """
        check if given data inside the domain specified by this interval
        :param x:
        :return:
        """
        if isinstance(x, np.ndarray) or isinstance(x, (int, float)):
            return np.all(x >= self.inf) and np.all(x <= self.sup)
        elif isinstance(x, list):
            x_arr = np.atleast_1d(x).astype(float)
            return np.all(x_arr >= self.inf) and np.all(x_arr <= self.sup)
        else:
            raise NotImplementedError

    def inclusion(self, other):
        """
        Check if the current set contains 'other'
        :param other:
        :return tv
        """
        if not isinstance(other, Interval):
            raise TypeError("The element type for calculating the difference set should be 'Interval'")

        if self.shape != other.shape:
            raise ValueError("The shape of the two Intervals for calculating the difference set should be consistent")

        if any(x > y for x, y in zip(self.inf, other.inf)) or any(x < y for x, y in zip(self.sup, other.sup)):
            return False
        return True

    def difference(self, other: Interval):
        """
        Calculate the difference set between this interval and 'other'
        :param other:
        :return diff_list
        """
        if not isinstance(other, Interval):
            raise TypeError("The element type for calculating the difference set should be 'Interval'")

        if self.shape != other.shape:
            raise ValueError("The shape of the two Intervals for calculating the difference set should be consistent")

        if not self.inclusion(other):
            raise ValueError("To calculate the difference set, the other set must be included in the current set")

        if any(x == y for x, y in zip(self.inf, other.inf)) or any(x == y for x, y in zip(self.sup, other.sup)):
            raise NotImplementedError

        diff_list = []
        split_inf = self.inf.copy()
        split_sup = self.sup.copy()

        for i in range(self.shape[0]):
            interval_1 = Interval(split_inf.copy(), split_sup.copy())
            interval_2 = Interval(split_inf.copy(), split_sup.copy())

            interval_1.sup[i] = other.inf[i]
            diff_list.append(interval_1)

            interval_2.inf[i] = other.sup[i]
            diff_list.append(interval_2)

            split_inf[i] = other.inf[i]
            split_sup[i] = other.sup[i]

        return diff_list

    def sample_gird_2d(self, n):
        x = np.linspace(self.inf[0], self.sup[0], n)
        y = np.linspace(self.inf[1], self.sup[1], n)

        x_list, y_list = np.meshgrid(x, y)

        points = np.vstack([x_list.ravel(), y_list.ravel()]).T

        return points

    def sample_gird(self, m):
        n = self.shape[0]
        sampled_points = []

        for i in range(n):
            if self.inf[i] == self.sup[i]:
                sampled_points.append(np.array([self.inf[i]]))
            else:
                sampled_points.append(np.linspace(self.inf[i], self.sup[i], m))

        grid = np.meshgrid(*sampled_points)
        sampled_points_combined = np.array(grid).T.reshape(-1, n)
        unique_points = np.unique(sampled_points_combined, axis=0)

        return unique_points

    def is_in_safe_set(self, fx_data):
        return np.all((fx_data >= self.inf) & (fx_data <= self.sup), axis=1)

    def generate_data(self, N, method):
        data = np.empty((0, self._degree))
        if method == "random":
            data = np.random.uniform(self.inf, self.sup, size=(N, len(self.inf)))
        elif method == "grid":
            N_each_degree = int(np.ceil(N ** (1 / self._degree)))
            grids = [np.linspace(self._inf[i], self._sup[i], N_each_degree) for i in range(self._degree)]
            data = np.array(np.meshgrid(*grids)).T.reshape(-1, self._degree)
        else:
            raise NotImplementedError()
        return data