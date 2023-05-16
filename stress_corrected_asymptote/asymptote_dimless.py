from typing import Union, Sequence
import numpy as np
import scipy.optimize as sp_opt
import scipy.integrate as sp_int
import scipy.interpolate as sp_interp
from . import utils

FloatOrSequence = Union[float, Sequence[float]]


class WidthDimless(object):
    def __call__(self, s: FloatOrSequence, chi: float, layers_pos: FloatOrSequence, delta_sigma: FloatOrSequence):
        ss = np.atleast_1d(s)
        layers_pos = np.atleast_1d(layers_pos)
        delta_sigma = np.atleast_1d(delta_sigma)

        result = self._call_impl(ss, chi, layers_pos, delta_sigma)
        return result if not np.isscalar(s) else result.item()

    def _call_impl(self, x: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        raise NotImplementedError


class ToughnessCorrectedDimless(WidthDimless):
    def __init__(self, n_stress_iter: int = 1, tol: float = 1e-6):
        super().__init__()

        self.n_stress_iter = max(1, n_stress_iter)
        self.tol = tol

    @staticmethod
    def implicit_form(s: np.ndarray, w: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        s_hat = s / np.power(w, 3)
        k_hat = (1 + utils.g_sigma(s, layers_pos, delta_sigma)) / w
        c_hat = chi / w
        return s_hat - utils.g_delta(k_hat, c_hat)

    def _call_impl(self, s: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        # Initial approximation in the case of zero stress
        w0 = utils.width_dimless_initial_approx(s, chi)

        # Continuation method: gradually increase the amplitude of the stress
        for i in range(self.n_stress_iter):
            # Compute continuation parameter for stress
            stress_mult = (i + 1) / self.n_stress_iter

            # Compute width for the current stress_mult
            w0 = sp_opt.root(
                lambda w: self.implicit_form(s, w, chi, layers_pos, stress_mult * delta_sigma),
                x0=w0, method='lm', options={'xtol': self.tol}
            ).x

        return w0


class ODEDimless(WidthDimless):
    def __init__(self, grid_start: float = 1e-15, atol: float = 1e-10, rtol: float = 1e-6):
        super().__init__()

        self.c1 = np.power(utils.BETA_M, 3) / 3
        self.c2 = np.power(utils.BETA_MT, 4) / 4

        self.grid_start = grid_start
        self.atol = atol
        self.rtol = rtol

        self._solve_ode_vectorized = np.vectorize(self._solve_ode, excluded=['chi', 'layers_pos', 'delta_sigma'])

    def _rhs(self, s: np.ndarray, w: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        g_sigma = utils.g_sigma(s, layers_pos, delta_sigma)
        return self.c1 / np.power(w + g_sigma, 2) + chi * self.c2 / np.power(w + g_sigma, 3)

    def _solve_ode(self, s: float, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        # Solve initial value problem for w_hat
        result = sp_int.solve_ivp(
            lambda ss, w: self._rhs(ss, w, chi, layers_pos, delta_sigma),
            t_span=(self.grid_start, s), y0=[1.0], atol=self.atol, rtol=self.rtol
        )
        return result.y[0][-1]

    def _call_impl(self, s: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        w_hat = self._solve_ode_vectorized(s=s, chi=chi, layers_pos=layers_pos, delta_sigma=delta_sigma)
        return w_hat + utils.g_sigma(s, layers_pos, delta_sigma)


class IntegralDimless(WidthDimless):
    def __init__(
        self, grid_start: float = 1e-15, grid_stop: float = 1e18, grid_n: int = 1500,
        is_adaptive_grid=False, n_stress_iter: int = 1, tol: float = 1e-6
    ):
        super().__init__()

        self._grid_start = grid_start
        self._grid_stop = grid_stop
        self._grid_n = grid_n
        self._is_adaptive_grid = is_adaptive_grid
        self.n_stress_iter = max(1, n_stress_iter)
        self.tol = tol

        self._grid = None
        self._weights = None
        self._g_matrix = None

        if not self._is_adaptive_grid:
            self._update_grid(np.array([]))

    def _update_grid(self, layers_pos: np.ndarray):
        self._grid = self.simpson_grid_adaptive(self._grid_start, self._grid_stop, self._grid_n, layers_pos)
        self._weights = self.simpson_weights(self._grid)
        self._g_matrix = utils.g_kernel(np.outer(1 / self._grid, self._grid))

    @staticmethod
    def simpson_weights(grid: np.ndarray):
        h = np.diff(grid)
        omega = np.zeros_like(grid)
        n_simpson = grid.size

        # In the case of odd sub-intervals (even nodes) apply trapezoidal rule to the last sub-interval
        if grid.size % 2 == 0:
            n_simpson = grid.size - 1
            omega[-2] += 0.5 * h[-1]
            omega[-1] += 0.5 * h[-1]

        # Weights according to Simpson's rule
        slice_0 = slice(0, n_simpson - 2, 2)
        slice_1 = slice(1, n_simpson - 1, 2)
        slice_2 = slice(2, n_simpson, 2)

        h_sum = h[slice_0] + h[slice_1]
        omega[slice_0] += (h_sum / 6) * (2 - h[slice_1] / h[slice_0])
        omega[slice_1] += (h_sum / 6) * (np.power(h_sum, 2) / (h[slice_0] * h[slice_1]))
        omega[slice_2] += (h_sum / 6) * (2 - h[slice_0] / h[slice_1])
        return omega

    @staticmethod
    def simpson_grid_adaptive(
        start: float, stop: float, n_min: int, layers_pos: np.ndarray,
        thickening: float = 2, deviation_ratio: float = 0.3
    ):
        log_start = np.log10(start)
        log_stop = np.log10(stop)
        log_h_max = (log_stop - log_start) / (n_min - 1)

        # Compute variance of the mesh thickening in the vicinity of the stress layer
        sigma = np.log10((1 + deviation_ratio) / (1 - deviation_ratio)) / 6

        # Density function of the grid nodes distribution
        def rho(x: float):
            exp_term = np.sum(np.exp(-0.5 * np.power((x - np.log10(layers_pos)) / sigma, 2)))
            return 1 + thickening * np.minimum(1, exp_term)

        # Appending new nodes according to the density function
        log_grid = [log_start]
        tol = 100 * (np.nextafter(log_stop, log_stop + 1) - log_stop)
        while True:
            h_k = np.minimum(log_h_max, log_h_max / rho(log_grid[-1]))
            log_x_k = log_grid[-1] + h_k
            if log_x_k < log_stop - tol:
                log_grid.append(log_x_k)
            else:
                log_grid.append(log_stop)
                break

        result = np.power(10, log_grid)

        # Make sure the endpoints match the start and stop arguments
        result[0] = start
        result[-1] = stop

        return result

    def _nonlinear_system(self, w: np.ndarray, chi: float, g_sigma: np.ndarray):
        nonlinear_part = 1 / np.power(w, 2) + chi / np.power(w, 3)
        weights_mul_nonlinear = self._weights * nonlinear_part
        return w - 1 - (8 / np.pi) * self._g_matrix.dot(weights_mul_nonlinear) - g_sigma

    def _jacobi_matrix(self, w: np.ndarray, chi: float):
        nonlinear_derivative = -(2 / np.power(w, 3) + 3 * chi / np.power(w, 4))
        weights_mul_nonlinear = self._weights * nonlinear_derivative
        return np.eye(self._g_matrix.shape[0]) - (8 / np.pi) * self._g_matrix * weights_mul_nonlinear[None, :]

    def _call_impl(self, s: np.ndarray, chi: float, layers_pos: np.ndarray, delta_sigma: np.ndarray):
        if self._is_adaptive_grid:
            # Generate adaptive grid and auxiliary data
            self._update_grid(layers_pos)

        # Compute G_Sigma term for RHS
        g_sigma = utils.g_sigma(self._grid, layers_pos, delta_sigma)

        # Initial approximation in the case of zero stress
        w_prev = utils.width_dimless_initial_approx(self._grid, chi)

        # Continuation method: gradually increase the amplitude of the stress
        for i in range(self.n_stress_iter):
            # Compute continuation parameter for stress
            stress_mult = (i + 1) / self.n_stress_iter

            # Newton's method
            error = 1
            while error > self.tol:
                jacobi = self._jacobi_matrix(w_prev, chi)
                rhs = -self._nonlinear_system(w_prev, chi, stress_mult * g_sigma)
                w_delta = np.linalg.solve(jacobi, rhs)

                error = np.max(np.abs(np.divide(w_delta, w_prev)))
                w_prev = w_prev + w_delta

        # Interpolation at the given points
        width_result = sp_interp.interp1d(self._grid, w_prev, kind='quadratic', fill_value='extrapolate')(s)
        return width_result
