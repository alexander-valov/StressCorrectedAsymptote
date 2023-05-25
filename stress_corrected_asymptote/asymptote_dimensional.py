from typing import Sequence
import numpy as np
import scipy.optimize as sp_opt
import warnings
from .asymptote_dimless import IntegralDimless, ToughnessCorrectedDimless, ODEDimless

METHODS = ['integral', 'toughness-corrected', 'ode']


class StressCorrectedAsymptote(object):
    def __init__(self, min_velocity: float = 1e-8, method: str = METHODS[-1], options=None):
        """
        Dimensional form of the stress-corrected asymptotic solution at the vicinity of the hydraulic fracture tip.

        Parameters
        ----------
        min_velocity : float, default: 1e-8
            Threshold for the front propagation velocity.
            Details of the velocity approximation can be found in [2]_.
        method: str, default: 'ode'
            Method to approximate the problem of a semi-infinite hydraulic fracture
            propagating through multiple stress layers. Appropriate dimensionless
            solvers are `IntegralDimless`, `ToughnessCorrectedDimless`, and `ODEDimless`.
            The underlying mathematical background is described in [1]_.
        options: dict, optional
            Solver specific settings. See docs of a corresponding dimensionless solver.

        References
        ----------
        .. [1] A.V. Valov and E.V. Dontsov.
           2023. On the layer crossing problem for a semi-infinite hydraulic fracture.
        .. [2] A.V. Valov, E.V. Dontsov, A.N. Baykin, S.V. Golovin.
           2023. An implicit level set algorithm for hydraulic fracturing with a stress-layer asymptote.
        """
        self.min_velocity = min_velocity

        method = method.lower()
        if method in METHODS:
            self.method = method
        else:
            self.method = METHODS[-1]
            warnings.warn(
                f'Unknown method \'{method}\', the \'{self.method}\' is used instead. Available methods are {METHODS}',
                RuntimeWarning
            )

        if options is None:
            options = {}

        if self.method == 'integral':
            self.width_solver_dimless = IntegralDimless(**options)
        elif self.method == 'toughness-corrected':
            self.width_solver_dimless = ToughnessCorrectedDimless(**options)
        elif self.method == 'ode':
            self.width_solver_dimless = ODEDimless(**options)
        else:
            self.width_solver_dimless = NotImplemented

    def width(
        self, s: float, velocity: float, rho_layers: Sequence[float], delta_sigma: Sequence[float],
        e_prime: float, k_prime: float, c_prime: float, mu_prime: float
    ):
        """
        Computes the stress-corrected asymptote at the given distance from the fracture tip `s`.

        Implements the dimensional form of the stress corrected asymptote [1]_.
        The use of this asymptotic solution for the purpose of front tracking
        in hydraulic fracturing simulators is described in [2]_.

        Parameters
        ----------
        s : float
            Distance to the fracture tip.
        velocity : float
            Tip propagation velocity.
        rho_layers : array_like
            Signed distance from the computation point (in this case it is `s`)
            to the location of the stress layers. Note that the location of the
            stress layers is computed as ``layers_pos = s - rho_layers``. This
            method adopts `rho_layers` instead of `layers_pos` for the purpose
            of similarity to the `distance` method.
        delta_sigma : array_like
            Amplitude of the stress barriers corresponding to `rho_layers`.
        e_prime: float
            Plane-strain elasticity modulus. `e_prime` equals E / (1 - nu ** 2),
            where E is the Young's modulus and nu is the Poisson's ratio.
        k_prime: float
            Scaled fracture toughness. `k_prime` equals sqrt(32 / pi) * K_{Ic},
            where K_{Ic} is the fracture toughness.
        c_prime: float
            Scaled leak-off coefficient. `c_prime` equals 2 * C_L,
            where C_L is the Carter's leak-off coefficient.
        mu_prime: float
            Scaled fluid viscosity. `mu_prime` equals 12 * mu,
            where mu is the fluid viscosity.

        Returns
        -------
        val : float
            Stress-corrected asymptote computed at the given distance from the fracture tip `s`.

        References
        ----------
        .. [1] A.V. Valov and E.V. Dontsov.
           2023. On the layer crossing problem for a semi-infinite hydraulic fracture.
        .. [2] A.V. Valov, E.V. Dontsov, A.N. Baykin, S.V. Golovin.
           2023. An implicit level set algorithm for hydraulic fracturing with a stress-layer asymptote.
        """
        rho_layers = np.atleast_1d(rho_layers)
        delta_sigma = np.atleast_1d(delta_sigma)

        # Compute dimensionless complexes
        velocity = np.maximum(velocity, self.min_velocity)
        l_scale = np.power(k_prime * np.power(k_prime / e_prime, 2) / (mu_prime * velocity), 2)
        chi = 2 * c_prime * e_prime / (k_prime * np.sqrt(velocity))

        # Scaling
        s_dimless = np.sqrt(s / l_scale)
        layers_pos = np.maximum(0, s - rho_layers)
        layers_pos_dimless = np.sqrt(layers_pos / l_scale)
        delta_sigma_dimless = delta_sigma * np.sqrt(l_scale) / k_prime

        # Compute fracture width
        width_dimless = self.width_solver_dimless(s_dimless, chi, layers_pos_dimless, delta_sigma_dimless)
        return width_dimless * k_prime * np.sqrt(s) / e_prime

    def distance(
        self, w: float, s_old: float, dt: float, rho_layers: Sequence[float], delta_sigma: Sequence[float],
        e_prime: float, k_prime: float, c_prime: float, mu_prime: float, atol: float = 1e-4, rtol: float = 1e-2
    ):
        """
        Computes the distance to the fracture front for the given fracture width.

        Implements the dimensional form of the stress corrected asymptote [1]_.
        The use of this asymptotic solution for the purpose of front tracking
        in hydraulic fracturing simulators is described in [2]_.

        Parameters
        ----------
        w : float
            Fracture width corresponding to a required distance to the fracture front.
        s_old : float
            Distance to the fracture front from the previous time step.
            Used to approximate fracture front velocity, see [2]_.
        dt: float
            Time step. Used to approximate fracture front velocity, see [2]_.
        rho_layers : array_like
            Signed distance from the computation point to the location
            of the stress layers. Note that the location of the stress
            layers is computed as ``layers_pos = s - rho_layers``, where ``s``
            is the required distance to the fracture front. Since
            the distance to the fracture front is unknown, the method
            takes `rho_layers` instead of `layers_pos`. The discussions of
            front tracking algorithm, fracture front locating, and `rho_layers`
            can be found in [2]_.
        delta_sigma : array_like
            Amplitude of the stress barriers corresponding to `rho_layers`.
        e_prime: float
            Plane-strain elasticity modulus. `e_prime` equals E / (1 - nu ** 2),
            where E is the Young's modulus and nu is the Poisson's ratio.
        k_prime: float
            Scaled fracture toughness. `k_prime` equals sqrt(32 / pi) * K_{Ic},
            where K_{Ic} is the fracture toughness.
        c_prime: float
            Scaled leak-off coefficient. `c_prime` equals 2 * C_L,
            where C_L is the Carter's leak-off coefficient.
        mu_prime: float
            Scaled fluid viscosity. `mu_prime` equals 12 * mu,
            where mu is the fluid viscosity.
        atol, rtol : float
            Relative and absolute tolerances for the root finding. The solver keeps the local error
            estimates less than ``atol + rtol * abs(y)``.

        Returns
        -------
        val : float
            Stress-corrected asymptote computed at the given distance from the fracture tip `s`.

        References
        ----------
        .. [1] A.V. Valov and E.V. Dontsov.
           2023. On the layer crossing problem for a semi-infinite hydraulic fracture.
        .. [2] A.V. Valov, E.V. Dontsov, A.N. Baykin, S.V. Golovin.
           2023. An implicit level set algorithm for hydraulic fracturing with a stress-layer asymptote.
        """
        rho_layers = np.atleast_1d(rho_layers)
        delta_sigma = np.atleast_1d(delta_sigma)

        def dist_func(s):
            return self.width(s, (s - s_old) / dt, rho_layers, delta_sigma, e_prime, k_prime, c_prime, mu_prime) - w

        def implicit_form(s):
            # Compute dimensionless complexes
            velocity = np.maximum((s - s_old) / dt, self.min_velocity)
            l_scale = np.power(k_prime * np.power(k_prime / e_prime, 2) / (mu_prime * velocity), 2)
            chi = 2 * c_prime * e_prime / (k_prime * np.sqrt(velocity))

            # Scaling
            s_dimless = np.sqrt(s / l_scale)
            w_dimless = e_prime * w / (k_prime * np.sqrt(s))
            layers_pos = np.maximum(0, s - rho_layers)
            layers_pos_dimless = np.sqrt(layers_pos / l_scale)
            delta_sigma_dimless = delta_sigma * np.sqrt(l_scale) / k_prime

            # Compute the implicit form of the toughness-corrected asymptote
            return ToughnessCorrectedDimless.implicit_form(
                s_dimless, w_dimless, chi, layers_pos_dimless, delta_sigma_dimless
            ).item()

        # In the case of the toughness-corrected asymptote solve directly the implicit form equation
        if self.method == 'toughness-corrected':
            dist_func = implicit_form

        result = s_old
        if dist_func(s_old) < 0:
            borders = np.append(s_old, rho_layers[rho_layers > s_old])
            s_min = s_old
            s_max = 10 * borders[-1]

            # Extends max boundary if needed
            while dist_func(s_max) < 0:
                s_max *= 2

            # Root localization
            borders = rho_layers[rho_layers > s_old]
            for border in borders:
                if dist_func(border) >= 0:
                    s_max = border
                    break
                else:
                    s_min = border

            # Find root
            result = sp_opt.brentq(dist_func, s_min, s_max, xtol=atol, rtol=rtol, full_output=False)

        return result
