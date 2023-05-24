import numpy as np
from math import pow, sqrt


BETA_M = pow(2, 1 / 3) * pow(3, 5 / 6)
BETA_MT = 4 / pow(15 * (sqrt(2) - 1), 1 / 4)


def g_delta(kh, ch):
    # input validation
    kh = np.atleast_1d(kh) if np.isscalar(kh) else np.asarray(kh)
    ch = np.atleast_1d(ch) if np.isscalar(ch) else np.asarray(ch)
    kh[np.where(kh < 0)] = 0
    kh += np.finfo(float).eps
    kh[np.where(kh > 1)] = 1
    ch[np.where(ch < 0)] = 0

    b0 = 0.75 * pow(BETA_MT, 4) / pow(BETA_M, 3)

    # function f, solution to differential equation
    def f(x, y, z):
        return np.divide(
            1 - np.power(x, 3) - 1.5 * np.multiply(y, 1 - np.power(x, 2)) + 3 * np.multiply(np.power(y, 2), 1 - x) -
            6 * np.multiply(np.power(y, 3), np.arctanh(np.divide(1 - x, 2 * y + x + 1))),
            3 * z
        )

    # k-mt edge expanded solution Ch>>1
    def fkmt(x, y, z):
        return np.divide(
            np.divide(1 - np.power(x, 4), 4 * y) -
            np.divide(1 - np.power(x, 5), 5 * np.power(y, 2)) +
            np.divide(1 - np.power(x, 6), 6 * np.power(y, 3)),
            z
        )

    # functions C1 and C2
    def c1(x):
        return np.divide(np.multiply(np.tan(np.pi * x), 4 * (1 - 2 * x)), np.multiply(x, 1 - x))

    def c2(x):
        return np.divide(np.multiply(np.tan(1.5 * np.pi * x), 16 * (1 - 3 * x)), np.multiply(3 * x, 2 - 3 * x))

    # use k-mt edge solution for large values of Ch
    ind_large_ch = np.where(ch > 1e3)

    betam_3_3 = pow(BETA_M, 3) / 3
    delta = betam_3_3 * np.multiply(f(kh, b0 * ch, betam_3_3), 1 + b0 * ch)
    delta[ind_large_ch] = betam_3_3 * np.multiply(
        fkmt(kh[ind_large_ch], b0 * ch[ind_large_ch], betam_3_3), 1 + b0 * ch[ind_large_ch]
    )

    delta[np.where(delta <= 0)] = 1e-6
    delta[np.where(delta >= 1 / 3)] = 1 / 3 - 1e-6

    bh = np.divide(c2(delta), c1(delta))

    # delta-correction
    xh = f(kh, np.multiply(ch, bh), c1(delta))
    xh[ind_large_ch] = fkmt(kh[ind_large_ch], np.multiply(ch[ind_large_ch], bh[ind_large_ch]), c1(delta[ind_large_ch]))
    return xh


def g_kernel(t):
    """
    Integral kernel G for a dimensionless semi-infinite fracture problem:
        G(t) = ((1 - t^2) / t) * ln(|(1 + t) / (1 - t)|) + 2.
    To regularize the calculations, the following limiting relations are used:
        G(t) = 4 - (4 / 3) * t^2 for t << 1,
        G(1) = 2,
        G(t) = 4 / (3 * t^2) for t >> 1.
    """
    s = np.atleast_1d(t)

    # calculate for regular indices
    result = np.zeros(s.shape)
    reg_ind = np.where((s > 1e-2) & (s != 1) & (s < 1e4))
    result[reg_ind] = (1 / s[reg_ind] - s[reg_ind]) * np.log(np.abs((1 + s[reg_ind]) / (1 - s[reg_ind]))) + 2

    # Calculate G-kernel for small values of s
    small_ind = np.where(s <= 1e-2)
    result[small_ind] = 4 * (1 - np.power(s[small_ind], 2) / 3)

    # Regularization for s = 1
    result[s == 1] = 2

    # Apply asymptotic for huge values of s
    far_ind = np.where(s >= 1e4)
    result[far_ind] = 4 / (3 * np.power(s[far_ind], 2))

    # convert result to scalar if input is scalar
    return result if not np.isscalar(t) else result.item()


def g_sigma(s, layers_pos, delta_sigma):
    g_stress = g_kernel(np.outer(np.divide(1, s), layers_pos))
    delta_sigma_s = np.multiply(delta_sigma, layers_pos)
    return (4 / np.pi) * g_stress.dot(delta_sigma_s)


def width_dimless_initial_approx(s, chi: float):
    k_regime = np.ones_like(s)
    mt_regime = BETA_MT * np.power(chi * s, 1 / 4)
    m_regime = BETA_M * np.power(s, 1 / 3)
    return np.maximum.reduce([k_regime, mt_regime, m_regime])
